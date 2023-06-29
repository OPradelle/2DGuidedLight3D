import torch
import torch.nn as nn

import torch.nn.functional as F
from model.PointNet import PointNet
from model.Segformer import mit_b2

class PtnetSegformerEarly(nn.Module):

    def __init__(self, n_classes, out_feat, device, dtype=torch.float32):
        super().__init__()
        self.pointNet = PointNet(out_feat)

        self.up_2d = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.merge_inputs = nn.Sequential(nn.Conv1d(out_feat+3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.segformer = mit_b2()
        model_state = self.segformer.state_dict()

        pretrained = torch.load("../../weights/pretrained/mitb2.pth")["state_dict"]
        dct = {}
        for elm in pretrained:
            if "patch_embed1.proj.weight" in elm:
                a = pretrained[elm]
                b = torch.cat([torch.repeat_interleave(a, 21, 1), a[:, 0, :, :].unsqueeze(1)], dim=1)
                dct[elm[9:]] = b
                continue
            if "backbone" in elm:
                dct[elm[9:]] = pretrained[elm]
        model_state.update(dct)
        self.segformer.load_state_dict(model_state)


        up_blocks = [
            nn.Sequential(nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(320), nn.ReLU()),
            nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()),
        ]
        self.up_blocks = nn.ModuleList(up_blocks)

        merge_blocks = [
            nn.Sequential(nn.Conv2d(640, 320, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(320), nn.ReLU(),
                          nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(320)),
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                          nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128)),
            nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64))
        ]

        self.merge_blocks = nn.ModuleList(merge_blocks)

        self.dropout = nn.Dropout2d(p=0.9)
        self.out = nn.Conv2d(64, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.device = device
        self.d_type = dtype

    def merge_img(self, rgbs, pts_img, projs):
        """
            Args:
                rgbs (torch.tensor):   The RGB images. (B x Fimg x H x W)
                pts_img (torch.tensor): The image's frsutrum clouds. (B x N x Fpts)
                projs (torch.tensor):  The projection images. (B x H x W)
            Returns:
                torch.tensor:         The merged image. (B x 64 x H x W)
        """
        merge_img = torch.zeros((rgbs.shape[0], pts_img.shape[1], pts_img.shape[2], rgbs.shape[3] + pts_img.shape[3]), device=self.device, dtype=self.d_type)
        m = projs != -1

        rgb_px = rgbs[m]
        pts_px = pts_img[m]

        merge_img[m] = self.merge_inputs(torch.cat([rgb_px, pts_px], dim=1).permute(1, 0).unsqueeze(0)).squeeze().permute(1,0)
        merge_img[~m] = self.up_2d(rgbs[~m].permute(1, 0).unsqueeze(0)).squeeze().permute(1,0)

        return merge_img.permute(0, 3, 1, 2)

    def forward(self, rgbs, clouds, projs, maps, nb_pts):
        """
            Args:
                rgbs (torch.tensor):   The RGB images. (B x Fimg x H x W)
                clouds (torch.tensor): The image's frsutrum clouds. (B x N x Fpts)
                projs (torch.tensor):  The projection images. (B x H x W)
                maps (torch.tensor):   The mapping between clouds and image projection. (B x N)
                                       -1 value indicate padding.
                nb_pts (ndarray long): The number of points in the frustrum. (B)
            Returns:
                torch.tensor:         The image predicted by the network. (B x C x H x W)
                torch.tensor or None: The ptnet features rotation. (B x 64 x 64)
        """

        mask_pts = torch.zeros((clouds.shape[0], clouds.shape[1]), device=self.device, dtype=torch.bool)
        mask_map = torch.zeros((maps.shape[0], maps.shape[1]), device=self.device, dtype=torch.bool)

        id_b = 0
        for i in range(0, len(nb_pts), 2):
            mask_pts[id_b, :nb_pts[i]] = True
            mask_map[id_b, :nb_pts[i+1]] = True
            id_b += 1

        pts_feat, ptnet_transf = self.pointNet(clouds.permute(0, 2, 1), mask_pts)

        img_feat = torch.zeros((rgbs.shape[0], rgbs.shape[2], rgbs.shape[3], pts_feat.shape[2]), device=self.device, dtype=self.d_type)
        for id_b, proj in enumerate(projs):
            m_proj = proj != -1
            pts_f = pts_feat[id_b]
            img_feat[id_b, m_proj.squeeze()] = pts_f[maps[id_b, mask_map[id_b]]]

        merged_feat = self.merge_img(rgbs.permute(0, 2, 3, 1), img_feat, projs)
        out = self.segformer(merged_feat)

        x = out[-1]
        for i, block in enumerate(self.up_blocks, 1):
            skip = out[len(out)-1-i]
            up_x = F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=True)
            up_x = block(up_x)

            x = torch.cat([up_x, skip], dim=1)
            x = self.merge_blocks[i-1](x)

        x = F.interpolate(x, merged_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, merged_feat], dim=1)
        x = self.merge_blocks[-1](x)
        x = self.out(x)

        return x, ptnet_transf
