import torch
import torch.nn as nn
import torch.nn.functional as F


class STNkd(nn.Module):
    """
        The TNet from PointNet architecture (https://arxiv.org/abs/1612.00593).
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).flatten().repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    """
        The PointNet features extractor architecture (https://arxiv.org/abs/1612.00593).
    """
    def __init__(self, nb_feature=3, input_transform=True, feature_transform=False, out_feat=False):
        super(PointNetfeat, self).__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.output_feature = out_feat

        if self.input_transform:
            self.stn = STNkd(k=3)

        if self.feature_transform:
            self.fstn = STNkd(k=64)

        self.block1 = nn.Sequential(nn.Conv1d(nb_feature, 64, 1, bias=True),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=True),
                                    nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=True))

    def forward(self, x, mask=None):
        nb_pts = x.shape[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = self.block1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        pt_feat = x

        x = self.block2(x)
        x = self.block3(x)

        # REPLACE MASKED POINT FEATURES WITH SMALL VALUES => will not be selected by max pool
        if mask is not None:
            x = x.permute(0, 2, 1)
            x[~mask] = -1e+4
            x = x.permute(0, 2, 1)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if not self.output_feature:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, nb_pts)
            return torch.cat([pt_feat, x], 1), trans, trans_feat


class PointNet(nn.Module):
    """
        The PointNet segmentation head (https://arxiv.org/abs/1612.00593).
    """
    def __init__(self, out_feat):
        super(PointNet, self).__init__()
        self.pointNet = PointNetfeat(3, input_transform=False, feature_transform=False, out_feat=True)
        self.out_feat = out_feat

        self.seg_head1 = nn.Sequential(nn.Conv1d(1088, 512, 1, bias=True),
                                       nn.ReLU())

        self.seg_head2 = nn.Sequential(nn.Conv1d(512, 256, 1, bias=True),
                                       nn.ReLU())

        self.seg_head3 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=True),
                                       nn.ReLU())

        self.out = nn.Conv1d(128, self.out_feat, 1)

    def forward(self, x, mask_pts):
        """
            Args:
                x (torch.tensor):        The clouds. (B x Fpts x N)
                mask_pts (torch.tensor): The mask of padded cloud. (B x N )

            Returns:
                torch.tensor:         The clouds features. (B x N x Fout)
                torch.tensor or None: The ptnet features rotation. (B x 64 x 64)
        """
        x, trans_input, trans_feat = self.pointNet(x, mask_pts)

        x = self.seg_head1(x)
        x = self.seg_head2(x)
        x = self.seg_head3(x)
        x = self.out(x)

        return x.permute(0, 2, 1), trans_feat
