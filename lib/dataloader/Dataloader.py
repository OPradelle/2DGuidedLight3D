import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from numpy.linalg import inv
import random
import cv2

from lib.utils.label_map import convert_label, ScannetLabel

# mean and std for the ImageNet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def random_mirror(rgb, gt, proj):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        proj = cv2.flip(proj, 1)
    return rgb, gt, proj


class FrustrumDataloader(Dataset):
    def __init__(self, list_files, img_size, ignore_label, test=False):
        self.files = list_files
        self.img_size = img_size
        self.label_mapping = ScannetLabel.label_map_Nyu40
        self.label_mapping.update({0: ignore_label})

        self.t = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.test = test

    def getImg(self, index):
        """
            Load RGB and Ground thruth images.
            Args:
                index (int):    The file id.
            Returns:
                (torch.tensor): The RGB image. (3xHxW)
                (torch.tensor): The Ground truth image. (HxW)
                (str):          The image name.
        """
        img = Image.open(self.files[index][0])
        img = img.crop((12, 12, img.size[0] - 12, img.size[1] - 12)).resize(
            self.img_size)  # remove black edging comming from the image distortion rectification.

        label = Image.open(self.files[index][1])
        label = label.crop((12, 12, label.size[0] - 12, label.size[1] - 12)).resize(self.img_size,
                                                                                    resample=Image.NEAREST)

        filename = self.files[index][0].split("color/")
        filename = filename[0].split("scans/")[1] + filename[1]

        label = np.array(label, dtype=np.int_)
        label = convert_label(label, self.label_mapping)

        img = self.t(img)
        label = torch.from_numpy(np.array(label, dtype=np.int_))

        return img, label, filename

    def getProjNpz(self, index, filename):
        """
            Load projection image.
            Args:
                index (int):    The file id.
                filename (str): The img name.
            Returns:
                (ndarray): The projection image. (HxW)
        """
        name = filename.split("/")[1].split(".")[0]
        path = self.files[index][2]
        array = np.load(path)       # (WxH array)

        idProj = torch.from_numpy(array[name])
        idProj = idProj.permute(1, 0)

        return idProj

    def invertMatrix3x4(self, matrix):
        """
            Args:
                matrix (ndarray): The camera's pose. (3x4)
            Returns:
                (ndarray): The rotation matrix. (3x3)
                (ndarray): The transpose vector. (3)
        """
        rot = inv(matrix[:3, :3])
        x = (rot[0, 0] * matrix[0, 3]) + (rot[0, 1] * matrix[1, 3]) + (rot[0, 2] * matrix[2, 3])
        y = (rot[1, 0] * matrix[0, 3]) + (rot[1, 1] * matrix[1, 3]) + (rot[1, 2] * matrix[2, 3])
        z = (rot[2, 0] * matrix[0, 3]) + (rot[2, 1] * matrix[1, 3]) + (rot[2, 2] * matrix[2, 3])
        t = [-x, -y, -z]
        return rot, t

    def getCloudFrustrum(self, index, idProj, name):
        """
            Args:
                index (int):            The file id.
                idProj (tensor.tensor): The projection image.
                name (str):             The file name.
            Returns:
                (torch.tensor): The frustrum's points. (N x Fpts)
                (torch.tensor): The Ground thruth image. (N)
        """
        cloud_path = self.files[index][3]
        cloud = np.load(cloud_path)['pts'][:, :3]

        # project cloud to camera coordinate system
        pose = np.loadtxt(self.files[index][4])
        rot, t = self.invertMatrix3x4(pose)

        frustrum_ids = np.load(self.files[index][5])[name.split("/")[1].split(".")[0]]
        pts_frustrum = cloud[frustrum_ids]

        pts_frustrum = np.einsum('mk,ik->im', rot, pts_frustrum)
        pts_frustrum = pts_frustrum + t

        id_proj = idProj[idProj != -1]
        map = np.searchsorted(frustrum_ids, id_proj)

        return torch.from_numpy(pts_frustrum).type(torch.float32), torch.from_numpy(map)

    def __getitem__(self, index):
        """
            Args:
                index (int): The file id.
            Returns:
                (torch.tensor): The RGB image.
                (torch.tensor): The Ground thruth image.
                (str):          The image name.
                (torch.tensor): The projection image.
                (torch.tensor): The frustrum's cloud.
                (torch.tensor): The map between frustrum's points and projected ones.
        """
        img, label, filename = self.getImg(index)
        idProj = self.getProjNpz(index, filename)

        cloud, map = self.getCloudFrustrum(index, idProj, filename)

        if self.test:
            return img, label, filename, idProj, cloud, map

        img, label, idProj = random_mirror(img.permute(1,2,0).numpy(), label.numpy(), idProj.numpy())

        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(label).type(torch.long), filename, torch.from_numpy(idProj), cloud, map

    def __len__(self):
        return len(self.files)


def readSetFromFiles(train_list_file, eval_list_file, test_list_file, data_path, img_size):
    """
        Args:
            train_list_file (str): The path to the training list file.
            eval_list_file (str):  The path to the evaluation list file.
            test_list_file (str):  The path to the test list file.
            data_path (str):       The data parent folder.
            img_size (str):        The image size.
        Returns:
            [(str)]: The set of files for each training's images.
                     (RGB path, GT path, proj path, cloud path, camera path, frustrum's point ids)
            [(str)]: The set of files for each evaluation's images.
                     (RGB path, GT path, proj path, cloud path, camera path, frustrum's point ids)
            [(str)]: The set of files for each testing's images.
                     (RGB path, GT path, proj path, cloud path, camera path, frustrum's point ids)
    """
    train_list = []
    eval_list = []
    test_list = []

    if train_list_file:
        file = open(train_list_file, "r")
        train = file.readlines()
        file.close()
        for line in train:
            path = line.split("\n")[0]
            rgb = data_path + path
            label = rgb.replace("color", "label").replace(".jpg", ".png")
            proj = rgb.split("/color")[0] + "/proj2sr_" + img_size + ".npz"
            cloud = rgb.split("/color")[0] + "/cloud_rgbgt.npz"
            cam = rgb.replace("color", "cam/pose").replace(".jpg", ".txt")
            frustrum = rgb.split("/color")[0] + "/fustrum2sr_" + img_size + ".npz"

            train_list.append((rgb, label, proj, cloud, cam, frustrum))

    if eval_list_file:
        file = open(eval_list_file, "r")
        eval = file.readlines()
        file.close()
        for line in eval:
            path = line.split("\n")[0]
            name = path.split("color/")[1].split(".")[0]
            if int(name)%100 != 0:
                continue

            rgb = data_path + path
            label = rgb.replace("color", "label").replace(".jpg", ".png")
            proj = rgb.split("/color")[0] + "/proj2sr_" + img_size + ".npz"
            cloud = rgb.split("/color")[0] + "/cloud_rgbgt.npz"
            cam = rgb.replace("color", "cam/pose").replace(".jpg", ".txt")
            frustrum = rgb.split("/color")[0] + "/fustrum2sr_" + img_size + ".npz"

            eval_list.append((rgb, label, proj, cloud, cam, frustrum))

    if test_list_file:
        file = open(test_list_file, "r")
        test = file.readlines()
        file.close()
        for line in test:
            path = line.split("\n")[0]
            name = path.split("color/")[1].split(".")[0]
            if int(name)%100 != 0:
                continue

            rgb = data_path + path
            label = rgb.replace("color", "label").replace(".jpg", ".png")
            proj = rgb.split("/color")[0] + "/proj2sr_" + img_size + ".npz"
            cloud = rgb.split("/color")[0] + "/cloud_rgbgt.npz"
            cam = rgb.replace("color", "cam/pose").replace(".jpg", ".txt")
            frustrum = rgb.split("/color")[0] + "/fustrum2sr_" + img_size + ".npz"

            test_list.append((rgb, label, proj, cloud, cam, frustrum))

    return train_list, eval_list, test_list


def getDataloader(cfg):
    """
        Args:
            config (dict): The config dict.
        Returns:
            (torch.Dataloader): The dataloader for training set.
            (torch.Dataloader): The dataloader for evaluation set.
            (torch.Dataloader): The dataloader for test set.
    """
    train_set, eval_set, test_set = readSetFromFiles(cfg["TRAIN.train_file"], cfg["TRAIN.eval_file"],
                                                     cfg["TEST.test_file"], cfg["DATA.data_root"], cfg["DATA.img_size"])

    img_size = cfg["TRAIN.img_size"]
    ignore_label = cfg["TRAIN.ignore_label"]
    batch_size = cfg["TRAIN.batch_size"]

    dataset_training = FrustrumDataloader(train_set, img_size, ignore_label, cfg["DATA.half_precision"])

    train_loader = torch.utils.data.DataLoader(dataset=dataset_training,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collateFn,
                                               drop_last=True,
                                               num_workers=cfg["TRAIN.num_workers"])

    dataset_eval = FrustrumDataloader(eval_set, img_size, ignore_label, True)
    eval_loader = torch.utils.data.DataLoader(dataset=dataset_eval,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collateFn,
                                              drop_last=True,
                                              num_workers=cfg["EVAL.num_workers"])

    dataset_test = FrustrumDataloader(test_set, cfg["TEST.img_size"], cfg["TEST.ignore_label"], True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                              batch_size=cfg["TEST.batch_size"],
                                              shuffle=False,
                                              collate_fn=collateFn,
                                              drop_last=True,
                                              num_workers=cfg["TEST.num_workers"])

    return train_loader, eval_loader, test_loader

def paddCloud(tensor, max_len):
    """
        Args:
            config (dict): The config dict.
        Returns:
            (torch.Dataloader): The dataloader for training set.
            (torch.Dataloader): The dataloader for evaluation set.
            (torch.Dataloader): The dataloader for test set.
    """
    pad_size = max_len - tensor.shape[0]
    padding = torch.zeros((pad_size, tensor.shape[1]), dtype=tensor.dtype)

    padded = torch.cat([tensor, padding], dim=0)

    return padded, torch.cat([torch.ones(tensor.shape[0]), torch.zeros(pad_size)])


def paddMap(tensor, max_len):
    pad_size = max_len - tensor.shape[0]
    padding = torch.full([pad_size], -1, dtype=torch.long)

    padded = torch.cat([tensor, padding], dim=0)

    return padded, torch.cat([torch.ones(tensor.shape[0]), torch.zeros(pad_size)])


def collateFn(data):
    # Find the largest batch
    (
        rgb,  # RGB images (CxHxW)
        label,  # Ground truth images (HxW)
        filename,  # image's names
        proj,  # projection's images
        clouds,  # frustrum's clouds
        maps  # map frustrum's point -> projected point
    ) = zip(*data)

    len_cloud = max([x.shape[0] for x in clouds])
    clouds = list(clouds)

    len_map = max([x.shape[0] for x in maps])
    maps = list(maps)

    nb_pts = []
    for id, (cloud_b, map_b) in enumerate(zip(clouds, maps)):
        padded, patch_mask = paddCloud(cloud_b, len_cloud)
        nb_pts.append(sum(patch_mask))
        clouds[id] = padded

        padded, patch_mask = paddMap(map_b, len_map)
        nb_pts.append(sum(patch_mask))
        maps[id] = padded

    batch = [
        torch.stack(rgb),
        torch.stack(label),
        filename,
        torch.stack(proj),
        torch.stack(clouds),
        torch.stack(maps),
        np.stack(nb_pts).astype(int)
    ]

    return batch
