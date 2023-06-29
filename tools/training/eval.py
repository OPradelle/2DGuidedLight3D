import torch
import numpy as np
import argparse
import _init_paths

from pathlib import Path

from torch import nn

from lib.dataloader.Dataloader import getDataloader
from lib.model.PtnetUnet import PtnetUnetEarly
from lib.utils.loss import PointNetLoss
from lib.utils.utils import get_confusion_matrix

from lib.utils.utils import load_cfg


def main():
    parser = argparse.ArgumentParser(description='2DGuided3D')
    parser.add_argument('--config', type=str, default='../../config/scannet_Unet34Ptnet.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None

    config = load_cfg(args.config)

    if config["TRAIN.gpu"] != "cpu":
        device = torch.device("cuda:" + config["TRAIN.gpu"][0])
    else:
        device = torch.device("cpu")

    net = PtnetUnetEarly(config["NETWORK.nb_class"], config["NETWORK.ptnet_feat"], device,
                         config["DATA.half_precision"])
    net.to(device)

    _, eval_loader, _ = getDataloader(config)
    eval(net, device, eval_loader, config)


def eval(net, device, eval_loader, config):
    """
        Args:
            net (torch.nn.Module):           The network.
            device (torch.device):           The network and data's device.
            eval_loader (torch.Dataloader):  The dataloader for evaluation set.
            config (dict):                   The config dict.
    """

    criterion = nn.CrossEntropyLoss(ignore_index=config["TRAIN.ignore_label"])

    output_path = config["DATA.save_dir"] + config["DATA.output_name"]
    Path(output_path).mkdir(parents=True, exist_ok=True)

    model_state_file = config["EVAL.checkpoint"]
    checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
    epoch = checkpoint['epoch']
    dct = checkpoint['state_dict']
    net.load_state_dict(dct)

    # validate
    val_loss, mean_IoU, class_IoU = validate(net, device, criterion, eval_loader, config, epoch, output_path)

    print("epoch number   val_loss")
    print(epoch, "   ", val_loss)
    print("class score :")
    print(class_IoU)
    print("mean_IoU :")
    print(mean_IoU)


def validate(net, device, criterion, eval_loader, config, curr_epoch, out_path):
    """
        Args:
            net (torch.nn.Module):          The network.
            device (torch.device):          The network and data's device.
            criterion (torch.nn.Loss):      The training loss.
            eval_loader (torch.Dataloader): The dataloader for evaluation set.
            config (dict):                  The config dict.
            curr_epoch (int):               The current epoch.
            out_path (str):                 The output path.
        Returns:
            float:         The epoch evaluation loss.
            float:         The epoch mean Intersection over Union.
            ndarray float: The class mean Intersection over Union.
    """

    net.eval()
    val_loss = 0.

    confusion_matrix = np.zeros((config["NETWORK.nb_class"], config["NETWORK.nb_class"]))

    Path(out_path + '/iou/').mkdir(parents=True, exist_ok=True)

    logger = open(out_path + '/iou/' + str(curr_epoch) + '_IoUArray.txt', "w")
    with torch.no_grad():
        for nb_batch, data in enumerate(eval_loader):
            inputs, labels, names, proj, points, maps, nb_pts = data[0].to(device), data[1].to(device), data[2], \
                                                            data[3].to(device), data[4].to(device), data[5].to(device), \
                                                            data[6]

            with torch.cuda.amp.autocast():
                preds, ptnet_transf = net(inputs, points, proj, maps, nb_pts)
                loss = PointNetLoss(preds, labels, criterion, ptnet_transf, device)
                val_loss += loss.item()

            for id_pred, pred in enumerate(preds):
                c, w, h = pred.shape
                pred = pred.view(1, c, w, h)

                w, h = labels[id_pred].shape
                label = labels[id_pred].view(1, w, h)

                img_matrix = get_confusion_matrix(label,
                                                  pred,
                                                  label.shape,
                                                  config["NETWORK.nb_class"],
                                                  config["TRAIN.ignore_label"])

                pos = img_matrix[...].sum(1)
                res = img_matrix[...].sum(0)
                tp = np.diag(img_matrix[...])
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                logger.write(names[id_pred] + " : ")
                for elm in IoU_array:
                    logger.write(str(elm) + ' ')
                logger.write('\n')

                confusion_matrix += img_matrix

            if nb_batch % config["EVAL.print_rate"] == (config["EVAL.print_rate"] - 1):
                print(nb_batch, " / ", int(len(eval_loader) / inputs.shape[0]))

    pos = confusion_matrix[...].sum(1)
    res = confusion_matrix[...].sum(0)
    tp = np.diag(confusion_matrix[...])
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    for elm in IoU_array:
        logger.write(str(elm) + ' ')
    logger.close()

    return val_loss / nb_batch, mean_IoU, IoU_array


if __name__ == '__main__':
    main()
