import torch
import numpy as np
import os
import argparse
import _init_paths

from pathlib import Path

from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.dataloader.Dataloader import getDataloader
from lib.model.PtnetUnet import PtnetUnetEarly
from lib.utils.loss import PointNetLoss
from lib.utils.utils import adjust_learning_rate, get_confusion_matrix

from lib.utils.utils import load_cfg


def main():
    parser = argparse.ArgumentParser(description='2DGuided3D')
    parser.add_argument('--config', type=str, default='../../config/scannet_SegformerB2Ptnet.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None

    config = load_cfg(args.config)

    if config["TRAIN.gpu"]:
        device = torch.device("cuda:" + config["TRAIN.gpu"][0])
    else:
        device = torch.device("cpu")

    net = PtnetUnetEarly(config["NETWORK.nb_class"], config["NETWORK.ptnet_feat"], device,
                         config["DATA.half_precision"])
    net.to(device)

    train_loader, eval_loader, _ = getDataloader(config)
    training(net, device, train_loader, eval_loader, config)


def training(net, device, train_loader, eval_loader, config):
    """
        Args:
            net (torch.nn.Module):           The network.
            device (torch.device):           The network and data's device.
            train_loader (torch.Dataloader): The dataloader for training set.
            eval_loader (torch.Dataloader):  The dataloader for evaluation set.
            config (dict):                   The config dict.
    """

    criterion = nn.CrossEntropyLoss(ignore_index=config["TRAIN.ignore_label"])
    optimizer = optim.SGD(net.parameters(), lr=config["TRAIN.lr"], momentum=config["TRAIN.momentum"],
                          weight_decay=config["TRAIN.weight_decay"])

    output_path = config["DATA.save_dir"] + config["DATA.output_name"]
    Path(output_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_path + "/runs/" + config["DATA.output_name"])

    model_state_file = config["TRAIN.checkpoint"]
    starting_epoch = 0

    if os.path.isfile(model_state_file):
        print("Load previous training")
        checkpoint = torch.load(model_state_file, map_location=device)
        starting_epoch = checkpoint['epoch']
        dct = checkpoint['state_dict']

        net.load_state_dict(dct)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Log epoch starting => ", starting_epoch)

    print("Start learning")
    lr = config["TRAIN.lr"]
    for epoch in range(starting_epoch, config["TRAIN.epoch_number"]):

        if epoch!=0 and config["TRAIN.div_rate"]!=0 and config["TRAIN.div_rate"]%epoch == 0:
            lr *= config["TRAIN.div_factor"]
        # train
        train_loss = train(net, device, criterion, epoch, train_loader, optimizer, config, lr)
        writer.add_scalar("loss", train_loss, epoch)

        # validate
        val_loss, mean_IoU, class_IoU = validate(net, device, criterion, eval_loader, config, epoch, output_path)
        writer.add_scalar("validation_loss", val_loss, epoch)
        writer.add_scalar("mean_IoU", mean_IoU, epoch)

        if epoch % config["TRAIN.save_rate"] == 0:
            Path(output_path + '/train_state/').mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_path, 'train_state/epoch_' + str(epoch) + '_checkpoint.pth.tar'))

        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, os.path.join(output_path, 'last_epoch_checkpoint.pth.tar'))

        print("epoch number   train_loss   val_loss")
        print(epoch, "   ", train_loss, "   ", val_loss)
        print("class score :")
        print(class_IoU)
        print("mean_IoU :")
        print(mean_IoU)

    writer.flush()
    writer.close()


def train(net, device, criterion, epoch, train_loader, optimizer, config, lr):
    """
        Args:
            net (torch.nn.Module):           The network.
            device (torch.device):           The network and data's device.
            criterion (torch.nn.Loss):       The training loss.
            epoch (int):                     The current epoch number.
            train_loader (torch.Dataloader): The dataloader for training set.
            optimizer (torch.optim):         The training optimizer.
            config (dict):                   The config dict.
            lr (float):                      The current learning rate.
        Returns:
            float: The epoch training loss.
    """
    net.train()

    nb_iter_epoch = len(train_loader)
    nb_iter = nb_iter_epoch * config["TRAIN.epoch_number"]
    start_iter = epoch * nb_iter_epoch

    running_loss = 0.
    epoch_loss = 0.

    print("[Curr_epoch/total][curr_iter/total], loss:    lr:    ")
    scaler = torch.cuda.amp.GradScaler()
    for nb_batch, data in enumerate(train_loader):

        inputs, labels, names, proj, points, maps, nb_pts = data[0].to(device), data[1].to(device), data[2], \
                                                            data[3].to(device), data[4].to(device), data[5].to(device), \
                                                            data[6]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds, ptnet_transf = net(inputs, points, proj, maps, nb_pts)
            loss = PointNetLoss(preds, labels, criterion, ptnet_transf, device)
            running_loss += loss.item()
            epoch_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr = adjust_learning_rate(optimizer,
                                  lr,
                                  nb_iter,
                                  nb_batch + start_iter)

        if nb_batch % config["TRAIN.print_rate"] == (config["TRAIN.print_rate"] - 1):
            print('[', epoch + 1, '/', config["TRAIN.epoch_number"], '][', nb_batch, '/',
                  int(nb_iter_epoch / inputs.shape[0]), '], loss: ',
                  running_loss / config["TRAIN.print_rate"], '  lr: ', lr)
            running_loss = 0.0

    return epoch_loss / nb_batch


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

            if nb_batch % config["TRAIN.print_rate"] == (config["TRAIN.print_rate"] - 1):
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
