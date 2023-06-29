import torch
import numpy as np
import argparse
from pathlib import Path


from lib.dataloader.Dataloader import getDataloader
from lib.model.PtnetUnet import PtnetUnetEarly
from PIL import Image

from lib.utils.label_map import convert_label
from lib.utils.utils import load_cfg

SCANNET_COLOR_MAP = [
    (174., 199., 232.),
    (152., 223., 138.),
    (31., 119., 180.),
    (255., 187., 120.),
    (188., 189., 34.),
    (140., 86., 75.),
    (255., 152., 150.),
    (214., 39., 40.),
    (197., 176., 213.),
    (148., 103., 189.),
    (196., 156., 148.),
    (23., 190., 207.),
    (247., 182., 210.),
    (219., 219., 141.),
    (255., 127., 14.),
    (158., 218., 229.),
    (44., 160., 44.),
    (112., 128., 144.),
    (227., 119., 194.),
    (82., 84., 163.)
]


def pred2Img(pred):
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for id in range(20):
        m = pred.squeeze() == id
        colors = np.array(SCANNET_COLOR_MAP[id])
        img[m] = colors

    img = Image.fromarray(img)
    return img


label_map_Nyu40 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 13: 255, 14: 12, 15: 255, 16: 13, 17: 255, 18: 255, 19: 255,
    20: 255, 21: 255, 22: 255, 23: 255, 24: 14, 25: 255, 26: 255, 27: 255, 28: 15, 29: 255,
    30: 255, 31: 255, 32: 255, 33: 16, 34: 17, 35: 255, 36: 18, 37: 255, 38: 255, 39: 19,
    40: 255
}


def test(net, eval_loader, output_path, colored_output=False):
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            inputs, names, proj, points, map, nb_pts = data[0], data[1], data[2], data[3], data[4], data[5]

            outputs, _ = net(inputs, points, proj, map, nb_pts)
            for id in range(outputs.shape[0]):
                _, pred = torch.max(outputs[id], dim=0)

                pred = convert_label(pred.numpy(), label_map_Nyu40, True)

                if colored_output:
                    out_img = pred2Img(pred)
                else:
                    out_img = Image.fromarray(np.uint8(pred))

                out_img.save(output_path + names[id].split("/")[0] + "_" +
                             names[id].split("/")[1].split(".")[0] + ".png")
    exit(0)


def main():
    parser = argparse.ArgumentParser(description='2DGuided3D')
    parser.add_argument('--config', type=str, default='../../config/scannet_Unet34Ptnet.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None

    config = load_cfg(args.config)

    if config["TEST.gpu"]:
        device = torch.device("cuda:" + config["TRAIN.gpu"][0])
    else:
        device = torch.device("cpu")

    net = PtnetUnetEarly(config["NETWORK.nb_class"], config["NETWORK.ptnet_feat"], device,
                         config["DATA.half_precision"])
    pretrained = torch.load(config["TEST.test_file"])['state_dict']
    net.load_state_dict(pretrained)

    output_path = config["DATA.save_dir"] + config["DATA.output_name"]
    Path(output_path).mkdir(parents=True, exist_ok=True)

    output_path = output_path + "/test_output/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    _, _, test_loader = getDataloader(config)

    test(net, test_loader, output_path)
