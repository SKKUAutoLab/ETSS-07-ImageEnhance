import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.utils
from PIL import Image
from torch.autograd import Variable

from dataloader.create_data import CreateDataset
from model.model import Finetunemodel
from utils.utils import sequential_judgment

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

root_dir = os.path.abspath("../")
sys.path.append(root_dir)

parser = argparse.ArgumentParser("ZERO-TIG")
parser.add_argument("--lowlight_images_path", type=str, default="./data",                    help="location of the data corpus")
parser.add_argument("--save",                 type=str, default="./results/",                help="location of the data corpus")
parser.add_argument("--model_pretrain",       type=str, default="./weights/BVI-RLV.pt",      help="location of the data corpus")
parser.add_argument("--gpu",                  type=int, default=0,                           help="gpu device id")
parser.add_argument("--seed",                 type=int, default=2,                           help="random seed")
parser.add_argument("--raft_model",           type=str, default="./weights/raft-sintel.pth", help="path to pre-trained raft model")
parser.add_argument("--of_scale",             type=int, default=3,                           help="downscale size when compute OF")
parser.add_argument("--dataset",              type=str, default="RLV",                       help="Specified data set")
args = parser.parse_args()

save_path = args.save
os.makedirs(save_path, exist_ok=True)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
mertic = logging.FileHandler(os.path.join(args.save, "log.txt"))
mertic.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(mertic)

logging.info("train file name = %s", os.path.split(__file__))
TestDataset = CreateDataset(args, task="test")
test_queue  = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
print("Total image number: ", str(TestDataset.__len__()))

logging.info("Model path = %s", str(args.model_pretrain))


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def cvt_array2tensor(arr):
    data = torch.from_numpy(arr).float()
    data = data.permute(2, 0, 1)
    data = (data - 0.5) * 2
    data = data.to(device).unsqueeze(0)
    return data


def main():
    model = Finetunemodel(args)
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for i, (input, img_name, img_path, last_img_path) in enumerate(test_queue):
            model.is_new_seq = sequential_judgment(img_path[0], last_img_path[0])
            if model.is_new_seq:
                print("Eval Get this img from: ", img_path, "\n Last img from: ", last_img_path)

            input = Variable(input).to(device)
            enhance, output, illum = model(input)

            if "RLV" == args.dataset:
                input_name      = img_name[0].split("/")[-1].split(".")[0]
                splits          = img_path[0].split(os.sep)
                data_source     = splits[-3]
                data_brightness = splits[-2]
                save_dir        = os.path.join(args.save, data_source, data_brightness)
            else:
                input_name = "%s" % (img_name[0])
                save_dir   = os.path.join(args.save, os.path.basename(os.path.split(img_path[0])[0]))
            os.makedirs(save_dir, exist_ok=True)
            enhance = save_images(enhance)
            output  = save_images(output)
            Image.fromarray(output).save( save_dir + "/" + input_name + "_denoise" + ".png", "PNG")
            Image.fromarray(enhance).save(save_dir + "/" + input_name + "_enhance" + ".png", "PNG")

    torch.set_grad_enabled(True)


if __name__ == "__main__":
    main()
