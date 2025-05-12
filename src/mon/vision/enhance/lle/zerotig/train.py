import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image
from torch.autograd import Variable

from dataloader.create_data import CreateDataset
from model.model import *
from utils import utils

parser = argparse.ArgumentParser("ZERO-TIG")
parser.add_argument("--batch_size",           type=int,   default=1,        help="batch size")
parser.add_argument("--cuda",                 type=bool,  default=True,     help="Use CUDA to train model")
parser.add_argument("--gpu",                  type=str,   default="0",      help="gpu device id")
parser.add_argument("--seed",                 type=int,   default=2,        help="random seed")
parser.add_argument("--epochs",               type=int,   default=5,        help="epochs")
parser.add_argument("--lr",                   type=float, default=0.0003,   help="learning rate")
parser.add_argument("--save",                 type=str,   default="./EXP/", help="location of the data corpus")
parser.add_argument("--model_pretrain",       type=str,   help="location of the data corpus")
parser.add_argument("--lowlight_images_path", type=str,   default="",       help="input data folder")
parser.add_argument("--raft_model",           type=str,   default="./weights/raft-sintel.pth", help="path to pre-trained raft model")
parser.add_argument("--of_scale",             type=int,   default=3,        help="downscale size when compute OF")
parser.add_argument("--dataset",              type=str,   default="RLV",    help="Specified data set")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save  = args.save + "/" + "Train-{}".format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))
model_path = args.save + "/model_epochs/"
os.makedirs(model_path, exist_ok=True)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type("torch.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled   = True
    logging.info("gpu device = %s" % args.gpu)
    logging.info("args = %s", args)

    model = Network(args)
    utils.save(model, os.path.join(args.save, "initial_weights.pt"))
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    
    try:
        base_weights    = torch.load(args.model_pretrain)
        pretrained_dict = base_weights
        model_dict      = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info("Loaded pre-trained model from %s." % args.model_pretrain)
    except:
        logging.info("Model is initialized without pre-trained model.")

    model     = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB        = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)

    # Dataset
    TrainDataset = CreateDataset(args, task="train")
    logging.info("Training data: %d", TrainDataset.__len__())
    TestDataset  = CreateDataset(args, task="test")
    logging.info("Test data: %d", TestDataset.__len__())

    train_queue = torch.utils.data.DataLoader(
        TrainDataset,
        batch_size   = args.batch_size,
        pin_memory   = True,
        num_workers  = 0,
        shuffle      = False,
        generator    = torch.Generator(device="cuda")
    )
    test_queue = torch.utils.data.DataLoader(
        TestDataset,
        batch_size  = 1,
        pin_memory  = True,
        num_workers = 0,
        shuffle     = False,
        generator   = torch.Generator(device="cuda")
    )

    total_step = 0
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for idx, (input, img_name, img_path, last_img_path) in enumerate(train_queue):
            model.is_new_seq = utils.sequential_judgment(img_path[0], last_img_path[0])
            if model.is_new_seq:
                print("Get this img from: ", img_path, "\n Last img from: ", last_img_path)

            total_step += 1
            input       = Variable(input, requires_grad=False).cuda()
            optimizer.zero_grad()
            optimizer.param_groups[0]["capturable"] = True
            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            logging.info("train-epoch %03d %03d %f", epoch, idx, loss)
        logging.info("train-epoch %03d %f", epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, "weights_%d.pt" % epoch))

        if epoch % 1 == 0 and total_step != 0:
            model.eval()
            with torch.no_grad():
                for idx, (input, img_name, img_path, last_img_path) in enumerate(test_queue):
                    model.is_new_seq = utils.sequential_judgment(img_path[0], last_img_path[0])
                    if model.is_new_seq:
                        print("Eval Get this img from: ", img_path, "\n Last img from: ", last_img_path)
                    input = Variable(input, volatile=True).cuda()
                    L_pred1,L_pred2,L2,s2,s21,s22,H2,H11,H12,H13,s13,H14,s14,H3,s3,H3_pred,H4_pred,L_pred1_L_pred2_diff,H13_H14_diff,H2_blur,H3_blur,H3_denoised1,H3_denoised2 = model(input)
                    input_name = "%s_%s" % (os.path.basename(os.path.split(img_path[0])[0]), img_name[0])
                    H3_img     = save_images(H3)
                    H2_img     = save_images(H2)
                    os.makedirs(args.save + "/result/denoise/", exist_ok=True)
                    os.makedirs(args.save + "/result/enhance/", exist_ok=True)
                    Image.fromarray(H3_img).save(args.save + "/result/denoise/" + input_name + "_denoise_" + str(epoch) + ".png", "PNG")
                    Image.fromarray(H2_img).save(args.save + "/result/enhance/" + input_name + "_enhance_" + str(epoch) + ".png", "PNG")


if __name__ == "__main__":
    main()
