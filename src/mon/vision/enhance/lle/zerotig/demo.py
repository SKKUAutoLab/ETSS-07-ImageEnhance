import sys
import time

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from model.RAFT.raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, [640,360])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(0)


def warp_img(flow, img1, img2, dst_size=[1080, 1920]):
    # flow: img1 -> img2

    h, w = flow.shape[:2]
    h_dst, w_dst = dst_size[:2]
    h_scale = float(h_dst / h)
    w_scale = float(w_dst / w)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # flow[..., 0] 是水平位移，flow[..., 1] 是垂直位移
    # 新的坐标位置 = 原始坐标 + 位移
    map_x = (grid_x - flow[..., 0]) * h_scale
    map_y = (grid_y - flow[..., 1]) * w_scale

    map_x = cv2.resize(map_x, [w_dst, h_dst])
    map_y = cv2.resize(map_y, [w_dst, h_dst])
    # 使用 remap 进行warp
    warped_image = cv2.remap(img1, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    overlap_image = cv2.addWeighted(warped_image, 0.5, img2, 0.5, 0)

    return warped_image, overlap_image

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    # model = RAFT_concise(args)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    img1 = cv2.imread(r"D:\LYN\5_Benchmark\6_Zero_IG\data\Esprit\00002.png")
    img2 = cv2.imread(r"D:\LYN\5_Benchmark\6_Zero_IG\data\Esprit\00003.png")

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            t1 = time.time()
            flow_low, flow_up = model(image1, image2, iters=15, test_mode=True)
            t2 = time.time()
            print(t2-t1)

            # warp
            flow = flow_up.squeeze().permute((1,2,0)).cpu().numpy()
            warped_image, overlap_image = warp_img(flow, img1, img2)
            pass
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
