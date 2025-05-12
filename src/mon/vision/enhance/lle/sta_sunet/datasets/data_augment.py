import torch
from torchvision import transforms 
import numpy as np


class RandomCrop(object):
    
    def __init__(self, output_size, topleft=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
        
    def __call__(self, sample):
        image, groundtruth = sample["image"], sample["groundtruth"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not self.topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not self.topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image       = image[top : top + new_h, left : left + new_w]
        groundtruth = groundtruth[top : top + new_h, left : left + new_w]
        return {"image": image, "groundtruth": groundtruth}


class ToTensor(object):
    
    def __init__(self, network="STASUNet"):
        self.network = network
        
    def __call__(self, sample):
        image, groundtruth = sample["image"], sample["groundtruth"]
        # swap color axis because
        # numpy image: H x W x C or H x W x C x T
        # torch image: C x H x W or T x C x H x W  
        if self.network=="STASUNet":
            image = image.transpose((3, 2, 0, 1))
        else:
            image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        if self.network=="STASUNet":
            image = (image-0.5)/0.5 # to range [-1,1]
        else:
            # handle multi-channels
            vallist = [0.5]*image.shape[0] 
            normmid = transforms.Normalize(vallist, vallist)
            image   = normmid(image)
        # ground truth
        vallist = [0.5] * groundtruth.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        groundtruth = normmid(groundtruth)
        return {"image": image, "groundtruth": groundtruth}


class RandomFlip(object):
    
    def __call__(self, sample):
        image, groundtruth = sample["image"], sample["groundtruth"]
        op = np.random.randint(0, 3)
        if op < 2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {"image": image, "groundtruth": groundtruth}
