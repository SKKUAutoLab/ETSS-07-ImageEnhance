import numpy as np
import torch
from PIL import Image

def get_dark_channel(image, w=15):
    if not isinstance(image, torch.Tensor):
        image = np.array(image)
        height, width, _ = image.shape
        padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
        darkch = np.zeros((height, width))
        for i, j in np.ndindex(darkch.shape):
            darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
    else:
        batch, channel, height, width = image.shape
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        padded = torch.nn.functional.pad(image, (w // 2, w // 2, w // 2, w // 2), mode='replicate')
        unfolded = torch.nn.functional.unfold(padded,(w, w), stride=1).view(batch, channel, w * w, -1)
        min_values = unfolded.min(dim=2).values
        min_values = min_values.min(dim=1, keepdim=True).values
        darkch = min_values.view(batch, 1, height, width)
    return darkch

def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    if not isinstance(image, torch.Tensor):
        image = image.transpose(1, 2, 0)
        # reference CVPR09, 4.4
        darkch = get_dark_channel(image, w)
        M, N = darkch.shape
        flatI = image.reshape(M * N, 3)
        flatdark = darkch.ravel()
        searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
        # return the highest intensity for each channel
        return np.max(flatI.take(searchidx, axis=0), axis=0)
    else:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        darkch = get_dark_channel(image, w)
        B, C, H, W = image.shape
        flatI = image.view(B, C, -1)
        flatdark = darkch.view(B, 1, -1)
        searchidx = (-flatdark).argsort(dim=-1)[:, :, :int(H * W * p)]
        return torch.gather(flatI, dim=2, index=searchidx.expand(-1, 3, -1)).max(dim=-1).values