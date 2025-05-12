import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LossFunction, TextureDifference
from model.RAFT.raft import RAFT
from torchvision.transforms.functional import equalize
from utils.utils import blur, InputPadder, pair_downsampler, viz, warp_tensor


class Denoise_1(nn.Module):
    
    def __init__(self, chan_embed=48):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise_2(nn.Module):
    
    def __init__(self, chan_embed=96):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(12, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):
    
    def __init__(self, layers, channels):
        super().__init__()

        kernel_size = 3
        dilation    = 1
        padding     = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)
        return fea


class Network(nn.Module):

    def __init__(self, dataset: str, raft_model: str, of_scale: int, device):
        super().__init__()
        self.enhance    = Enhancer(layers=3, channels=64)
        self.denoise_1  = Denoise_1(chan_embed=48)
        self.denoise_2  = Denoise_2(chan_embed=48)
        self._l2_loss   = nn.MSELoss()
        self._l1_loss   = nn.L1Loss()
        self.is_WB      = True if "underwater" == dataset else False
        self._criterion = LossFunction(self.is_WB)
        self.avgpool    = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

        self.last_H3    = None
        self.last_H3_wp = None
        self.last_s3    = None
        self.last_s3_wp = None
        self.is_new_seq = True

        # optical flow
        self.raft     = self.load_raft(raft_model)
        self.of_scale = of_scale

    def load_raft(self, raft_model: str):
        raft = torch.nn.DataParallel(RAFT())
        # load pre-trained data
        raft.load_state_dict(torch.load(str(raft_model)))
        raft = raft.module
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False
        return raft

    def cvt_ts2np(self, t):
        # convert tensor to array
        t = t.detach()
        n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
        return n

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
        # if isinstance(m, nn.Conv2d):
        # nn.init.xavier_uniform(m.weight)
        # nn.init.constant(m.bias, 0)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        """ concat output from last frm"""
        if self.is_new_seq:
            self.last_H3_wp = torch.zeros_like(L2)
            self.last_s3_wp = torch.zeros_like(L2)
            self.last_H31_wp = torch.zeros_like(L11)
            self.last_H32_wp = torch.zeros_like(L11)
            self.last_s31_wp = torch.zeros_like(L11)
            self.last_s32_wp = torch.zeros_like(L11)
        else:
            # OF + warp
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach())
            self.last_H31_wp, self.last_H32_wp = pair_downsampler(self.last_H3_wp)
            self.last_s31_wp, self.last_s32_wp = pair_downsampler(self.last_s3_wp)

        s2 = self.enhance(torch.cat([self.last_H3_wp, self.last_s3_wp, L2], 1).detach())
        s21, s22 = pair_downsampler(s2)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)

        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([self.last_H31_wp, self.last_s31_wp, H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([self.last_H32_wp, self.last_s32_wp, H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff= self.TextureDifference(H3_denoised1, H3_denoised2)

        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur, H3_denoised1, H3_denoised2

    def _loss(self, input):
        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur, H3_denoised1, H3_denoised2 = self(
            input)
        loss = 0

        loss += self._criterion(input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3,
                                H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur,
                                H3_blur)

        self.update_H3(H3, s3)
        return loss

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    def update_cache(self, last_H3, last_s3, L2):
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht,wd), mode='bilinear')
        L2_tmp = F.interpolate(L2, (ht,wd), mode='bilinear')

        # 1. Equalize the histogram
        # last_H3_tmp = equalize((last_H3_tmp * 255).to(torch.uint8))
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32) #/ 255.0

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32) #/ 255.0

        # 2. OF last->this
        # last_H3_tmp, L2_tmp = self.padder.pad(last_H3_tmp, L2_tmp) # [640, 360]
        _, flow_up = self.raft(last_H3_tmp, L2_tmp, iters=20, test_mode=True)
        # viz(last_H3_tmp, flow_up)

        # 3. Warp
        warped_tensor_H3, overlap_tensor = warp_tensor(flow_up, last_H3, L2)
        warped_tensor_s3, _ = warp_tensor(flow_up, last_s3, L2)
        # warped_img = self.cvt_ts2np(warped_tensor)
        # overlap_img = self.cvt_ts2np(overlap_tensor)
        #
        # img_flo = cv2.resize(np.concatenate([warped_img, overlap_img], axis=0), (1920, 2160))
        # cv2.imshow('image', img_flo)
        # cv2.waitKey(5)
        # cv2.imwrite('./img_flo.png', (img_flo*255).astype(np.uint8))

        return warped_tensor_H3, warped_tensor_s3


class Finetunemodel(nn.Module):

    def __init__(self, weights: str, raft_model: str, of_scale: int, device):
        super().__init__()
        self.enhance    = Enhancer(layers=3, channels=64)
        self.denoise_1  = Denoise_1(chan_embed=48)
        self.denoise_2  = Denoise_2(chan_embed=48)

        base_weights    = torch.load(str(weights), map_location=device)
        pretrained_dict = base_weights
        model_dict      = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        self.last_H3    = None
        self.last_H3_wp = None
        self.last_s3    = None
        self.last_s3_wp = None
        self.is_new_seq = True

        # optical flow
        self.raft     = self.load_raft(raft_model)
        self.of_scale = of_scale

    def load_raft(self, raft_model: str):
        raft = torch.nn.DataParallel(RAFT())
        # load pre-trained data
        raft.load_state_dict(torch.load(str(raft_model)))
        raft = raft.module
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False
        return raft

    def cvt_ts2np(self, t):
        t = t.detach()
        n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
        return n

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps   = 1e-4
        input = input + eps
        L2    = input - self.denoise_1(input)
        L2    = torch.clamp(L2, eps, 1)

        """ concat output from last frm"""
        if self.is_new_seq:
            self.last_H3_wp = torch.zeros_like(L2)
            self.last_s3_wp = torch.zeros_like(L2)
        else:
            # OF + warp
            self.last_H3_wp, self.last_s3_wp = self.update_cache(self.last_H3, self.last_s3, L2.detach())

        s2 = self.enhance(torch.cat([self.last_H3_wp, self.last_s3_wp, L2], 1).detach())
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        if self.is_new_seq:
            self.last_H3_wp = H2.detach()
            self.last_s3_wp = H2.detach()

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([self.last_H3_wp, self.last_s3_wp, H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3      = H5_pred[:, :3, :, :]
        s3      = H5_pred[:, 3:, :, :]

        self.update_H3(H3, s3)
        return H2, H3, s3

    def update_H3(self, H3, s3):
        self.last_H3 = H3.detach()
        self.last_s3 = s3.detach()

    def update_cache(self, last_H3, last_s3, L2):
        # 0. resize
        ht_org, wd_org = last_H3[0].shape[-2:]
        ht = ht_org // self.of_scale
        wd = wd_org // self.of_scale
        last_H3_tmp = F.interpolate(last_H3, (ht, wd), mode='bilinear')
        L2_tmp      = F.interpolate(L2, (ht, wd), mode='bilinear')

        # 1. Equalize the histogram
        # last_H3_tmp = equalize((last_H3_tmp * 255).to(torch.uint8))
        last_H3_tmp = last_H3_tmp * 255
        last_H3_tmp = last_H3_tmp.to(torch.float32)  # / 255.0

        L2_tmp = equalize((L2_tmp * 255).to(torch.uint8))
        L2_tmp = L2_tmp.to(torch.float32)  # / 255.0

        # 2. OF last->this
        # last_H3_tmp, L2_tmp = self.padder.pad(last_H3_tmp, L2_tmp) # [640, 360]
        _, flow_up = self.raft(last_H3_tmp, L2_tmp, iters=20, test_mode=True)
        # viz(last_H3_tmp, flow_up)

        # 3. Warp
        warped_tensor_H3, overlap_tensor = warp_tensor(flow_up, last_H3, L2)
        warped_tensor_s3, _ = warp_tensor(flow_up, last_s3, L2)
        # warped_img = self.cvt_ts2np(warped_tensor)
        # overlap_img = self.cvt_ts2np(overlap_tensor)
        #
        # img_flo = cv2.resize(np.concatenate([warped_img, overlap_img], axis=0), (1920, 2160))
        # cv2.imshow('image', img_flo)
        # cv2.waitKey(5)
        # cv2.imwrite('./img_flo.png', (img_flo*255).astype(np.uint8))

        return warped_tensor_H3, warped_tensor_s3
