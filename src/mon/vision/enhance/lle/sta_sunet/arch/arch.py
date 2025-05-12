import torch
from torch import nn as nn
from torch.nn import functional as F
from arch.arch_util import ResidualBlockNoBN, make_layer
from arch.arch_enhance import *
from arch.arch_align import PCDAlignment


class STASUNet(nn.Module):
    """ STA-SUNet for low-light video enhancement

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: True.
        embed_dim (int): Patch embedding feature dimension. Default: 64
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
    """

    def __init__(
        self,
        num_in_ch             = 3,
        num_out_ch            = 3,
        num_feat              = 64,
        num_frame             = 5,
        deformable_groups     = 8,
        num_extract_block     = 5,
        num_reconstruct_block = 10,
        center_frame_idx      = None,
        hr_in                 = True,
        img_size              = 224,
        patch_size            = 4,
        embed_dim             = 64,
        depths                = [2, 2, 2 , 2],
        num_heads             = [3, 6, 12, 24],
        window_size           = 7,
        patch_norm            = False,
        final_upsample        = "Dual up-sample"
    ):
        super(STASUNet, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        '''Feature alignment module'''
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.fusion    = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1       = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv2       = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr       = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last     = nn.Conv2d(num_feat, 3       , 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        '''Enhancement module SUNet'''
        # add
        self.num_layers     = len(depths)
        self.patch_norm     = patch_norm
        self.embed_dim      = embed_dim
        self.final_upsample = final_upsample
        self.out_chans      = num_out_ch
        norm_layer          = nn.LayerNorm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            img_size   = img_size,
            patch_size = patch_size,
            in_chans   = embed_dim,
            embed_dim  = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None
        )
        num_patches             = self.patch_embed.num_patches
        patches_resolution      = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=0)

        # stochastic depth
        # set stochastic depth rate to 0.1
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim              = int(embed_dim * 2 ** i_layer),
                input_resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth            = depths[i_layer],
                num_heads        = num_heads[i_layer],
                window_size      = window_size,
                mlp_ratio        = 4.0,
                qkv_bias         = True,
                qk_scale         = None,
                drop             = 0,
                attn_drop        = 0,
                drop_path        = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer       = norm_layer,
                downsample       = PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint   = False
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up       = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            ) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(
                    input_resolution = patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    in_channels      = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    scale_factor     = 2
                )
            else:
                layer_up = BasicLayerUp(
                    dim              = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution = (patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth            = depths[(self.num_layers - 1 - i_layer)],
                    num_heads        = num_heads[(self.num_layers - 1 - i_layer)],
                    window_size      = window_size,
                    mlp_ratio        = 4.0,
                    qkv_bias         = True,
                    qk_scale         = None,
                    drop             = 0,
                    attn_drop        = 0,
                    drop_path        = dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer       = norm_layer,
                    upsample         = UpSample if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint   = False
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm    = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "Dual up-sample":
            self.up = UpSample(
                input_resolution = (img_size // patch_size, img_size // patch_size),
                in_channels      = embed_dim,
                scale_factor     = 4
            )
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1, padding=1, bias=False)  # kernel = 1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, residual, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W    = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "Dual up-sample":
            x = self.up(x)
            # x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment Module
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        # fuse frames together
        feat = self.fusion(aligned_feat)
        # reconstruction and upsample
        out  = self.reconstruction(feat)
        out  = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        # SUNet enhencement module
        sunet_x   = self.conv_first(out)
        sunet_x, residual, x_downsample = self.forward_features(sunet_x)
        sunet_x   = self.forward_up_features(sunet_x, x_downsample)
        sunet_x   = self.up_x4(sunet_x)
        out_sunet = self.output(sunet_x)

        return out_sunet
