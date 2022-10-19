# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import ConvBlock, Conv3x3, upsample


class TargetDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(TargetDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.avg = nn.AdaptiveAvgPool2d(1)
        #self.add = nn.Linear(512, 1280)
        self.add = nn.Linear(2048, 1280)
        self.add2 = nn.Linear(1280, 3)
        #     t.nn.ReLU(inplace = True) # we use ReLU here as default
    def forward(self, input_features):
        self.outputs = {}
        B, C, H, W = input_features[-1].shape
        # decoder
        x = input_features[-1]
        # for i in range(4, -1, -1):
        #     x = self.convs[("upconv", i, 0)](x)
        #     x = [upsample(x)]
        #     if self.use_skips and i > 0:
        #         x += [input_features[i - 1]]
        #     x = torch.cat(x, 1)
        #     x = self.convs[("upconv", i, 1)](x)
        #     if i in self.scales:
        #         self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        x = self.avg(x).squeeze()
        #output= x.view(B,C,H*W)
        #output= self.outputs[("disp"),0].view(B,1,H*W*4)
        #output  = self.avg(self.outputs[("disp"),0]).squeeze()
        #x_avg.squeeze()
        output = self.add(x)
        self.outputs[("out_xyc")]= self.add2(output).squeeze()
        #self.outputs[("out_xyc")]= self.sigmoid(self.add2(output).squeeze())

        
        return self.outputs
