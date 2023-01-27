# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from collections import OrderedDict


class FocalDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, input_width=640, input_height=480):
        super(FocalDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        self.input_width = input_width
        self.input_height = input_height

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeezef")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("focal", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("focal", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("focal", 2)] = nn.Conv2d(256, 2, 1)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeezef"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        outf = cat_features
        for i in range(3):
            outf = self.convs[("focal", i)](outf)
            #if i != 2:
        outf = self.sigmoid(outf)

        outf = outf.mean(3).mean(2)

        outf = outf.view(-1, 1, 2)

        return outf

class OffsetDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(OffsetDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeezeo")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("offset", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("offset", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("offset", 2)] = nn.Conv2d(256, 2, 1)

        self.relu = nn.ReLU(inplace=False)

        self.sigmoid = nn.Sigmoid()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeezeo"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        outo = cat_features
        for i in range(3):
            outo = self.convs[("offset", i)](outo)
            # if i != 2:
        outo = self.sigmoid(outo)

        outo = outo.mean(3).mean(2)

        outo =outo.view(-1, 1,  2)


        return outo