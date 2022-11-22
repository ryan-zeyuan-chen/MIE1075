###############################################################################################
# This piece of code was written by Zhongdao Wang for UniTrack project,                       #
# [NeurIPS 2021] Do different tracking tasks require different appearance model?              #
# Access from https://github.com/Zhongdao/UniTrack                                            #
###############################################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

class RandomFeatGenerator(nn.Module):
    def __init__(self, args):
        super(RandomFeatGenerator, self).__init__()
        self.df = args.down_factor
        self.dim = args.dim
        self.dummy = nn.Linear(2,3)
    def forward(self, x):
        if len(x.shape) == 4:
            N,C,H,W = x.shape
        elif len(x.shape) == 5:
            N,C,T,H,W = x.shape
        else:
            raise ValueError
        c, h, w = self.dim, round(H/self.df), round(W/self.df)

        if len(x.shape) == 4:
            feat = torch.rand(N,c,h,w).cuda()
        elif len(x.shape) == 5:
            feat = torch.rand(N,c,T,h,w).cuda()
        return feat

    def __str__(self):
        return ''
