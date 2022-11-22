###############################################################################################
# This piece of code was written by Zhongdao Wang for UniTrack project,                       #
# [NeurIPS 2021] Do different tracking tasks require different appearance model?              #
# Access from https://github.com/Zhongdao/UniTrack                                            #
###############################################################################################

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Corr_Up(nn.Module):
    """
    SiamFC head
    """
    def __init__(self):
        super(Corr_Up, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        if not self.training:
            return 0.1 * F.conv2d(x_f, z_f)
        else:
            return 0.1 * self._conv2d_group(x_f, z_f)

class SiamFC(nn.Module):
    def __init__(self,  **kwargs):
        super(SiamFC, self).__init__()
        self.features = None
        self.connect_model = Corr_Up()
        self.zf = None  # for online tracking
        if kwargs['base'] is None:
            self.features = ResNet22W()
        else:
            self.features = kwargs['base']
        self.model_alphaf = 0
        self.zf = None 
        self.features.eval()

    def feature_extractor(self, x):
        return self.features(x)

    def forward(self, x):
        xf = self.feature_extractor(x) * torch.Tensor(np.outer(np.hanning(65), np.hanning(65))).cuda()
        zf = self.zf 
        response = self.connect_model(zf, xf)
        return response
    
    def update(self, z, lr=0):
        zf = self.feature_extractor(z).detach() 
        _, _, ts, ts = zf.shape

        bg = ts//2-int(ts//(2*4.5))
        ed = ts//2+int(ts//(2*4.5))
        zf = zf[:,:,bg:ed, bg:ed]

        if self.zf is None:
            self.zf =  zf
        else:
            self.zf = (1 - lr) * self.zf + lr * zf



