import torch
import torch.nn as nn
from misc.layer import Conv2d
import torch.nn.functional as F
from misc.utils import *
from .SCC_BaseModel import SCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR

class mcnn(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(mcnn, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))

        initialize_weights(self.modules())   
        
    def forward(self, im_data):
        h, w = im_data.shape[-2:]
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        x = F.interpolate(x, [h,w])
        #x = F.upsample(x,scale_factor=4)
        return x

class MCNN(SCC_BaseModel):
    def __init__(self, dataloader, cfg, dataset_cfg, pwd):
        super(MCNN, self).__init__(dataloader, cfg, dataset_cfg, pwd)

        self.net = mcnn()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)


        if len(self.gpus) == 1:
            self.net = self.net.cuda()
        elif len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus).cuda()

        if len(self.gpus) >= 1:
            self.loss_mse_fn = nn.MSELoss().cuda()

if __name__ == '__main__':
    dummy = torch.randn([2,3,512,512]).cuda()
    model = mcnn().cuda()
    pre = model(dummy)
    print(pre.shape)