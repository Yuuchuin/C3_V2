from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *
from .SCC_BaseModel import SCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR

# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg16, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])


        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))



    def forward(self, x):
        shape = x.shape[-2:]



        x = self.features4(x)       
        x = self.de_pred(x)

        x = F.interpolate(x, shape)

        return x

class VGG16(SCC_BaseModel):
    def __init__(self, dataloader, cfg, dataset_cfg, pwd):
        super(VGG16, self).__init__(dataloader, cfg, dataset_cfg, pwd)

        self.net = vgg16()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)


        if len(self.gpus) == 1:
            self.net = self.net.cuda()
        elif len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus).cuda()

        if len(self.gpus) >= 1:
            self.loss_mse_fn = nn.MSELoss().cuda()