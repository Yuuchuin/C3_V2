import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from .M2T2OCC_BaseModel import M2T2OCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR


class template(nn.Module):
    def __init__(self):
        super(template, self).__init__()
        '''
        The structure of your model.
        '''
        pass

    def forward(self, x):
        pass


class Template(M2T2OCC_BaseModel):
    def __init__(self, dataloader, cfg, dataset_cfg, pwd):
        super(Template, self).__init__(dataloader, cfg, dataset_cfg, pwd)

        self.net = template()  # Instance of your model
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        if len(self.gpus) == 1:
            self.net = self.net.cuda()
        elif len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus).cuda()

        if len(self.gpus) >= 1:
            self.loss_mse_fn = nn.MSELoss().cuda()
            self.loss_bce_fn = nn.BCELoss().cuda()


if __name__ == '__main__':
    dummy = torch.randn([2, 3, 512, 512]).cuda()
    model = template().cuda()
    pre = model(dummy)
    print(pre.shape)