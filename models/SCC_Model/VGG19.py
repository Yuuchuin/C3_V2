import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from .SCC_BaseModel import SCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        self.frontend = make_layers(net_cfg['E'])
        self.frontend.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):

        x = self.frontend(x)
        x = F.interpolate(x, scale_factor=2)
        #x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)



class VGG19(SCC_BaseModel):
    def __init__(self, dataloader, cfg, dataset_cfg, pwd):
        super(VGG19, self).__init__(dataloader, cfg, dataset_cfg, pwd)

        self.net = vgg19()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)


        if len(self.gpus) == 1:
            self.net = self.net.cuda()
        elif len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus).cuda()

        if len(self.gpus) >= 1:
            self.loss_mse_fn = nn.MSELoss().cuda()






def make_layers(net_cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in net_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

net_cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


if __name__ == '__main__':
    dummy = torch.randn([2,3,512,512])
    model = vgg19()
    pre = model(dummy)
    print(pre.shape)
