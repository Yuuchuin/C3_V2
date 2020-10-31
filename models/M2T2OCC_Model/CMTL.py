#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from misc.utils import weights_normal_init
from misc.utils import *
from .M2T2OCC_BaseModel import M2T2OCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR


class cmtl(nn.Module):
    '''
    Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density
    Estimation for Crowd Counting (Sindagi et al.)
    '''

    def __init__(self, bn=False, num_classes=10):
        super(cmtl, self).__init__()

        self.num_classes = num_classes
        self.base_layer = nn.Sequential(Conv2d(3, 16, 9, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn))

        self.hl_prior_1 = nn.Sequential(Conv2d(32, 16, 9, same_padding=True, NL='prelu', bn=bn),
                                        nn.MaxPool2d(2),
                                        Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn),
                                        nn.MaxPool2d(2),
                                        Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d(16, 8, 7, same_padding=True, NL='prelu', bn=bn))

        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32, 32)),
                                        Conv2d(8, 4, 1, same_padding=True, NL='prelu', bn=bn))

        self.hl_prior_fc1 = FC(4 * 1024, 512, NL='prelu')
        self.hl_prior_fc2 = FC(512, 256, NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')

        self.de_stage_1 = nn.Sequential(Conv2d(32, 20, 7, same_padding=True, NL='prelu', bn=bn),
                                        nn.MaxPool2d(2),
                                        Conv2d(20, 40, 5, same_padding=True, NL='prelu', bn=bn),
                                        nn.MaxPool2d(2),
                                        Conv2d(40, 20, 5, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d(20, 10, 5, same_padding=True, NL='prelu', bn=bn))

        self.de_stage_2 = nn.Sequential(Conv2d(18, 24, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d(24, 32, 3, same_padding=True, NL='prelu', bn=bn),
                                        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        nn.PReLU(),
                                        Conv2d(8, 1, 1, same_padding=True, NL='relu', bn=bn))

        # weights_normal_init(self.base_layer, self.hl_prior_1, self.hl_prior_2, self.hl_prior_fc1, self.hl_prior_fc2, \
        #                     self.hl_prior_fc3, self.de_stage_1, self.de_stage_2)
        initialize_weights(self.modules())



    def forward(self, im_data):
        x_base = self.base_layer(im_data)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1)
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)
        x_den = self.de_stage_1(x_base)
        x_den = torch.cat((x_hlp1, x_den), 1)
        x_den = self.de_stage_2(x_den)
        return x_den, x_cls


class CMTL(M2T2OCC_BaseModel):
    def __init__(self, dataloader, cfg, dataset_cfg, pwd):
        super(CMTL, self).__init__(dataloader, cfg, dataset_cfg, pwd)

        self.net = cmtl()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)


        if len(self.gpus) == 1:
            self.net = self.net.cuda()
        elif len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus).cuda()

        ce_weights = torch.from_numpy(self.pre_weights()).float()

        if len(self.gpus) >= 1:
            self.loss_mse_fn = nn.MSELoss().cuda()
            self.loss_bce_fn = nn.BCELoss(weight=ce_weights).cuda()

    def pre_weights(self):
        count_class_hist = np.zeros(self.num_classes)
        for i, data in enumerate(self.train_loader, 0):
            if i < 100:
                _, gt_map = data
                for j in range(0, gt_map.size()[0]):
                    temp_count = gt_map[j].sum() / self.cfg_data.LOG_PARA
                    class_idx = min(int(temp_count / self.bin_val), self.num_classes - 1)
                    count_class_hist[class_idx] += 1

        wts = count_class_hist
        wts = 1 - wts / (sum(wts));
        wts = wts / sum(wts);
        print('pre_wts:')
        print(wts)

        return wts

    def online_assign_gt_class_labels(self, gt_map_batch):
        batch = gt_map_batch.size()[0]
        # pdb.set_trace()
        label = np.zeros((batch, self.num_classes), dtype=np.int)

        for i in range(0, batch):
            # pdb.set_trace()
            gt_count = (gt_map_batch[i].sum().item() / self.cfg_data.LOG_PARA)

            # generate gt's label same as implement of CMTL by Viswa
            gt_class_label = np.zeros(self.num_classes, dtype=np.int)
            # bin_val = ((self.max_gt_count - self.min_gt_count)/float(self.num_classes))
            class_idx = min(int(gt_count / self.bin_val), self.num_classes - 1)
            gt_class_label[class_idx] = 1
            # pdb.set_trace()
            label[i] = gt_class_label.reshape(1, self.num_classes)

        return torch.from_numpy(label).float()