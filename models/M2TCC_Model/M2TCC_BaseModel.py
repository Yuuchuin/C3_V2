import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from config import cfg
from torch.autograd import Variable
from misc.utils import *


class M2TCC_BaseModel(nn.Module):
    def __init__(self, dataloader, cfg, cfg_data, pwd, loss_1_fn,loss_2_fn):
        super(M2TCC_BaseModel, self).__init__()

        self.gpus = cfg.GPU_ID
        self.cfg_data = cfg_data
        self.data_set = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        if cfg.LOGGER:
            self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    @property
    def loss(self):
        return self.loss_1, self.loss_2 * cfg.LAMBDA_1

    def predict(self, img, gt_map):
        density_map = self.net(img)
        self.loss_1 = self.loss_1_fn(density_map.squeeze(), gt_map.squeeze())
        self.loss_2 = 1 - self.loss_2_fn(density_map, gt_map[:, None, :, :])

        return density_map

    def test_predict(self, img):
        density_map = self.net(img)
        return density_map


    def trainer(self):

        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            self.epoch = epoch

            # training
            self.timer['train time'].tic()
            self.train()
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
                    self.validate_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):  # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.predict(img, gt_map)
            loss1, loss2 = self.loss
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.writer.add_scalar('train_loss1', loss1.item(), self.i_tb)
                self.writer.add_scalar('train_loss2', loss2.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                    gt_map[0].sum().data / self.cfg_data.LOG_PARA, pred_map[0].sum().data / self.cfg_data.LOG_PARA))

    def validate_V1(self):  # validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.predict(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    loss1, loss2 = self.net.loss
                    loss = loss1.item() + loss2.item()
                    losses.update(loss)
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, \
                                         [mae, mse, loss], self.train_record, self.log_txt)
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)

    def validate_V2(self):  # validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        for i_sub, i_loader in enumerate(self.val_loader, 0):

            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.predcit(img, gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(), i_sub)
                        maes.update(abs(gt_count - pred_cnt), i_sub)
                    if vi == 0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map,
                                    gt_map)

        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, \
                                         [mae, 0, loss], self.train_record, self.log_txt)
        print_summary(self.exp_name, [mae, 0, loss], self.train_record)

    def validate_V3(self):  # validate_V3 for GCC

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}
        c_mses = {'level': AverageCategoryMeter(9), 'time': AverageCategoryMeter(8), 'weather': AverageCategoryMeter(7)}

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.predict(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map) / self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    loss1, loss2 = self.net.loss
                    loss = loss1.item() + loss2.item()
                    losses.update(loss)
                    maes.update(s_mae)
                    mses.update(s_mse)
                    attributes_pt = attributes_pt.squeeze()
                    c_maes['level'].update(s_mae, attributes_pt[0])
                    c_mses['level'].update(s_mse, attributes_pt[0])
                    c_maes['time'].update(s_mae, attributes_pt[1] / 3)
                    c_mses['time'].update(s_mse, attributes_pt[1] / 3)
                    c_maes['weather'].update(s_mae, attributes_pt[2])
                    c_mses['weather'].update(s_mse, attributes_pt[2])

                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, \
                                         [mae, mse, loss], self.train_record, self.log_txt)

        print_GCC_summary(self.log_txt, self.epoch, [mae, mse, loss], self.train_record, c_maes, c_mses)


    def test(self, state_path):  # test_V1 for SHHA, SHHB, UCF-QNRF, UCF50
        print(state_path)
        self.net.eval()

        maes = AverageMeter()
        mses = AverageMeter()

        for ti, data in enumerate(self.test_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.test_predict(img)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA


                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        mae = maes.avg
        mse = np.sqrt(mses.avg)

        print('MAE={}\tMSE={}\n'.format(mae, mse))

    def game_test(self, state_path):
        print(state_path)
        self.net.eval()
        state = torch.load(state_path)

        try:
            self.net.load_state_dict(state)
        except KeyError:
            state = {key[4:]: value for key, value in
                     state.items()}  # multi GPUs are used when training, while single GPU is used when testing
            self.net.load_state_dict(state)


        game0 = AverageMeter()
        game1 = AverageMeter()
        game2 = AverageMeter()
        game3 = AverageMeter()

        for vi, data in enumerate(self.test_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.test_predict(img)

                for pred, gt in zip(pred_map, gt_map):
                    game0.update(game(pred, gt, 0, self.cfg_data.LOG_PARA))
                    game1.update(game(pred, gt, 1, self.cfg_data.LOG_PARA))
                    game2.update(game(pred, gt, 2, self.cfg_data.LOG_PARA))
                    game3.update(game(pred, gt, 3, self.cfg_data.LOG_PARA))

            if vi % 10 == 0:
                print(vi)

        game0_ = game0.avg
        game1_ = game1.avg
        game2_ = game2.avg
        game3_ = game3.avg
        print('GAME0:{}\tGAME1:{}\tGAME2:{}\tGAME3:{}'.format(game0_, game1_, game2_, game3_))



