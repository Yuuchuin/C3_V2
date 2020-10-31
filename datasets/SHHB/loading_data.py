import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHB import SHHB
from .setting import dataset_cfg
import torch


def loading_data():
    mean_std = dataset_cfg.MEAN_STD
    log_para = dataset_cfg.LOG_PARA
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomCrop(dataset_cfg.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])

    '''
    val_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(dataset_cfg.TRAIN_SIZE)
    ])
    '''

    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    #density_size = (int(dataset_cfg.TRAIN_SIZE[0]/dataset_cfg.SCALE_FACTOR),
                    #int(dataset_cfg.TRAIN_SIZE[1]/dataset_cfg.SCALE_FACTOR))

    if dataset_cfg.SCALE_FACTOR == 1:
        gt_transform = standard_transforms.Compose([
            own_transforms.LabelNormalize(log_para)
        ])
    else:
        gt_transform = standard_transforms.Compose([
            own_transforms.GTScaleDown(dataset_cfg.SCALE_FACTOR),
            own_transforms.LabelNormalize(log_para)
        ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHB(dataset_cfg.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=dataset_cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    

    val_set = SHHB(dataset_cfg.DATA_PATH+'/test', 'test', main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=dataset_cfg.VAL_BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

    test_set = SHHB(dataset_cfg.DATA_PATH + '/test', 'test', main_transform=val_main_transform,
                   img_transform=img_transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_set, batch_size=dataset_cfg.TEST_BATCH_SIZE, shuffle=True, num_workers=8,
                            drop_last=False)

    return train_loader, val_loader, test_loader, restore_transform
