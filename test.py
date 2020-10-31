import os
import numpy as np
import torch

from config import cfg
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES


#------------prepare enviroment------------
seed = cfg.SEED

if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_set = cfg.DATASET
if data_set is 'SHHA':
    from datasets.SHHA.loading_data import loading_data
    from datasets.SHHA.setting import dataset_cfg
elif data_set is 'SHHB':
    from datasets.SHHB.loading_data import loading_data
    from datasets.SHHB.setting import dataset_cfg
elif data_set is 'QNRF':
    from datasets.QNRF.loading_data import loading_data
    from datasets.QNRF.setting import dataset_cfg
elif data_set is 'UCF50':
    from datasets.UCF50.loading_data import loading_data
    from datasets.UCF50.setting import dataset_cfg
elif data_set is 'WE':
    from datasets.WE.loading_data import loading_data
    from datasets.WE.setting import dataset_cfg
elif data_set is 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import dataset_cfg
elif data_set is 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import dataset_cfg
elif data_set is 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import dataset_cfg


#------------Prepare Model------------
net = cfg.NET

#print([Net[:-3] for Net in os.listdir('./models/SCC_Model/')])
if net in [Net[:-3] for Net in os.listdir('./models/SCC_Model/')]:
    exec('from models.SCC_Model.{} import {} as MODEL'.format(net,net))



#------------Start Testing------------
state_path = ''

pwd = os.path.split(os.path.realpath(__file__))[0]
model = MODEL(loading_data,cfg,dataset_cfg,pwd)
print('Start testing!')
model.test(state_path)
