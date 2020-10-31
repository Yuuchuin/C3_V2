import numpy as np
import torch
import os
#from models.MCNN import MCNN as NET
#from models.C_CNN import C_CNN as NET                 #GPU02 140    #CPU08 4.2
#from models.SCNN import SCNN as NET
#from models.SC_CNN import SC_CNN as NET
#from models.GCCNN_7_SC_Dila import GCCNN as NET
from models.CSRNet import CSRNet as NET
#from models.Res101_SFCN import Res101_SFCN as NET
#from models.PFNet54 import PFNet as NET
#from models.PFNet import PFNet as NET
#from models.GPFANet import GPFANet as NET
#from models.ghost_C_CNN_9 import ghost_C_CNN as NET   #CPU08  5.4
#from models.C_CNN_9 import ghost_C_CNN as NET         #GPU02 142
#from models.G_LC_CNNAC import LC_CNN as NET
#from models.LC_CNNAC import LC_CNN as NET

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dummy_path = './dummy'


# 启动torch，避免初次启动torch耗时影响
dummy = torch.randn(1,3,512,512)#.cuda()
model = NET()#.cuda()
model.eval()
model(dummy)



dummy_list = os.listdir(dummy_path)
#torch.cuda.synchronize()
start = time.time()
for idx, dummy_name in enumerate(dummy_list):
    
    dummy = torch.from_numpy(np.load('./dummy/' + dummy_name)).float()#.cuda()
    model(dummy)
    if idx % 10 == 0:
        print("{} th dummy".format(idx+1))
    
    if idx > 100:
        break
    
#torch.cuda.synchronize()
end = time.time()

cost = end-start



print()
print('Average FPS: {}'.format(len(dummy_list)/cost))
