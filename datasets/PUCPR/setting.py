from easydict import EasyDict as edict

# init
__C_PUCPR = edict()

cfg_data = __C_PUCPR

__C_PUCPR.STD_SIZE = (544,960)
__C_PUCPR.TRAIN_SIZE = (408,720)
__C_PUCPR.DATA_PATH = '/home/zhangli/yhs/ProcessedData/ProcessedData/PUCPR'

__C_PUCPR.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_PUCPR.LABEL_FACTOR = 1
__C_PUCPR.LOG_PARA = 1000.

__C_PUCPR.RESUME_MODEL = ''#model path

__C_PUCPR.TRAIN_BATCH_SIZE = 4 #imgs

__C_PUCPR.VAL_BATCH_SIZE = 4 #

__C_PUCPR.TEST_BATCH_SIZE = 4 #


