from easydict import EasyDict as edict

# init
__C_SHHB = edict()

dataset_cfg = __C_SHHB

__C_SHHB.STD_SIZE = (768,1024)
__C_SHHB.TRAIN_SIZE = (576,768) # for random crop when training
__C_SHHB.DATA_PATH = '/home/zhangli/yhs/ProcessedData/ProcessedData/shanghaitech_part_B_BACKUP'

__C_SHHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_SHHB.SCALE_FACTOR = 1 # shrink the density map to 1/SCALE_FACTOR of input image size
__C_SHHB.LOG_PARA = 1000.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 24 #imgs

__C_SHHB.VAL_BATCH_SIZE = 24 #
__C_SHHB.TEST_BATCH_SIZE = 24 #


