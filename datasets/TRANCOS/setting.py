from easydict import EasyDict as edict

# init
__C_TRANCOS = edict()

cfg_data = __C_TRANCOS

__C_TRANCOS.STD_SIZE = (480, 640)
__C_TRANCOS.TRAIN_SIZE = (360,480)
__C_TRANCOS.DATA_PATH = ''

__C_TRANCOS.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_TRANCOS.LABEL_FACTOR = 1
__C_TRANCOS.LOG_PARA = 1000.

__C_TRANCOS.RESUME_MODEL = ''#model path

__C_TRANCOS.TRAIN_BATCH_SIZE = 4 #imgs

__C_TRANCOS.VAL_BATCH_SIZE = 4 #

__C_TRANCOS.TEST_BATCH_SIZE = 4 #


