import torch
#分布式聚合参数
E = 0.001
B = 0.001
Y = 0.1
#股票参数
PRE_DATA = './data/pre_data.csv'
DATA = './data/stock.csv'
TABLE_NAMES = ['open', 'high', 'low', 'close']
IMG_PATH = './res/'
SAVE_PTH = './models/'
FEATURE = 4
#神经网络参数
BATCH_SIZE = 64
NMODELS =10
EPOECHS = 10
#transformer
D_MODEL = 512
D_FFN = 2048
D_HEAD = 64
N_LAYERS = 16
N_HEADS = 8
SEQ_SIZE = 64
#中间层的更新权重矩阵
AR_WEIGHT = [
          [1/2,1/2,0,0,0,0,0,0,0,0],
          [1/3,1/3,1/3,0,0,0,0,0,0,0],
          [0,0,1/2,0,0,0,1/2,0,0,0],
          [1/4,0,0,1/4,1/4,0,0,1/4,0,0],
          [0,0,0,1/2,1/2,0,0,0,0,0],
          [0,0,0,0,1/3,1/3,1/3,0,0,0],
          [0,0,1/4,0,0,1/4,1/4,0,0,1/4],
          [0,0,0,1/2,0,0,0,1/2,0,0],
          [0,0,0,0,0,0,0,1/3,1/3,1/3],
          [0,0,0,0,0,0,0,0,1/2,1/2],
          ]
#模型的更新权重矩阵
AC_WEIGHT = [
          [1/3,1/2,0,0,0,0,0,0,0,0],
          [1/3,1/2,1/3,0,0,0,0,0,0,0],
          [0,0,1/3,0,0,0,1/3,0,0,0],
          [1/3,0,0,1/3,1/3,0,0,1/3,0,0],
          [0,0,0,1/3,1/3,0,0,0,0,0],
          [0,0,0,0,1/3,1/2,1/3,0,0,0],
          [0,0,1/3,0,0,1/2,1/3,0,0,1/3],
          [0,0,0,1/3,0,0,0,1/3,0,0],
          [0,0,0,0,0,0,0,1/3,1/2,1/3],
          [0,0,0,0,0,0,0,0,1/2,1/3],
          ]
#权重矩阵
class Config:
    def __init__(self,ar,ac):
        self.ar = ar
        self.ac = ac

    def flush_weight(self):
        AR = [[(i, val) for i, val in enumerate(column) if val != 0] for column in self.ar]
        AC = [[(i, val) for i, val in enumerate(column) if val != 0] for column in self.ac]
        return {'ar': AR, 'ac': AC}
#随机初始化权重矩阵
# AR_WEIGHT = torch.nn.init.normal_(torch.empty(10,10),mean=0,std=1)
# AC_WEIGHT = torch.nn.init.normal_(torch.empty(10,10),mean=0,std=1)
configure = Config(AR_WEIGHT,AC_WEIGHT)
