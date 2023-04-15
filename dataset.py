import itertools
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import config


def get_one_data(data_path,table_names):
    data_all = pd.read_csv(data_path, encoding='gbk')
    data_process_once = []
    length = len(data_all)
    for  name in table_names:
        data_raw = data_all[name].values.astype(np.float64)
        data_raw  = data_raw .reshape(length, 1)
        data_process_once.append(data_raw)
    data_processed = torch.from_numpy(np.concatenate(data_process_once, axis=1))

    data = torch.zeros_like(data_processed)
    for i in range(len(data_processed)):
        data[i, :] = data_processed[len(data_processed) - 1 - i, :]

    label = data[5:,:].type(torch.float32)
    data = data[0:-5, :].type(torch.float32)
    dataset = TensorDataset(data, label)
    data_loader = DataLoader(dataset, batch_size=config.SEQ_SIZE, shuffle=False)
    return data_loader

def get_data(data_path,table_names,batch_size=32):
    gather_data,gather_label,res_data,res_label = [],[],[],[]
    now = 0
    for x,y in get_one_data(data_path,table_names):
        x1,y1 = torch.zeros(config.SEQ_SIZE, config.FEATURE),torch.zeros(config.SEQ_SIZE,4)
        for i,(x2,y2) in enumerate(itertools.zip_longest(x,y)):
            x1[i] = x2
            y1[i] = y2
        x1,y1 = x1.unsqueeze(0),y1.unsqueeze(0)
        gather_data.append(x1)
        gather_label.append(y1)

    for i in range(0,len(gather_data),batch_size):
        res_data.append(torch.concat(gather_data[i:i + batch_size],0))
        res_label.append(torch.concat(gather_label[i:i + batch_size], 0))
        now = i

    res_data.append(torch.concat(gather_data[now:], 0))
    res_label.append(torch.concat(gather_label[now:], 0))
    return res_data,res_label