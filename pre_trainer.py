import torch.nn as nn
import torch.optim as optim
from models import Transformer
import config
from dataset import get_data
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import torch
import copy

if not os.path.exists(config.IMG_PATH) :
    os.mkdir(config.IMG_PATH)
if (not os.path.exists(config.SAVE_PTH)):
    os.mkdir(config.SAVE_PTH)

def trainer(table_name,pre_trained):
    index_of_table = config.TABLE_NAMES.index(table_name)
    datas,labels = get_data(config.PRE_DATA,config.TABLE_NAMES)
    model = Transformer(d_model=config.D_MODEL,
                        d_head=config.D_HEAD,
                        n_heads=config.N_HEADS,
                        d_ffn=config.D_FFN,
                        layers=config.N_LAYERS,
                        feature=config.FEATURE).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=7e-3, momentum=0.99)
    if pre_trained:
        model.load_state_dict(torch.load("./models/best.pth"))

    best_loss = 100000
    best_epoch = 0
    for epoch in range(80):
        epoch_loss = 0
        pre,true = [],[]
        for data, label in itertools.zip_longest(datas, labels):
            label_process = label[:,:,index_of_table].unsqueeze(2)
            data, label_process = data.cuda(), label_process.cuda()
            outputs = model(data)
            loss = criterion(outputs, label_process)
            loss_num = loss.item()
            epoch_loss += loss_num
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            pre.append(outputs.cpu().detach().numpy())
            true.append(label_process.cpu().detach().numpy())

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "./models/best.pth")

        real, pred = [], []
        for x, y in itertools.zip_longest(true, pre):
            for n in range(x.shape[0]):
                xx = x[n, :, :]
                yy = y[n, :, :]
                real.append(xx)
                pred.append(yy)
        real, pred =real[:], pred[:]
        pre = np.concatenate(pred, axis=0).squeeze(1)
        true = np.concatenate(real, axis=0).squeeze(1)


        plt.plot(true, color="blue", label="the real data",alpha=0.5)
        plt.plot(pre, color="red", label="the prediction data",alpha=0.5)
        plt.plot(pre - true, color="green",label="loss", alpha=0.8)
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.legend()

        plt.savefig(os.path.join(config.IMG_PATH, '{}.png'.format( epoch)))
        plt.close()
        print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss))
    print('best_loss: ', best_loss, '  best_epoch:', best_epoch)


trainer('high',pre_trained=True)

