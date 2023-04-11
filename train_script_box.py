import argparse
from datetime import datetime
import logging
import os
import random
import shutil
import sys
import torch
import gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.classification import MulticlassConfusionMatrix

from data.dataset import Driftset
from model.resnet import BasicBlock, ResNet
from model.unet import UNet
from torchsummary import summary
from config import configModel

import sklearn.utils.class_weight as class_weight


cfg = configModel()
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda:0")
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

def train_model(
        model,
        device,
        train_loader,
        val_loader,
        fold,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
     # 1. Create dataset
   
    # (Initialize logging)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
                            #   lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best  =0
    metric = MulticlassConfusionMatrix(num_classes=cfg.n_class)
    metric.to(device)
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        trainacclist = []
        valacclist = []
        loss_sum  =0
        train_acc_count = 0
        counter = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

        y_list = []
        pre_list = []
        for batch in train_loader:
            images, y = batch
            images = images.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            outputs = model(images)
            # s = F.sigmoid(outputs)
            s = F.softmax(outputs, dim=1)

            ce = nn.CrossEntropyLoss(weight=class_weights)
            loss = ce(s,y)

            # bce = torch.nn.BCEWithLogitsLoss()
            # loss = bce(outputs.flatten(),y.float())
            loss_sum += loss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            predict= s.argmax(1)
            
            y_list.append(y)
            pre_list.append(predict)
            
            # print(y)
            a = predict==y
            train_acc_count += int(torch.sum(a))
            counter += batch_size
            print('\rtrain acc : %.3f , Loss : %.3f'%(train_acc_count/counter, loss_sum/(counter//batch_size) )
                ,end='')
        
        pp = torch.cat(pre_list)
        yy = torch.cat(y_list)

        print('\n',metric(pp,yy))

        train_acc = train_acc_count / n_train
        trainacclist.append(train_acc)
        print(epoch,'th epoch : Final train acc : %.3f , Loss : %.3f'%(train_acc, loss_sum/(n_train//batch_size) ))
        model.eval()

        val_dab = []
        val_prid = []
        for B, batch in enumerate(val_loader):
            # print(B)
            inputs, labels = batch
            inputs = inputs.to(torch.device('cuda:0'))
            labels = labels.to(torch.device('cuda:0'))


            val_dab.append(labels.cpu().detach())
            outputs = model(inputs)

            # s = F.sigmoid(outputs)

            s = F.softmax(outputs, dim=1)

            val_prid.append (s.argmax(1).cpu().detach())
            # val_prid.append ((s.cpu().detach()>0.5).int())
        

        vd = np.concatenate([val for val in val_dab])
        vp = np.concatenate([val for val in val_prid])
        vp = vp.reshape(vp.shape[0])
        acc = np.sum(vd == vp) / len(vd)
        valacclist.append(acc)

        print('Fold:',fold, 'Epoc:',epoch, 'val_acc : ',acc)

        if acc > best:
            best = acc
            print('**best model updated with acc : ',acc)
            torch.save(model.state_dict(),
                        ('box_fold%d_best.pth' % fold))
        print('\n')
    dt = datetime.now().strftime('%m%d_%H%M%S')
    filename = ('box_fold%d_best_%s.pth' % (fold,dt)) 
    shutil.copy (('box_fold%d_best.pth' % fold), os.path.join(cfg.model_archive_path , filename ))
    

def readNConcat(filelist):
    arr = []
    for file_ in filelist:
        ele = np.load(file_)
        arr.append(ele)
    return np.concatenate(arr,axis=0)

if __name__ == '__main__':
    K = 5
    srcpath  ='D:\\src\\box_detect\\data\\data2304_v03\\'
    filenames = [os.path.join(srcpath,'box_val_X_fold' + str(i)+'.npy') for i in range(K)]
    filenamesY = [os.path.join(srcpath,'box_val_Y_fold' + str(i)+'.npy') for i in range(K)]
    

    # filelist = []
    for i in range(1):
        print('fold ',i,'start')

        trainlist = filenames[0:i] + filenames[i+1:K]
        vallist = [filenames[i]]
        trainlistY = filenamesY[0:i] + filenamesY[i+1:K]
        vallistY = [filenamesY[i]]
        
        train_X = cfg.ch_select(readNConcat(trainlist))
        train_Y = readNConcat(trainlistY)
        val_X = cfg.ch_select(readNConcat(vallist))
        val_Y = readNConcat(vallistY)

        global class_weights
        class_weights=class_weight.compute_class_weight('balanced',classes=np.unique(train_Y),y=train_Y)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        class_weights=class_weights.to(device)
        print(class_weights)
        
        global n_val
        global n_train
        n_val = len(val_Y)
        n_train = len(train_Y)

        
        dftrainset = Driftset(train_X,train_Y)
        dfvalset = Driftset(val_X,val_Y)
        
        batch_size = 32

        iter_train = torch.utils.data.DataLoader(dftrainset, batch_size=batch_size,
                                             shuffle=True, pin_memory=False)
        iter_valid = torch.utils.data.DataLoader(dfvalset, batch_size=2, shuffle=False,
                                             pin_memory=True)


        # model = UNet(n_channels=5, n_classes=2, bilinear=True)
        model = ResNet(BasicBlock, [2,2,2,2],num_classes = cfg.n_class,in_ch=cfg.in_ch)
    
        model = model.to(device=device)
        if i== 0:
            summary(model, (cfg.in_ch,) + cfg.d_size)


        train_model(model,device,iter_train,iter_valid,fold=i,batch_size=batch_size,epochs = 20)

        print('-' * 50)
            # memory_format=torch.channels_last)
        
