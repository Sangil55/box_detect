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
from midas.run import load_model_dpt
from model.metric import BinaryMetrics, SegmentationMetrics
from model.resnet import BasicBlock, ResNet
from model.semantic_loss_functions import CrossentropyND
from model.unet import UNet
from model.dptAug import dptAug
from torchsummary import summary
from config import configModel

import sklearn.utils.class_weight as class_weight
from sklearn.model_selection import KFold


cfg = configModel()
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda:0")
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

def train_model(
        model,
        device,
        train_loader,
        fold,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
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
    losssum = 0
    counter = 0
    for epoch in range(1, epochs + 1):
      
        for batch in train_loader:
            images, y = batch
            images = (images/255).to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)


            recon_img , seg_img = model(images)
            loss = torch.norm (recon_img-images)
            losssum += loss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            if counter % 20 == 19:
                
                print('\r'+str(epoch)+'th_epoch Loss : %.3f, avg loss : %.3f'%(loss,losssum/(counter+1)) ,end='')
                if counter % 2000 == 19:
                    f, axarr = plt.subplots(2,2)
                    axarr[0,0].imshow(images[0].cpu().detach().swapaxes(0,2))
                    axarr[0,1].imshow(recon_img[0].cpu().detach().swapaxes(0,2))
                    axarr[1,0].imshow(images[1].cpu().detach().swapaxes(0,2))
                    axarr[1,1].imshow(recon_img[1].cpu().detach().swapaxes(0,2))
                    
                    plt.show()
                # plt.close()
            counter += 1
            # print('\r'+str(epoch)+'th_epoch Loss : %.3f'%(loss) ,end='')
            # s = F.sigmoid(outputs)
        torch.save(model.state_dict(),('pretrain_dpt.pth' ))
        print('\n')

def train_model_seg(
        model,
        device,
        train_loader,
        fold,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
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

    seg_metric = SegmentationMetrics()
    metric = MulticlassConfusionMatrix(num_classes=cfg.n_class)
    metric.to(device)
    # 5. Begin training
    losssum = 0
    counter = 0
    for epoch in range(1, epochs + 1):
      
        for batch in train_loader:
            images, y = batch
            images = (images/255).to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)


            recon_img , seg_img = model(images)
            # seg_img = seg_img.permute(0,2,3,1)
            loss = CrossentropyND()(seg_img,y)
            # loss = torch.norm (recon_img-images)
            losssum += loss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            counter+=1
            if counter % 20 == 19:
                n_classes = seg_img.shape[-1]
                print('\r'+str(epoch)+'th_epoch Loss : %.3f, avg loss : %.3f'%(loss,losssum/(counter+1)) ,end='')
                pixel_acc, dice, precision, recall = seg_metric (y.view(-1,),seg_img.reshape(-1,n_classes) )
                print( ' ', '%.3f, %.3f, %.3f, %.3f'%(pixel_acc, dice, precision, recall) ,end = '')
                # print(torch.unique(y.view(-1,)), torch.unique(seg_img.reshape(-1,n_classes).argmax(1)))

        
                if counter % 2000 == 19:
                    
                    def hook_fn(m, i, o):
                        v = o 

                    real_img = images[0].permute(1,2,0).cpu().detach().numpy()
                    y_img = y[0].cpu().detach().numpy()
                    predicted = seg_img[0].argmax(-1).cpu().detach().numpy()
                    last_depth = model.get_last_hook()[0].cpu().detach().numpy()

                    f, axarr = plt.subplots(2,2)
                    axarr[0,0].imshow(real_img)
                    axarr[0,1].imshow(y_img)
                    axarr[1,0].imshow(predicted)
                    axarr[1,1].imshow(last_depth)
                    f.savefig('last.png') 
                    plt.close(f)
                    
                    
                    # print( y[0],seg_img[0])
            # print('\r'+str(epoch)+'th_epoch Loss : %.3f'%(loss) ,end='')
            # s = F.sigmoid(outputs)
        torch.save(model.state_dict(),('pretrain_dpt.pth' ))
        print('\n')


def train_model_OOG(
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
    metric = MulticlassConfusionMatrix(num_classes=2)
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

            ce = nn.CrossEntropyLoss()
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
    import matplotlib.pyplot as plt

    K = 5
    srcpath  ='D:\\src\\box_detect\\data\\'
    filenames = [os.path.join(srcpath,'box_val_X_fold' + str(i)+'_256.npy') for i in range(K)]
    filenamesY = [os.path.join(srcpath,'box_val_Y_fold' + str(i)+'_256.npy') for i in range(K)]
    

    # filelist = []
    for i in range(1):
        print('fold ',i,'start')
        ##########################################################
        #########stage1 : pretrain depth model(optional)##########
        ##########################################################

        trainlist = filenames[0:i] + filenames[i+1:K]
        vallist = filenames
        trainlistY = filenamesY[0:i] + filenamesY[i+1:K]
        vallistY = filenamesY
        
        train_X = readNConcat(vallist)[:,:,:,0:3]
        train_Y = readNConcat(vallistY)
        dftrainset = Driftset(train_X,train_Y)
        
        batch_size = 8
        global n_train
        n_train = len(train_Y)

        iter_train = torch.utils.data.DataLoader(dftrainset, batch_size=batch_size,
                                             shuffle=True, pin_memory=False)

        dpt_model,_,_,_ = load_model_dpt(device=device,model_path=cfg.depth_model_path,model_type=cfg.depth_model_type)
        model = dptAug(dpt_model,mode='pretrain')
        model = model.to(device=device)
        
        if i== 0:
            summary(model, (3,) + cfg.d_size)

        # train_model(model,device,iter_train,fold=i,batch_size=batch_size,epochs = 200,learning_rate=0.001)
        
        ##########################################################
        #########stage2 : Depth ->Segment (optional)##########
        ##########################################################

        fileX = 'data\\citydataX_256.npy'
        fileY = 'data\\citydataY_256_category.npy'
        train_X = np.load(fileX)
        train_Y = np.load(fileY)

        label_unique = np.unique(train_Y)
        print( 'label.unique : ',label_unique )
        l_max = np.max(label_unique) +1
        print( 'set segment class label number as : ',l_max)


        # train_X = readNConcat(vallist)[:,:,:,0:3]
        # train_Y = readNConcat(vallistY)
        dftrainset = Driftset(train_X,train_Y)
        
        batch_size = 8
        n_train = len(train_Y)

        iter_train = torch.utils.data.DataLoader(dftrainset, batch_size=4,
                                             shuffle=True, pin_memory=False)

        dpt_model,_,_,_ = load_model_dpt( device=device,model_path=cfg.depth_model_path,model_type=cfg.depth_model_type )
        for m in dpt_model.parameters():
            m.requires_grad = False
        model = dptAug(dpt_model,mode='segment', seg_label_num = l_max)
        model = model.to(device=device)
        
        train_model_seg(model,device,iter_train,fold=i,batch_size=batch_size,epochs = 200,learning_rate=0.001)
        
        
        ##########################################################
        ###########stage3 : Abnormal classification ##############
        ##########################################################
        
        idx_0 = train_Y==0
        train_X = train_X[idx_0]
        train_Y = train_Y[idx_0]

        rand_X = np.random.random(size = train_X.shape)*255
        rand_Y = np.ones_like(train_Y)

        train_X = np.concatenate([train_X,rand_X],axis = 0)
        train_Y = np.concatenate([train_Y,rand_Y],axis = 0) 

        seed = 0
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        for i, (idx_train, idx_test) in enumerate(kf.split(train_Y)):
            break
        
        
        dftrainset = Driftset(train_X[idx_train],train_Y[idx_train])
        dfvalset = Driftset(train_X[idx_test],train_Y[idx_test])
        global n_val
        n_train = len(train_Y[idx_train])
        n_val = len(train_Y[idx_test])
        

        iter_train = torch.utils.data.DataLoader(dftrainset, batch_size=batch_size,
                                             shuffle=True, pin_memory=False)
        
        iter_val = torch.utils.data.DataLoader(dfvalset, batch_size=batch_size,
                                             shuffle=True, pin_memory=False)
        model.set_mode(mode='classify')
        train_model_OOG(model,device,iter_train,iter_val,fold=i,batch_size=batch_size,epochs = 10)


        print('-' * 50)


        
