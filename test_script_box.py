import shutil
import argparse
import logging
import os
import random
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

from data.dataset import Driftset
from midas.run import load_model_dpt
from model.dptAug import dptAug
from model.resnet import BasicBlock, ResNet
from model.unet import UNet
from torchsummary import summary
import collections

import cv2

from preprcess_box import test_set_generation
from config import configModel
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import confusion_matrix


cfg = configModel()
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda:0")
# print(torch.cuda.memory_summary(device=None, abbreviated=False))
test_mode = 'abnormal'

def test_model(
        model,
        device,
        val_loader,
        dataset_name
        ):
    val_dab = []
    val_prid = []
    counter = 0
    
    if os.path.exists('D:\\src\\box_detect\\misprediction\\'+dataset_name+'0'):
        shutil.rmtree('D:\\src\\box_detect\\misprediction\\'+dataset_name+'0')
        shutil.rmtree('D:\\src\\box_detect\\misprediction\\'+dataset_name+'1')
        shutil.rmtree('D:\\src\\box_detect\\misprediction\\'+dataset_name+'2')
    os.makedirs('D:\\src\\box_detect\\misprediction\\'+dataset_name+'0')
    os.makedirs('D:\\src\\box_detect\\misprediction\\'+dataset_name+'1')
    os.makedirs('D:\\src\\box_detect\\misprediction\\'+dataset_name+'2')

    y_list = []
    pre_list = []
    n_class = cfg.n_class   
    acounter = np.zeros(n_class)
    for B, batch in enumerate(val_loader):
        # print(B)
        inputs, labels = batch
        inputs = inputs.to(torch.device('cuda:0'))
        inputs_2ch = cfg.ch_select2(inputs)
        labels = labels.to(torch.device('cuda:0'))


        val_dab.append(labels.cpu().detach())
        outputs = model(inputs_2ch)
        
        if test_mode == 'abnormal':
            predict = (outputs[:,0]<0.5).int()
        else:
            s = F.softmax(outputs, dim=1)    
            predict= s.argmax(1)

        y_list.append(labels)
        pre_list.append(predict)
        

        for idx,p_val in enumerate(predict):
            p = int(p_val.cpu().detach())
            dpath = 'd:\\src\\box_detect\\misprediction\\'+dataset_name+str(p)
            img = inputs[idx].cpu().detach()[0:3]
            img = np.swapaxes(np.swapaxes(img.numpy(),0,2),0,1)
            cv2.imwrite(os.path.join(dpath,str(counter)+'_'+str(float(outputs[idx][p]))+'.jpeg'), img) 
            counter += 1
      
            

        
        # for idx,s_val in enumerate(s):
        #     counter += 1
        #     if float(s_val.cpu().detach()) > 0.5:
        #         img = inputs[idx].cpu().detach()[0:3]
        #         img = np.swapaxes(np.swapaxes(img.numpy(),0,2),0,1)
        #         cv2.imwrite(os.path.join('prediction','true',str(counter)+'_'+str(float(s_val))+'.jpeg'), img) 
        #         acounter[1] += 1
        #     if float(s_val.cpu().detach()) <= 0.5:
        #         img = inputs[idx].cpu().detach()[0:3]
        #         img = np.swapaxes(np.swapaxes(img.numpy(),0,2),0,1)
        #         cv2.imwrite(os.path.join('prediction','false',str(counter)+'_'+str(float(s_val))+'.jpeg'), img) 
        #         acounter[0] += 1
    pp = torch.cat(pre_list,axis=0).cpu().detach().numpy()
    print(collections.Counter(pp))
    pp[pp>1] = 1
    yy = torch.cat(y_list,axis=0).cpu().detach().numpy()

    return pp,yy

    # print ('' )
    # print ('total data count : ',counter, '  positive predicted : ', acounter[1] , '  negative :', acounter[0])


def readNConcat(filelist):
    arr = []
    for file_ in filelist:
        ele = np.load(file_)
        arr.append(ele)
    return np.concatenate(arr,axis=0)

if __name__ == '__main__':
    K = 5
    srcpath  ='D:\\src\\box_detect\\data\\data2303\\'
    filenames = [os.path.join(srcpath,'box_val_X_fold' + str(i)+'.npy') for i in range(K)]
    filenamesY = [os.path.join(srcpath,'box_val_Y_fold' + str(i)+'.npy') for i in range(K)]
    
    PATH = 'box_fold0_best.pth'
    # model = ResNet(BasicBlock, [2,2,2,2],num_classes = cfg.n_class,in_ch=cfg.in_ch)
    # model = model.to(device=device)
    dpt_model,_,_,_ = load_model_dpt(device=device,model_path=cfg.depth_model_path,model_type=cfg.depth_model_type)
    model = dptAug(dpt_model,mode='classify')
    model = model.to(device=device)

    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    # mode = 'new_test_set'
    mode = 'test'
    
    if mode == 'new_test_set':
            
        # target_datapath = 'D:\\data\\KD\\test\\self_test2\\2'
        # target_datapath = 'D:\\data\\KD\\test\\opertest\\'
        target_datapath = 'D:\\data\\KD\\test\\plat_test\\'
        target_datapath = 'D:\\data\\KD\\test\\self_test1\\'
        # outputpath = 'data\\test\\box_test0_X_256.npy'
        # outputpath = 'data\\test\\box_test0_X_256.npy'
        outputpath = 'data\\test\\box_test2_X_256.npy'
        # target_datapath = 'box_test3_X.npy'
        # target_datapath = 'box_test2_X.npy'
        # target_datapath = 'box_test1_X.npy'
        # target_datapath = 'D:\\data\\KD\\test\\self_test2\\2'
        # outputpath = 'data\\test\\box_test4-2_X_256.npy'
        # target_datapath = 'D:\\data\\KD\\test\\self_test2\\1'
        # outputpath = 'data\\test\\box_test4-1_X_256.npy'
        # target_datapath = 'D:\\data\\KD\\test\\self_test2\\0'
        # outputpath = 'data\\test\\box_test4-0_X_256.npy'
        val_X = test_set_generation(path = target_datapath,outputpath=outputpath)
        target_datapath_lst = [outputpath]
        target_datapath_GT= [0]
    else:
        target_datapath_lst = ['data\\test\\box_test1_X.npy','data\\test\\box_test2_X.npy','data\\test\\box_test3_X.npy'
                       ,'data\\test\\box_test4-0_X.npy','data\\test\\box_test4-1_X.npy','data\\test\\box_test4-2_X.npy']
        target_datapath_lst = ['data\\test\\box_test1_X_256.npy','data\\test\\box_test2_X_256.npy','data\\test\\box_test3_X_256.npy'
                       ,'data\\test\\box_test4-0_X_256.npy','data\\test\\box_test4-1_X_256.npy','data\\test\\box_test4-2_X_256.npy']
        target_datapath_GT= [1,1,0,0,1,1]
    
    prid_lst = []
    gt_lst = []
    for i,target_datapath in enumerate(target_datapath_lst):
        print( '\ntesting dataset - ',target_datapath)
        if mode == 'new_test_set':
            pass
        else:
            if target_datapath.split('.')[-1] == 'npy':
                # vallist = [os.path.join(srcpath,'box_test_X.npy')]
                vallist = [target_datapath]
                val_X = readNConcat(vallist)        
        print('test set count : ',val_X.shape[0])
        val_Y = np.zeros(val_X.shape[0],dtype=int) + target_datapath_GT[i]
        dfvalset = Driftset(val_X,val_Y)

        iter_valid = torch.utils.data.DataLoader(dfvalset, batch_size=2, shuffle=False,
                                             pin_memory=True)


        prid,gt = test_model(model,device,val_loader=iter_valid,dataset_name=target_datapath.split('\\')[-1])
        prid_lst.append(prid)
        gt_lst.append(gt)
        print( '-'*100)

    prid_lst = np.concatenate(prid_lst,axis=0)
    gt_lst = np.concatenate(gt_lst,axis=0)
    metric = MulticlassConfusionMatrix(num_classes=cfg.n_class)
    print(confusion_matrix(prid_lst,gt_lst))
    # print(metric(prid_lst,gt_lst))
    # print(prid_lst)
    
            # memory_format=torch.channels_last)
        
