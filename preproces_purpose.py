
import os
import cv2
import webp
import matplotlib.pyplot as plt
import numpy as np
import logging

from find_rect import find_rect_v1
from sklearn.model_selection import KFold


seed = 0

def find_files(dir_path,endwith='webp'):
    cfiles = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(endwith):
                cfiles.append(os.path.join(root, file))
    return cfiles

if __name__ == '__main__':
    
    # p1='D:\\data\\KD\\oper1'
    # lst_files = os.listdir('D:\\data\\KD\\oper1')

    c0_list = []
    c1_list = []
    c2_list = []
    p0 = 'D:\\data\\KD\\self_2303\\0'
    c0_list.extend(find_files(p0))
    c0_list.extend(find_files(p0,'jpg'))
    ratio = len(c0_list) / 1000
    pick = np.random.random(len(c0_list))< 1/ratio
    c0_list = np.array(c0_list)[pick].tolist()
    p2 = 'D:\\data\\KD\\self_2303\\2'
    c0_list.extend(find_files(p2))
    
    # print(len(negative_list))

    p1 = 'D:\\data\\KD\\self_2303\\1'
    c1_list.extend(find_files(p1))
    p3 = 'D:\\data\\KD\\self_2303\\3'
    c1_list.extend(find_files(p3))

    p4 = 'D:\\data\\KD\\self_2303\\4'
    c2_list.extend(find_files(p4))

    c0_arr_data = []
    for file_ in c0_list:
        img_origin,img_opening,img_cany = find_rect_v1(file_)    
        #resize data to 3x4
        reimg_origin = cv2.resize(img_origin, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_opening = cv2.resize(img_opening, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_cany = cv2.resize(img_cany, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        # print (reimg_opening[:,:,np.newaxis].shape)
        img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis]], axis=2)
        c0_arr_data.append(img_5ch)
    c0_arr_data = np.array(c0_arr_data)
    Y = np.zeros(len(c0_arr_data),dtype=int)
    
    c1_arr_data = []
    for file_ in c1_list:
        img_origin,img_opening,img_cany = find_rect_v1(file_)    
        #resize data to 3x4
        reimg_origin = cv2.resize(img_origin, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_opening = cv2.resize(img_opening, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_cany = cv2.resize(img_cany, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        # print (reimg_opening[:,:,np.newaxis].shape)
        img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis]], axis=2)
        c1_arr_data.append(img_5ch)

    c1_arr_data = np.array(c1_arr_data)
    X = np.concatenate([c0_arr_data,c1_arr_data],axis=0)
    Y = np.concatenate([Y,np.ones(len(c1_arr_data),dtype=int)],axis=0)

    c2_arr_data = []
    for file_ in c2_list:
        img_origin,img_opening,img_cany = find_rect_v1(file_)    
        #resize data to 3x4
        reimg_origin = cv2.resize(img_origin, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_opening = cv2.resize(img_opening, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        reimg_cany = cv2.resize(img_cany, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        # print (reimg_opening[:,:,np.newaxis].shape)
        img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis]], axis=2)
        c2_arr_data.append(img_5ch)
    c2_arr_data = np.array(c2_arr_data)

    X = np.concatenate([X,c2_arr_data],axis=0)
    Y = np.concatenate([Y,np.ones(len(c2_arr_data),dtype=int)*2],axis=0)
    
    print(c0_arr_data.shape,c1_arr_data.shape,c2_arr_data.shape,Y.shape)
    
    
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for i, (idx_train, idx_test) in enumerate(kf.split(Y)):
        # print(idx_test)
        data_train = X[idx_train]
        data_val = X[idx_test]

        y_train = Y[idx_train]
        y_val = Y[idx_test]

        np.save('data\\data2303\\purpose_train_X_fold'+str(i)+'.npy', data_train)
        np.save('data\\data2303\\purpose_val_X_fold'+str(i)+'.npy', data_val)

        np.save('data\\data2303\\purpose_train_Y_fold'+str(i)+'.npy', y_train)
        np.save('data\\data2303\\purpose_val_Y_fold'+str(i)+'.npy', y_val)




