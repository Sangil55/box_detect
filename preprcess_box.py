
import os
import cv2
import webp
import matplotlib.pyplot as plt
import numpy as np
import logging
from config import configModel

from find_rect import find_rect_v1
from sklearn.model_selection import KFold
from midas.run import run_MDE

seed = 0
# mode = 'train'
mode = 'train'
# dsize = (320,240)
# dsize = (640,480)
cfg = configModel()
dsize = cfg.d_size
# D:\src\MiDaS-master


def test_set_generation(path ='D:\\src\\box_detect\\out\\kart_test\\',outputpath = 'data\\test.npy'):
    list_files = (find_files(path))
    print(len(list_files))
    arr_data = []
    for file_ in list_files:
        img_origin,img_opening,img_cany,img_sobeix,img_sobeiy = find_rect_v1(file_)    
        #resize data to 3x4
        reimg_origin = cv2.resize(img_origin, dsize=dsize, interpolation=cv2.INTER_AREA)
        reimg_opening = cv2.resize(img_opening, dsize=dsize, interpolation=cv2.INTER_AREA)
        reimg_cany = cv2.resize(img_cany, dsize=dsize, interpolation=cv2.INTER_AREA)
        reimg_sobeix = cv2.resize(img_sobeix, dsize=dsize, interpolation=cv2.INTER_AREA)
        reimg_sobeiy = cv2.resize(img_sobeiy, dsize=dsize, interpolation=cv2.INTER_AREA)

        # print (reimg_opening[:,:,np.newaxis].shape)
        img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis],reimg_sobeix[:,:,np.newaxis],reimg_sobeiy[:,:,np.newaxis] ], axis=2)
        arr_data.append(img_5ch)
    
    arr_data = np.array(arr_data)
    # Depth estimation inference (input_path, output_path, model_path, model_type="dpt_beit_large_512"
    model_path = cfg.depth_model_path
    model_type = cfg.depth_model_type
    img_depth = run_MDE(input_path = list_files,output_path = None,model_path =model_path , model_type = model_type)
    
    arr_data = np.concatenate([arr_data,img_depth[:,:,:,np.newaxis]],axis=3)

    np.save(outputpath, arr_data)
    return arr_data


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
     

    if mode == 'train':
        negative_list=[]
        p0 = 'D:\\data\\KD\\self_2303\\0'
        negative_list.extend(find_files(p0))
        negative_list.extend(find_files(p0,'jpg'))
        ratio = len(negative_list) / 1000
        pick = np.random.random(len(negative_list))< 1/ratio
        negative_list = np.array(negative_list)[pick].tolist()
        print(len(negative_list))

        p1 = 'D:\\data\\KD\\self_2303\\1'
        negative_list.extend(find_files(p1))

        positive_list=[]
        p2 = 'D:\\data\\KD\\self_2303\\2'
        positive_list.extend(find_files(p2))
        p3 = 'D:\\data\\KD\\self_2303\\3'
        positive_list.extend(find_files(p3))

        abnormal_list=[]
        p4 = 'D:\\data\\KD\\self_2303\\4'
        abnormal_list.extend(find_files(p4))

        negative_arr_data = []

        for file_ in negative_list:
            img_origin,img_opening,img_cany,img_sobeix,img_sobeiy = find_rect_v1(file_)    
            #resize data to 3x4
            reimg_origin = cv2.resize(img_origin, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_opening = cv2.resize(img_opening, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_cany = cv2.resize(img_cany, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeix = cv2.resize(img_sobeix, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeiy = cv2.resize(img_sobeiy, dsize=dsize, interpolation=cv2.INTER_AREA)

            # print (reimg_opening[:,:,np.newaxis].shape)
            img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis],reimg_sobeix[:,:,np.newaxis],reimg_sobeiy[:,:,np.newaxis] ], axis=2)
            negative_arr_data.append(img_5ch)
        
        negative_arr_data = np.array(negative_arr_data)

        # Depth estimation inference (input_path, output_path, model_path, model_type="dpt_beit_large_512"
        # model_path = 'midas\\weights\\'+'dpt_swin2_base_384.pt'
        # model_type = 'dpt_swin2_base_384.pt'
        model_path = 'midas\\weights\\'+'midas_v21_small_256.pt'
        model_type = 'midas_v21_small_256'
        # midas_v21_small_256.pt
        img_depth = run_MDE(input_path = negative_list,output_path = None,model_path =model_path , model_type = model_type)
        
        negative_arr_data = np.concatenate([negative_arr_data,img_depth[:,:,:,np.newaxis]],axis=3)
        Y = np.zeros(len(negative_arr_data),dtype=int)
        
        poitive_arr_data = []

        for file_ in positive_list:
            img_origin,img_opening,img_cany,img_sobeix,img_sobeiy = find_rect_v1(file_)
            #resize data to 3x4
            reimg_origin = cv2.resize(img_origin, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_opening = cv2.resize(img_opening, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_cany = cv2.resize(img_cany, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeix = cv2.resize(img_sobeix, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeiy = cv2.resize(img_sobeiy, dsize=dsize, interpolation=cv2.INTER_AREA)

            # print (reimg_opening[:,:,np.newaxis].shape)
            img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis],reimg_sobeix[:,:,np.newaxis],reimg_sobeiy[:,:,np.newaxis]], axis=2)
            poitive_arr_data.append(img_5ch)

        poitive_arr_data = np.array(poitive_arr_data)
        # Depth estimation inference (input_path, output_path, model_path, model_type="dpt_beit_large_512"
        
        # model_path = 'midas\\weights\\'+'dpt_swin2_base_384.pt'
        # model_type = 'dpt_swin2_base_384.pt'
        model_path = 'midas\\weights\\'+'midas_v21_small_256.pt'
        model_type = 'midas_v21_small_256'

        img_depth = run_MDE(input_path = positive_list,output_path = None,model_path =model_path , model_type = model_type)
        poitive_arr_data = np.concatenate([poitive_arr_data,img_depth[:,:,:,np.newaxis]],axis=3)

        # X = np.concatenate([negative_arr_data,poitive_arr_data],axis=0)
        Y = np.concatenate([Y,np.ones(len(poitive_arr_data),dtype=int)],axis=0)

        abnormal_arr_data = []

        for file_ in abnormal_list:
            img_origin,img_opening,img_cany,img_sobeix,img_sobeiy = find_rect_v1(file_)
            #resize data to 3x4
            reimg_origin = cv2.resize(img_origin, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_opening = cv2.resize(img_opening, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_cany = cv2.resize(img_cany, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeix = cv2.resize(img_sobeix, dsize=dsize, interpolation=cv2.INTER_AREA)
            reimg_sobeiy = cv2.resize(img_sobeiy, dsize=dsize, interpolation=cv2.INTER_AREA)

            # print (reimg_opening[:,:,np.newaxis].shape)
            img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis],reimg_sobeix[:,:,np.newaxis],reimg_sobeiy[:,:,np.newaxis]], axis=2)
            abnormal_arr_data.append(img_5ch)

        abnormal_arr_data = np.array(abnormal_arr_data)
        # Depth estimation inference (input_path, output_path, model_path, model_type="dpt_beit_large_512"
        # model_path = 'midas\\weights\\'+'dpt_swin2_base_384.pt'
        # model_type = 'dpt_swin2_base_384.pt'
        model_path = 'midas\\weights\\'+'midas_v21_small_256.pt'
        model_type = 'midas_v21_small_256'

        img_depth = run_MDE(input_path = abnormal_list,output_path = None,model_path =model_path , model_type = model_type)
        abnormal_arr_data = np.concatenate([abnormal_arr_data,img_depth[:,:,:,np.newaxis]],axis=3)

        X = np.concatenate([negative_arr_data,poitive_arr_data,abnormal_arr_data],axis=0)
        Y = np.concatenate([Y,np.ones(len(abnormal_arr_data) ,dtype=int)*2 ],axis=0)

        
        print(negative_arr_data.shape,poitive_arr_data.shape,Y.shape)
    
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        for i, (idx_train, idx_test) in enumerate(kf.split(Y)):
            # print(idx_test)
            data_train = X[idx_train]
            data_val = X[idx_test]

            y_train = Y[idx_train]
            y_val = Y[idx_test]

            # np.save('data\\box_train_X_fold'+str(i)+'.npy', data_train)
            np.save('data\\box_val_X_fold'+str(i)+'_224.npy', data_val)

            # np.save('data\\box_train_Y_fold'+str(i)+'.npy', y_train)
            np.save('data\\box_val_Y_fold'+str(i)+'_224.npy', y_val)

    if mode == 'test':
        p0 = 'D:\\src\\box_detect\\out\\kart_test\\'
        list_files = (find_files(p0))
        print(len(list_files))
        arr_data = []
        for file_ in list_files:
            img_origin,img_opening,img_cany,img_sobeix,img_sobeiy = find_rect_v1(file_)    
            #resize data to 3x4
            reimg_origin = cv2.resize(img_origin, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            reimg_opening = cv2.resize(img_opening, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            reimg_cany = cv2.resize(img_cany, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # print (reimg_opening[:,:,np.newaxis].shape)
            img_5ch = np.concatenate([reimg_origin,reimg_opening[:,:,np.newaxis],reimg_cany[:,:,np.newaxis]], axis=2)
            arr_data.append(img_5ch)
        arr_data = np.array(arr_data)
        np.save('data\\self_test1.npy', arr_data)

            # np.save('data\\box_val_X_fold'+str(i)+'.npy', data_val)



