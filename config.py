import numpy as np
class configModel:
    def __init__(self):
        self.input_datapath = ''
        self.in_ch = 1
        # self.in_ch_range=[4:8]
        self.d_size = (256,256)#(384,384)
        self.n_class=3
        self.model_archive_path = 'archive\\'+str(self.in_ch)+'ch'
        
        # model = UNet(n_channels=5, n_classes=2, bilinear=True)
        # model_path = 'midas\\weights\\'+'dpt_swin2_base_384.pt'
        # model_type = 'dpt_swin2_base_384'
        # model_path = 'midas\\weights\\'+'dpt_next_vit_large_384.pt'
        # model_type = 'dpt_next_vit_large_384'
        self.depth_model_path = 'midas\\weights\\'+'midas_v21_small_256.pt'
        self.depth_model_type = 'midas_v21_small_256'

    def ch_select(self,x):
        return x[:,:,:,0:3]
        # return np.expand_dims(x[:,:,:,7],axis=3)
        
    def ch_select2(self,x):
        return x[:,0:3,:,:]
        # return x[:,7,:,:].unsqueeze(1)