import torch.nn as nn
from .fundamental import *


widefactor = 16

class Segment_body(nn.Module):

    def __init__(self,n_channels, n_classes, bilinear=False, mode ='pretrain'):
        super(Segment_body, self).__init__()
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.widefactor = 4

        self.inc = (DoubleConv(n_channels, 64//self.widefactor))
        self.down1 = (Down(64, 128, wide_factor=self.widefactor))
        self.down2 = (Down(128, 256, wide_factor=self.widefactor))
        self.down3 = (Down(256, 512, wide_factor=self.widefactor))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, wide_factor=self.widefactor))
        self.up1 = (Up(1024, 512 // factor, bilinear, wide_factor=self.widefactor))
        self.up2 = (Up(512, 256 // factor, bilinear, wide_factor=self.widefactor))
        self.up3 = (Up(256, 128 // factor, bilinear, wide_factor=self.widefactor))
        self.up4 = (Up(128, 64, bilinear, wide_factor=self.widefactor))

        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(640*480*64//widefactor,10)
        # self.fc2 = nn.Linear(10,1)
        self.outc = (OutConv(64//self.widefactor, n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.mode == 'pretrain':
            return x
        elif self.mode == 'classify':
            return x5
        elif self.mode == 'segment':
             y = self.outc (x)
             y = self.softmax(y)
             return y
        

        # return x5,y

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        # self.last_hidden_state_zero_layer = nn.Linear(...)
        # self.logit_layer = nn.Linear(...)


class Reconstruct_body(nn.Module):
    def __init__(self,n_channels, n_classes=17, bilinear=False, mode ='pretrain'):
        super(Reconstruct_body, self).__init__()
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = (DoubleConv(n_channels, 64//widefactor))
        self.down1 = (Down(64, 128, wide_factor=widefactor))
        self.down2 = (Down(128, 256, wide_factor=widefactor))
        self.down3 = (Down(256, 512, wide_factor=widefactor))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, wide_factor=widefactor))
        self.up1 = (Up(1024, 512 // factor, bilinear, wide_factor=widefactor))
        self.up2 = (Up(512, 256 // factor, bilinear, wide_factor=widefactor))
        self.up3 = (Up(256, 128 // factor, bilinear, wide_factor=widefactor))
        self.up4 = (Up(128//widefactor, 128//widefactor, bilinear, wide_factor=1))

        # self.flatten = nn.Flatten()
        self.outc = (OutConv(128, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.mode == 'pretrain' or self.mode == 'segment':
            return x
        elif self.mode == 'classify':
            return x5
        # elif self.mode == 'segment':
        #     logits = self.outc(x)
        #     return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        # self.last_hidden_state_zero_layer = nn.Linear(...)
        # self.logit_layer = nn.Linear(...)


class dptAug(nn.Module):
    def __init__(self, Pretrained_model, mode , seg_label_num = 8):
        super(dptAug, self).__init__()
        self.mode = mode
        self.DPTbackbone = Pretrained_model
        
        #for depth > segment consistency
        self.seg_body = Segment_body(n_channels=1, n_classes=seg_label_num, mode = mode)
        #for regularization, domain adaptaion
        self.recon_body = Reconstruct_body(n_channels=1, n_classes=2, mode = mode)
        

        self.conv_reduce1 = conv_1x1(64,16)
        self.conv_reduce2 = conv_1x1(64,16)
        self.abnormal_head = nn.Sequential(
            
            nn.Linear(16*16*16*2,100),
            nn.Linear(100,2) )
        self.selected_out = {}
        self.fhooks = []

        # for i,l in enumerate(list(self.DPTbackbone._modules.keys())):

        #     self.fhooks.append(getattr(self.DPTbackbone,l).register_forward_hook(self.forward_hook(l)))
        self.DPTbackbone._modules['scratch'].output_conv[6].register_forward_hook(self.forward_hook('output_conv'))
        # self.conv_reduce_ = conv_1x1(1392,128)
        # self.head_ab = nn.Sequential(
        #         nn.Linear(128*8*8,100),
        #         nn.Linear(100,2)       )
        # self.head_ab = nn.Linear(128*8*8,100)
        # self.head1_abnormal_classification = nn.Linear()
        # self.head2_for

    def forward(self, x):
        x = self.DPTbackbone(x)
        x = x.unsqueeze(1)
        # print(x.shape)
        if self.mode == 'pretrain':
            seg_x = self.seg_body(x)
            recon_x = self.recon_body(x)

            return recon_x,seg_x
        elif self.mode == 'classify':
            latent_seg_x = self.seg_body(x)
            latent_recon_x = self.recon_body(x)
            # print(latent_seg_x.shape)
            latent_seg_x = self.conv_reduce1(latent_seg_x)
            latent_recon_x = self.conv_reduce1(latent_recon_x)

            latent_seg_x = nn.Flatten()(latent_seg_x)
            latent_recon_x = nn.Flatten()(latent_recon_x)

            latent_x = torch.cat([latent_seg_x,latent_recon_x],axis=1)
            y = self.abnormal_head(latent_x)
            return y
            #latent_x > head to classify
        elif self.mode == 'segment':
            seg_x = self.seg_body(x)
            recon_x = self.recon_body(x)
            # seg_x
            return recon_x,seg_x
        # full_x = torch.cat([seg_x,recon_x],axis=2)
        
        # print(full_x.shape)

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = torch.squeeze(output,dim=1)
        return hook
    def get_last_hook(self):
        return self.selected_out['output_conv']
    
    def set_mode(self,mode):
        self.mode = mode
        self.seg_body.mode = mode
        self.recon_body.mode = mode



        
        
