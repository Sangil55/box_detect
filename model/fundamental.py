import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_kx1(in_channels, out_channels, kernel_size, stride=1, **kwargs):
    """ kx1 convolution with padding """
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False, **kwargs)
    # return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same', bias=False, **kwargs)


    # padding = kernel_size - stride
    # padding_left = padding // 2
    # padding_right = padding - padding_left
    # layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    # layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False))
    # return nn.Sequential(*layers)


def conv_1x1(in_channels, out_channels, **kwargs):
    """ 1x1 convolution """
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1,1 ), bias=False, **kwargs)

def l2normlayer(x,dim=-1,keepdim=True):
    normx = torch.linalg.norm(x, dim=-dim, keepdim=keepdim)
    return x / normx

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        max_ = torch.max(x, 1)
        avg_ = torch.mean(x, 1)
        stacked = torch.cat((max_[0].unsqueeze(1), avg_.unsqueeze(1)), dim=1)
        return stacked


class MLP(nn.Module):
    def __init__(self, ch_list):
        super(MLP, self).__init__()
        self.ch_list = ch_list
        self.relu = nn.ReLU()
        self.mlp = self._stack_layers()

    def _stack_layers(self):
        layers = []
        num_layers = len(self.ch_list)
        if num_layers < 2:
            return nn.Sequential(*layers)
        else:
            for i_ch in range(num_layers-1):
                layers.append(conv_1x1(self.ch_list[i_ch], self.ch_list[i_ch+1]))
                if i_ch != num_layers-1:
                    layers.append(self.relu)
            return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Head_MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim=None, mode='projection'):
        super(Head_MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.act = nn.ReLU(inplace=True)
        self.mode = mode

        if mode == 'projection':
            self.linear2 = nn.Linear(in_dim, mid_dim)
            self.bn2 = nn.BatchNorm1d(mid_dim)

            self.linear3 = nn.Linear(in_dim, mid_dim)
            self.bn3 = nn.BatchNorm1d(mid_dim)
        elif mode == 'prediction':
            self.linear2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.linear1(x)))
        x = self.linear2(x)

        if self.mode == 'projection':
            x = self.act(self.bn2(x))
            x = self.bn3(self.linear3(x))

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wide_factor = 1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels//wide_factor, out_channels//wide_factor)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, wide_factor=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels//wide_factor, out_channels//wide_factor, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels//wide_factor, in_channels//(wide_factor* 2), kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//wide_factor, out_channels//wide_factor)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)