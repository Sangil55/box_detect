import torch
import numpy as np


class Driftset(torch.utils.data.Dataset):
    def __init__(self, X,Y,Fn=None):
        self.X = X
        self.Y = Y
        # self.Fn = Fn
        # self.df = df
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        # fn = self.Fn[i]

        x = torch.from_numpy(x).float().permute(2,0,1)
        y = torch.tensor(y)
        # fn
        # pos = torch.from_numpy(pos)
        # ft =  torch.from_numpy(ft).float()
        return x,y
