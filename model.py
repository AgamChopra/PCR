import torch.nn as nn
import torch
#from torch2trt import torch2trt

def conv_block(in_c, out_c):
    
    out = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 5, stride = 2), nn.ReLU(), nn.BatchNorm2d(out_c, track_running_stats = True))
    
    return out

class auto(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.mp = nn.MaxPool2d(2,2)
        self.f1 = conv_block(3, 6)#178
        self.f2 = conv_block(6, 12)#42
        self.f3 = conv_block(12, 24)#8
        self.f4 = nn.Sequential(nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size = 4, stride = 1), nn.ReLU())
        self.f5 = nn.Sequential(nn.Conv2d(in_channels = 48, out_channels = 2, kernel_size = 1, stride = 1), nn.Sigmoid())#1
        
    def forward(self, x):
        
        y = self.f1(x)
        y = self.mp(y)
        y = self.f2(y)
        y = self.mp(y)
        y = self.f3(y)
        y = self.mp(y)
        y = self.f4(y)
        y = self.f5(y)
        
        return y