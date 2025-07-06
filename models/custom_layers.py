import torch
import torch.nn as nn
import torch.nn.functional as F


class L1RegularizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, l1_lambda=0.001): 
        super().__init__()
        self.l1_lambda = l1_lambda
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.conv(x)
    
    def l1_loss(self):
        return self.l1_lambda * torch.sum(torch.abs(self.conv.weight))
    

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map
    

class LearnableSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    

class LpPool2d(nn.Module):
    def __init__(self, p=2, kernel_size=2, stride=2):
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        return F.avg_pool2d(x.pow(self.p), self.kernel_size, self.stride).pow(1./self.p)
    

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = L1RegularizedConv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.attn1 = SpatialAttention(32)
        self.swish1 = LearnableSwish()
        self.pool1 = LpPool2d(p=1.5)
        
        self.conv2 = L1RegularizedConv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = SpatialAttention(64)
        self.swish2 = LearnableSwish()
        self.pool2 = LpPool2d(p=1.5)
        
        self.fc = nn.Linear(64*7*7, 10)
        
        self._init_weights() 
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.swish1(self.bn1(self.conv1(x)))  
        x = self.attn1(x)
        x = self.pool1(x)
        
        x = self.swish2(self.bn2(self.conv2(x)))  
        x = self.attn2(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def total_l1_loss(self):
        return self.conv1.l1_loss() + self.conv2.l1_loss()