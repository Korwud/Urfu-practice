import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

# 2. Bottleneck Residual блок
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

# 3. Wide Residual блок
class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        mid_channels = out_channels * widen_factor
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return F.relu(out)

# Общая архитектура для тестирования блоков
class ResNetTest(nn.Module):
    def __init__(self, block_type, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Слои с Residual блоками
        self.layer1 = self._make_layer(block_type, 32, 64, 1)
        self.layer2 = self._make_layer(block_type, 64, 128, 2)
        self.layer3 = self._make_layer(block_type, 128, 256, 2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, block_type, in_channels, out_channels, stride):
        if block_type == 'basic':
            return ResidualBlock(in_channels, out_channels, stride)
        elif block_type == 'bottleneck':
            return BottleneckResidualBlock(in_channels, out_channels, stride)
        elif block_type == 'wide':
            return WideResidualBlock(in_channels, out_channels, stride)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RegularizedCNNWithResidual(CNNWithResidual):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

class ConvConfig:
    def __init__(self, kernel_sizes, channels):
        self.kernel_sizes = kernel_sizes
        self.channels = channels


class ConvNetMNIST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(1, config.channels[0], 
                             kernel_size=config.kernel_sizes[0], 
                             padding=config.kernel_sizes[0]//2)
        
        self.fc = nn.Linear(config.channels[0] * 28 * 28, 10)
        
        self.activations = None
        self.conv.register_forward_hook(self.get_activation_hook())
    
    def get_activation_hook(self):
        def hook(module, input, output):
            self.activations = output.detach()
        return hook
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

def get_cnn_models():
    models = {
        "ShallowCNN": nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7, 10)
        ),
        "MediumCNN": nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7, 10)
        ),
        "DeepCNN": nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(128*3*3, 10) 
        ),
        "ResCNN": CNNWithResidual(input_channels=1)
    }
    return models