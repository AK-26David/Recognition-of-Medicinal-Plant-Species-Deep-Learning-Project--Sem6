import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3):
        super(MultiScaleCNN, self).__init__()
        
        # Convolutional Blocks
        self.conv1 = self._conv_block(3, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 256)
        self.conv5 = self._conv_block(256, 512)
        
        # Adaptive Pooling to ensure fixed-size feature maps
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer with Multi-Scale Feature Fusion
        self.fc = nn.Linear(32 + 64 + 128 + 256 + 512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduce spatial dimensions
        )
    
    def forward(self, x):
        # Feature Extraction
        x1 = self.conv1(x)  # (B, 32, H/2, W/2)
        x2 = self.conv2(x1) # (B, 64, H/4, W/4)
        x3 = self.conv3(x2) # (B, 128, H/8, W/8)
        x4 = self.conv4(x3) # (B, 256, H/16, W/16)
        x5 = self.conv5(x4) # (B, 512, H/32, W/32)
        
        # Global Pooling for Multi-Scale Feature Fusion
        x1_pool = self.global_pool(x1).view(x.size(0), -1)
        x2_pool = self.global_pool(x2).view(x.size(0), -1)
        x3_pool = self.global_pool(x3).view(x.size(0), -1)
        x4_pool = self.global_pool(x4).view(x.size(0), -1)
        x5_pool = self.global_pool(x5).view(x.size(0), -1)
        
        # Concatenate Multi-Scale Features
        x_fused = torch.cat([x1_pool, x2_pool, x3_pool, x4_pool, x5_pool], dim=1)
        x_fused = self.dropout(x_fused)
        
        # Classification Layer
        out = self.fc(x_fused)
        return out

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='path_to_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example Usage
if __name__ == "__main__":
    num_classes = len(dataset.classes)  # Dynamically determine number of species
    model = MultiScaleCNN(num_classes=num_classes)
    sample_input, _ = next(iter(dataloader))  # Get a batch of images
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (batch_size, num_classes)
