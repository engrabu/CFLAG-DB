import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this missing import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from scipy import stats
import copy

class FemnistCNN(nn.Module):
    """Original CNN model for FEMNIST/FashionMNIST"""
    def __init__(self):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

class CIFAR10CNN(nn.Module):
    """CNN model for CIFAR-10"""
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class HARLSTM(nn.Module):
    """LSTM model for Human Activity Recognition dataset"""
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, num_classes=6):
        super(HARLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2, bidirectional=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: batch_size, seq_length, channels
        # For HAR dataset: batch_size, 128, 9
        
        # Reshape if input is [batch, height, width, channels]
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            x = x.reshape(batch_size, height, width * channels)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return F.log_softmax(out, dim=1)


class IoTAnomalyDetector(nn.Module):
    """Neural network for IoT anomaly detection"""
    def __init__(self, input_size=30, hidden_dim=64):
        super(IoTAnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)  # Binary classification: normal/anomaly
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ChestXrayResNet(nn.Module):
    """Custom ResNet model for chest X-ray classification"""
    def __init__(self, num_classes=14):
        super(ChestXrayResNet, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_residual_block(32, 64, 2)
        self.layer2 = self._make_residual_block(64, 128, 2, stride=2)
        self.layer3 = self._make_residual_block(128, 256, 2, stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_residual_block(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First residual connection might downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Rest of the blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # For multi-label classification, use sigmoid instead of softmax
        return torch.sigmoid(x)
class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

def get_model_for_dataset(dataset_name, device):
    """
    Factory function to create appropriate model for each dataset
    
    Args:
        dataset_name: Name of the dataset
        device: Device to place model on
        
    Returns:
        model: Initialized model
    """
    if dataset_name.lower() == 'fashion_mnist':
        return FemnistCNN().to(device)
    elif dataset_name.lower() == 'cifar10':
        return CIFAR10CNN().to(device)
    elif dataset_name.lower() == 'har':
        return HARLSTM().to(device)
    elif dataset_name.lower() == 'iot':
        return IoTAnomalyDetector().to(device)
    elif dataset_name.lower() == 'chestxray':
        return ChestXrayResNet().to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")