import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class IRCNN(nn.Module):
    def __init__(self, cfg):
        super(IRCNN, self).__init__()
        self.h, self.w = cfg['data']['image_size']['h'], cfg['data']['image_size']['w']
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # Assuming IR images are grayscale, hence 1 channel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * ((self.h//2)//2) * ((self.w//2)//2), 128)
        self.fc2 = nn.Linear(128, 4) # Output: [x1, y1, x2, y2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64 * ((self.h//2)//2) * ((self.w//2)//2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
