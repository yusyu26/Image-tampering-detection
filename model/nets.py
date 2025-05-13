import torch, torch.nn as nn

class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def dw_conv(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in),
                nn.Conv2d(c_in, c_out, 1), nn.ReLU())
        self.net = nn.Sequential(
            dw_conv(in_ch, 32),
            nn.MaxPool2d(2),
            dw_conv(32, 64),
            nn.MaxPool2d(2),
            dw_conv(64,128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
