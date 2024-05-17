from torch import nn
import torch
from torch.nn import functional as F
import math


def pos_encoding(t, channels, device="cpu"):
    inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
    ).to(device)
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


class TimeEmbeddings(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.stack = nn.Sequential(
            nn.Linear(channels, channels),
            #nn.ReLU(),
            #nn.Linear(channels, channels),
            nn.ReLU(),
        )
    
    def forward(self, time):
        out = self.stack(pos_encoding(time, self.channels, device=time.device))
        return out[:,:, None, None]


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_embed = TimeEmbeddings(out_channels)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x) + self.time_embed(t).to(x.device)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_embed = TimeEmbeddings(out_channels)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) + self.time_embed(t).to(x.device)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.down5 = (Down(1024, 2048))
        self.up1 = (Up(2048, 1024))
        self.up2 = (Up(1024, 512))
        self.up3 = (Up(512, 256))
        self.up4 = (Up(256, 128))
        self.up5 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x6 = self.down5(x5, t)
        x = self.up1(x6, x5, t)
        x = self.up2(x, x4, t)
        x = self.up3(x, x3, t)
        x = self.up4(x, x2, t)
        x = self.up5(x, x1, t)
        logits = self.outc(x)
        return logits

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


class DiffusionModel(nn.Module):
    def __init__(self,device="cpu", lr=0.00001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNet(3, 3).to(device)
        self.optimizer = torch.optim.SGD(self.unet.parameters(), lr=lr)
        self.criterion = nn.L1Loss()
    
    def forward(self, x, t):
        return self.unet(x, t)
    
    def training_step(self, x_0, x_1, t):
        self.optimizer.zero_grad()
        x_pred = self.unet(x_0, t)
        loss = self.criterion(x_pred, x_1)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def save(self, file_name='models/catfusion.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='models/catfusion.pth'):
        self.load_state_dict(torch.load(file_name))