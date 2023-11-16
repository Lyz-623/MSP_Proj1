import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):  # 继承nn.Module

    # 子模块创建
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()  # 对父类nn.Module的属性进行初始化

        # 序贯模型定义网络架构
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 归一化处理，防止数据过大导致网络不稳定
            nn.Dropout2d(0.3),  # Dropout2d防止过拟合
            nn.LeakyReLU(),  # 激活函数
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    # 前向传播
    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # 卷积操作
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)
        self.c5 = ConvBlock(512, 1024)
        self.c6 = ConvBlock(1024, 512)
        self.c7 = ConvBlock(512, 256)
        self.c8 = ConvBlock(256, 128)
        self.c9 = ConvBlock(128, 64)

        # 降采样操作
        self.d1 = DownSample(64)
        self.d2 = DownSample(128)
        self.d3 = DownSample(256)
        self.d4 = DownSample(512)

        # 上采样操作
        self.u1 = UpSample(1024)
        self.u2 = UpSample(512)
        self.u3 = UpSample(256)
        self.u4 = UpSample(128)

        # 输出分割map
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet(3)
    print(net(x).shape)
