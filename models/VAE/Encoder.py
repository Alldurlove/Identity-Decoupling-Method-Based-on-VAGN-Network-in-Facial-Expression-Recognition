import torch
import torch.nn as nn
from config import nz,nc


class PPRL_VGAN_Encoder(nn.Module):
    def __init__(self, nc=3, nz=128):
        super(PPRL_VGAN_Encoder, self).__init__()

        # 共享卷积层 (来源: Table 1)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, 32, 5, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 拆分为两个分支：均值 mu 和 对数方差 logvar (来源: [1, 2])
        # 输入维度: 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(4096, nz)
        self.fc_logvar = nn.Linear(4096, nz)

    def reparameterize(self, mu, logvar):
        """ 重参数化技巧: z = mu + std * eps (来源: [2]) """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.conv_layers(x)
        h = torch.flatten(h, 1)

        # 计算分布参数
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 采样获得 f(I)
        f_I = self.reparameterize(mu, logvar)

        # 必须返回 3 个值，以匹配 Generator 的调用
        return f_I, mu, logvar

