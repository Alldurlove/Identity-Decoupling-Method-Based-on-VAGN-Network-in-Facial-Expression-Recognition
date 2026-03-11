from config import *
import torch.nn as nn
from data.weightinitialize import weights_init_pprl
import torch

class PPRL_VGAN_Discriminator(nn.Module):
    def __init__(self, nc=nc, n_id=n_id, n_e=n_e):

        super(PPRL_VGAN_Discriminator, self).__init__()

        # 1. 共享卷积层 (Shared Convolutional Layers)
        # 根据 Table 1: 5x5 卷积, 步长 2 (下采样), BatchNorm, LeakyReLU [2]
        self.shared_conv = nn.Sequential(
            # Layer 1: 5x5x32
            nn.Conv2d(nc, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 5x5x64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 5x5x128
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 5x5x256
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. 共享全连接层 (Shared FC Layer)
        # 根据 Table 1: 256 维全连接, LeakyReLU [2, 5]
        # 输入维度: 256通道 * 4*4特征图 = 4096
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 独立输出分支 (Independent Output Branches) [1, 5]
        # D1: 真假图像判别 (Binary)
        self.d1_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率 D1(I) [6, 7]
        )

        # D2: 身份识别 (Person Identifier)
        self.d2_head = nn.Linear(256, n_id)

        # D3: 表情分类 (Expression Classifier)
        self.d3_head = nn.Linear(256, n_e)

    def forward(self, x):
        features = self.shared_conv(x)
        shared_embedding = self.shared_fc(features)

        # 同时返回三个任务的结果
        d1_out = self.d1_head(shared_embedding).view(-1)
        d2_out = self.d2_head(shared_embedding)
        d3_out = self.d3_head(shared_embedding)

        return d1_out, d2_out, d3_out

