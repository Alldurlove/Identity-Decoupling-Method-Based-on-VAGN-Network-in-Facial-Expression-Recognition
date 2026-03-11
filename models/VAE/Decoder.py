import torch
import torch.nn as nn
from config import *

class PPRL_VGAN_Decoder(nn.Module):
    def __init__(self, nz=nz, n_id=n_id, nc=nc):

        super(PPRL_VGAN_Decoder, self).__init__()

        # 输入维度 = 潜在表示维度 (128) + 身份代码维度 (n_id)
        self.input_dim = nz + n_id

        # Layer 1: 2048 维全连接层
        # 后接 Reshape 为 4x4x128 维度的特征图
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 根据 Table 1 设计的反卷积层序列
        # 使用 5x5 卷积核，步长为 2 (上采样 ↑)
        # 按照论文要求，除最后一层外均使用 Batch Normalization
        self.deconv_layers = nn.Sequential(
            # Layer 2: 输入 4x4, 输出 8x8, 通道数变更为 256
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 输出 16x16, 通道数变更为 128
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 输出 32x32, 通道数变更为 64
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 输出 64x64, 通道数 3, 激活函数为 Tanh
            # 注意：最后一层不加 Batch Normalization
            nn.ConvTranspose2d(64, nc, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, f_I, c):
        """
        f_I: 编码器提取的特征 (Batch, 128)
        c: 身份 One-hot 代码 (Batch, n_id)
        """
        # 核心步骤：将 f(I) 与 c 进行拼接
        combined_input = torch.cat([f_I, c], dim=1)

        # 全连接映射并重塑形状为 4x4x128
        out = self.fc1(combined_input)
        out = out.view(-1, 128, 4, 4)

        # 通过反卷积序列生成图像
        out = self.deconv_layers(out)
        return out



