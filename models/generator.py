from torch import nn

# Generator Code
class PPRL_VGAN_Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(PPRL_VGAN_Generator, self).__init__()
        # 整合之前定义的编码器和解码器 [1, 3]
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, identity_code):
        """
        img: 输入的 64x64 图像 I [7, 9]
        identity_code: 目标身份的 One-hot 向量 c [4, 5]
        """
        # 1. 通过编码器获得身份无关特征 f(I)
        # 注意：在 VAE 模式下，encoder 通常返回 mu 和 logvar [8, 11]
        f_I, mu, logvar = self.encoder(img)

        # 2. 通过解码器进行合成
        # 解码器内部会执行 f(I) 与 c 的拼接 (⊕) [3, 5]
        synthesized_img = self.decoder(f_I, identity_code)

        # 返回合成图像以及用于计算 KL 散度的 mu 和 logvar [8]
        return synthesized_img, mu, logvar

