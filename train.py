import torch
from torch import nn
from config import *
import torch.optim as optim
from models.generator import PPRL_VGAN_Generator
from models.discriminator import PPRL_VGAN_Discriminator
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.VAE.Decoder import PPRL_VGAN_Decoder
from data.data import dataloader
from data.data import device
import matplotlib.pyplot as plt
from data.weightinitialize import weights_init_pprl
import os
# 基础损失函数
criterion_GAN = nn.BCELoss()           # 用于 D1 (真假判别)
criterion_Class = nn.CrossEntropyLoss() # 用于 D2, D3 (多分类任务)

#判别器损失Ld
def compute_D_loss(D, real_imgs, fake_imgs, real_id_labels, real_exp_labels):
    # --- 1. 处理真实图像 ---
    d1_real, d2_real, d3_real = D(real_imgs)

    # D1 损失: 识别真实图像为 1 [1]
    label_real = torch.full_like(d1_real, 1.0)
    loss_D1_real = criterion_GAN(d1_real, label_real)

    # D2/D3 损失: 识别真实图像的身份和表情 [1]
    loss_D2 = criterion_Class(d2_real, real_id_labels)
    loss_D3 = criterion_Class(d3_real, real_exp_labels)

    # --- 2. 处理合成图像 ---
    d1_fake, _, _ = D(fake_imgs.detach())

    # D1 损失: 识别合成图像为 0 [1]
    label_fake = torch.full_like(d1_fake, 0.0)
    loss_D1_fake = criterion_GAN(d1_fake, label_fake)

    # --- 3. 加权求和 (公式 5) ---
    loss_D = lambda_D1 * (loss_D1_real + loss_D1_fake) + \
             lambda_D2 * loss_D2 + \
             lambda_D3 * loss_D3

    return loss_D

#生成器损失函数Lg
def compute_G_loss(D, fake_imgs, target_id_labels, original_exp_labels, mu, logvar):
    d1_fake, d2_fake, d3_fake = D(fake_imgs)

    # G1: 对抗损失 (欺骗 D1，使其认为合成图是 1) [2, 4]
    label_real = torch.full_like(d1_fake, 1.0)
    loss_G1 = criterion_GAN(d1_fake, label_real)

    # G2: 身份保持损失 (使 D2 识别出目标身份 c) [2]
    loss_G2 = criterion_Class(d2_fake, target_id_labels)

    # G3: 表情保持损失 (使 D3 识别出原始表情 ye) [2]
    loss_G3 = criterion_Class(d3_fake, original_exp_labels)

    # G4: KL 散度损失 (约束潜在表示符合正态分布) [2, 5]
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss_G4 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 加权求和 (公式 6) [2, 3]
    loss_G = lambda_G1 * loss_G1 + \
             lambda_G2 * loss_G2 + \
             lambda_G3 * loss_G3 + \
             lambda_G4 * loss_G4

    return loss_G

if __name__ == "__main__":

    #实例化Generator
    # 1. 实例化编码器
    encoder = PPRL_VGAN_Encoder(nc, nz)
    # 2. 实例化解码器
    decoder = PPRL_VGAN_Decoder(nz, n_id, nc)
    netG=PPRL_VGAN_Generator(encoder,decoder)

    #实例化Discriminator
    netD=PPRL_VGAN_Discriminator()


    netG.apply(weights_init_pprl)
    netD.apply(weights_init_pprl)
    # 使用 RMSprop 优化器而非 Adam [3]
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)

    # Training Loop


    num_epochs = 200
    # 简化版训练循环逻辑
    # 用于追踪损失变化的列表 [1]
    G_losses = []
    D_losses = []
    # 用于追踪各分项损失（可选，便于调试）
    D1_losses, D2_losses, D3_losses = [], [], []

    # 创建权重保存目录
    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (imgs, id_labels, exp_labels) in enumerate(dataloader):
            # 将数据移动到设备
            real_imgs = imgs.to(device)
            real_id_labels = id_labels.to(device)
            real_exp_labels = exp_labels.to(device)
            b_size = real_imgs.size(0)

            ############################
            # (1) 更新判别器 D：最大化公式 (5) [3, 5, 6]
            ###########################
            netD.zero_grad()

            # 准备目标身份代码 c (One-hot)
            # 随机生成目标身份用于合成 fake 图像 [7, 8]
            target_id_idx = torch.randint(0, 6, (b_size,)).to(device)
            c = torch.nn.functional.one_hot(target_id_idx, num_classes=6).float().to(device)

            # 生成合成图像 [8]
            fake_imgs, mu, logvar = netG(real_imgs, c)

            # 计算判别器损失 (使用之前定义好的 compute_D_loss)
            errD = compute_D_loss(netD, real_imgs, fake_imgs, real_id_labels, real_exp_labels)
            errD.backward()
            optimizerD.step()

            ############################
            # (2) 更新生成器 G：更新两次 [3]
            ###########################
            for _ in range(2):
                netG.zero_grad()
                # 重新生成图像以保持梯度流 [8]
                fake_imgs, mu, logvar = netG(real_imgs, c)

                # 计算生成器损失 (使用之前定义好的 compute_G_loss)
                # 目标是欺骗 D1, 匹配目标身份 c, 保留原表情 ye, 满足 KL 散度 [9]
                errG = compute_G_loss(netD, fake_imgs, target_id_idx, real_exp_labels, mu, logvar)
                errG.backward()
                optimizerG.step()

            # 记录损失 [1]
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # 每 50 个 batch 输出一次统计状态 [1, 10]
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # --- 每个 Epoch 结束后的操作 ---

        # 1. 自动保存模型权重
        torch.save(netG.state_dict(), f'{save_dir}/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'{save_dir}/netD_epoch_{epoch}.pth')
        print(f"Weights saved for epoch {epoch}")






    # 2. 实时保存当前损失图像 (防止训练中断无法查看) [2]
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_curve_epoch_{epoch}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Final Generator and Discriminator Loss")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show() # [2]