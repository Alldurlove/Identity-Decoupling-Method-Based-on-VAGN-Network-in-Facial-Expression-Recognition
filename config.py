

dataroot = "E:\FERG_DB_256"
workers = 2
batch_size = 256
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
n_e=7
beta1 = 0.5
ngpu = 1
n_id=6 #n_id: 身份代码 c 的长度 (FERG 数据集为 6)
lambda_D1, lambda_D2, lambda_D3 = 0.25, 0.5, 0.25   # 判别器 D 的损失权重 (公式 5)
lambda_G1, lambda_G2, lambda_G3, lambda_G4 = 0.108, 0.6, 0.29, 0.002# 生成器 G 的损失权重 (公式 6)