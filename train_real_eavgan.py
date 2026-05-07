import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config import (
    image_size,
    lambda_D1,
    lambda_D2,
    lambda_D3,
    lambda_G1,
    lambda_G2,
    lambda_G3,
    lambda_G4,
    n_id,
    nc,
    nz,
)
from data.fer2013_dataset import FER2013Dataset
from data.weightinitialize import weights_init_pprl
from models.VAE.Decoder import PPRL_VGAN_Decoder
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.discriminator import PPRL_VGAN_Discriminator
from models.generator import PPRL_VGAN_Generator


criterion_GAN = nn.BCELoss()
criterion_Class = nn.CrossEntropyLoss()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def compute_D_loss(
    net_d: nn.Module,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    real_id_labels: torch.Tensor,
    real_exp_labels: torch.Tensor,
) -> torch.Tensor:
    d1_real, d2_real, d3_real = net_d(real_imgs)
    label_real = torch.full_like(d1_real, 1.0)
    loss_d1_real = criterion_GAN(d1_real, label_real)
    loss_d2 = criterion_Class(d2_real, real_id_labels)
    loss_d3 = criterion_Class(d3_real, real_exp_labels)

    d1_fake, _, _ = net_d(fake_imgs.detach())
    label_fake = torch.full_like(d1_fake, 0.0)
    loss_d1_fake = criterion_GAN(d1_fake, label_fake)
    return (
        lambda_D1 * (loss_d1_real + loss_d1_fake)
        + lambda_D2 * loss_d2
        + lambda_D3 * loss_d3
    )


def compute_G_loss(
    net_d: nn.Module,
    fake_imgs: torch.Tensor,
    target_id_labels: torch.Tensor,
    original_exp_labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    d1_fake, d2_fake, d3_fake = net_d(fake_imgs)
    label_real = torch.full_like(d1_fake, 1.0)
    loss_g1 = criterion_GAN(d1_fake, label_real)
    loss_g2 = criterion_Class(d2_fake, target_id_labels)
    loss_g3 = criterion_Class(d3_fake, original_exp_labels)
    loss_g4 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (
        lambda_G1 * loss_g1
        + lambda_G2 * loss_g2
        + lambda_G3 * loss_g3
        + lambda_G4 * loss_g4
    )


def evaluate_expr_acc(
    net_g: nn.Module,
    net_d: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    net_g.eval()
    net_d.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, _, exp_labels in loader:
            real_imgs = imgs.to(device)
            exp_labels = exp_labels.to(device)
            b_size = real_imgs.size(0)
            target_id_idx = torch.randint(0, n_id, (b_size,), device=device)
            c = torch.nn.functional.one_hot(target_id_idx, num_classes=n_id).float()
            fake_imgs, _, _ = net_g(real_imgs, c)
            _, _, d3_fake = net_d(fake_imgs)
            preds = torch.argmax(d3_fake, dim=1)
            correct += (preds == exp_labels).sum().item()
            total += exp_labels.size(0)
    return correct / max(1, total)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    transform = build_transform()
    train_ds = FER2013Dataset(
        data_root=args.data_root,
        split="train",
        transform=transform,
        data_format=args.data_format,
    )
    val_ds = FER2013Dataset(
        data_root=args.data_root,
        split="val",
        transform=transform,
        data_format=args.data_format,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EAVGAN from scratch on FER2013 real dataset.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="auto", choices=["auto", "csv", "folder"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints/real_from_scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g-updates", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_loaders(args)

    encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
    decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
    net_g = PPRL_VGAN_Generator(encoder, decoder).to(device)
    net_d = PPRL_VGAN_Discriminator().to(device)
    net_g.apply(weights_init_pprl)
    net_d.apply(weights_init_pprl)

    optimizer_g = optim.RMSprop(net_g.parameters(), lr=args.lr)
    optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    best_val = -1.0

    for epoch in range(args.epochs):
        net_g.train()
        net_d.train()
        running_d = 0.0
        running_g = 0.0
        steps = 0

        for imgs, id_labels, exp_labels in train_loader:
            real_imgs = imgs.to(device)
            real_id_labels = id_labels.to(device)
            real_exp_labels = exp_labels.to(device)
            b_size = real_imgs.size(0)

            target_id_idx = torch.randint(0, n_id, (b_size,), device=device)
            c = torch.nn.functional.one_hot(target_id_idx, num_classes=n_id).float()

            fake_imgs, mu, logvar = net_g(real_imgs, c)
            net_d.zero_grad(set_to_none=True)
            err_d = compute_D_loss(net_d, real_imgs, fake_imgs, real_id_labels, real_exp_labels)
            err_d.backward()
            optimizer_d.step()

            for _ in range(max(1, args.g_updates)):
                net_g.zero_grad(set_to_none=True)
                fake_imgs, mu, logvar = net_g(real_imgs, c)
                err_g = compute_G_loss(net_d, fake_imgs, target_id_idx, real_exp_labels, mu, logvar)
                err_g.backward()
                optimizer_g.step()

            running_d += err_d.item()
            running_g += err_g.item()
            steps += 1

        val_expr_acc = evaluate_expr_acc(net_g, net_d, val_loader, device)
        avg_d = running_d / max(1, steps)
        avg_g = running_g / max(1, steps)
        print(
            f"[{epoch + 1}/{args.epochs}] "
            f"Loss_D={avg_d:.4f} Loss_G={avg_g:.4f} val_expr_acc={val_expr_acc:.4f}"
        )

        torch.save(net_g.state_dict(), os.path.join(args.save_dir, f"netG_epoch_{epoch + 1}.pth"))
        torch.save(net_d.state_dict(), os.path.join(args.save_dir, f"netD_epoch_{epoch + 1}.pth"))
        if val_expr_acc > best_val:
            best_val = val_expr_acc
            torch.save(net_g.state_dict(), os.path.join(args.save_dir, "netG_best.pth"))
            torch.save(net_d.state_dict(), os.path.join(args.save_dir, "netD_best.pth"))

        history.append(
            {
                "epoch": epoch + 1,
                "loss_d": avg_d,
                "loss_g": avg_g,
                "val_expr_acc": val_expr_acc,
            }
        )

    with open(os.path.join(args.save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_expr_acc": best_val,
                "epochs": args.epochs,
                "data_root": args.data_root,
                "data_format": args.data_format,
                "save_dir": args.save_dir,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Training complete. Best val expr acc = {best_val:.4f}")


if __name__ == "__main__":
    main()
