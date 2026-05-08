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
criterion_Recon = nn.L1Loss()


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


def _to_unit_grayscale(x: torch.Tensor) -> torch.Tensor:
    # x shape: [B, 3, H, W], normalized to [-1, 1]
    x_unit = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    return 0.299 * x_unit[:, 0:1] + 0.587 * x_unit[:, 1:2] + 0.114 * x_unit[:, 2:3]


def _edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    gray = _to_unit_grayscale(x)
    device = gray.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _batch_quality_metrics(x: torch.Tensor) -> Tuple[float, float]:
    gray = _to_unit_grayscale(x)
    b = gray.shape[0]

    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray, lap_kernel, padding=1)

    sharpness = lap.view(b, -1).var(dim=1).mean().item()
    contrast = gray.view(b, -1).std(dim=1).mean().item()
    return float(sharpness), float(contrast)


def compute_D_loss(
    net_d: nn.Module,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    real_id_labels: torch.Tensor,
    real_exp_labels: torch.Tensor,
    id_loss_scale: float,
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
        + (lambda_D2 * id_loss_scale) * loss_d2
        + lambda_D3 * loss_d3
    )


def compute_G_loss(
    net_d: nn.Module,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    target_id_labels: torch.Tensor,
    original_exp_labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lambda_recon: float,
    lambda_edge: float,
    id_loss_scale: float,
) -> torch.Tensor:
    d1_fake, d2_fake, d3_fake = net_d(fake_imgs)
    label_real = torch.full_like(d1_fake, 1.0)
    loss_g1 = criterion_GAN(d1_fake, label_real)
    loss_g2 = criterion_Class(d2_fake, target_id_labels)
    loss_g3 = criterion_Class(d3_fake, original_exp_labels)
    loss_g4 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_g5 = criterion_Recon(fake_imgs, real_imgs)
    loss_g6 = criterion_Recon(_edge_magnitude(fake_imgs), _edge_magnitude(real_imgs))
    return (
        lambda_G1 * loss_g1
        + (lambda_G2 * id_loss_scale) * loss_g2
        + lambda_G3 * loss_g3
        + lambda_G4 * loss_g4
        + lambda_recon * loss_g5
        + lambda_edge * loss_g6
    )


def evaluate_metrics(
    net_g: nn.Module,
    net_d: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    net_g.eval()
    net_d.eval()
    expr_correct_fake = 0
    expr_correct_real = 0
    id_correct_fake = 0
    id_correct_real = 0
    sharpness_fake_sum = 0.0
    sharpness_real_sum = 0.0
    contrast_fake_sum = 0.0
    contrast_real_sum = 0.0
    batch_count = 0
    total = 0
    with torch.no_grad():
        for imgs, id_labels, exp_labels in loader:
            real_imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            exp_labels = exp_labels.to(device)
            b_size = real_imgs.size(0)
            target_id_idx = torch.randint(0, n_id, (b_size,), device=device)
            c = torch.nn.functional.one_hot(target_id_idx, num_classes=n_id).float()
            fake_imgs, _, _ = net_g(real_imgs, c)
            _, d2_real, d3_real = net_d(real_imgs)
            _, d2_fake, d3_fake = net_d(fake_imgs)

            pred_exp_real = torch.argmax(d3_real, dim=1)
            pred_exp_fake = torch.argmax(d3_fake, dim=1)
            pred_id_real = torch.argmax(d2_real, dim=1)
            pred_id_fake = torch.argmax(d2_fake, dim=1)

            s_fake, c_fake = _batch_quality_metrics(fake_imgs)
            s_real, c_real = _batch_quality_metrics(real_imgs)
            sharpness_fake_sum += s_fake
            contrast_fake_sum += c_fake
            sharpness_real_sum += s_real
            contrast_real_sum += c_real
            batch_count += 1

            expr_correct_real += (pred_exp_real == exp_labels).sum().item()
            expr_correct_fake += (pred_exp_fake == exp_labels).sum().item()
            id_correct_real += (pred_id_real == id_labels).sum().item()
            id_correct_fake += (pred_id_fake == target_id_idx).sum().item()
            total += exp_labels.size(0)

    denom = max(1, total)
    return {
        "expr_acc_fake": expr_correct_fake / denom,
        "expr_acc_real": expr_correct_real / denom,
        "id_acc_fake": id_correct_fake / denom,
        "id_acc_real": id_correct_real / denom,
        "sharpness_fake_mean": sharpness_fake_sum / max(1, batch_count),
        "sharpness_real_mean": sharpness_real_sum / max(1, batch_count),
        "contrast_fake_mean": contrast_fake_sum / max(1, batch_count),
        "contrast_real_mean": contrast_real_sum / max(1, batch_count),
    }


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


def save_training_curves(history: List[Dict[str, float]], save_dir: str) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip curve plotting because matplotlib is unavailable: {exc}")
        return

    epochs = [h["epoch"] for h in history]
    loss_d = [h["loss_d"] for h in history]
    loss_g = [h["loss_g"] for h in history]
    expr_fake = [h["val_expr_acc_fake"] for h in history]
    id_fake = [h["val_id_acc_fake"] for h in history]
    expr_real = [h["val_expr_acc_real"] for h in history]
    id_real = [h["val_id_acc_real"] for h in history]
    sharpness_ratio = [h["val_sharpness_ratio"] for h in history]
    contrast_ratio = [h["val_contrast_ratio"] for h in history]
    id_scale = [h["id_loss_scale"] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_d, label="loss_d")
    plt.plot(epochs, loss_g, label="loss_g")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EAVGAN Training Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, expr_fake, label="val_expr_acc_fake")
    plt.plot(epochs, id_fake, label="val_id_acc_fake")
    plt.plot(epochs, expr_real, label="val_expr_acc_real")
    plt.plot(epochs, id_real, label="val_id_acc_real")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EAVGAN Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, sharpness_ratio, label="val_sharpness_ratio")
    plt.plot(epochs, contrast_ratio, label="val_contrast_ratio")
    plt.plot(epochs, id_scale, label="id_loss_scale")
    plt.axhline(y=1.0, linestyle="--", linewidth=1.0, label="real_reference")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.title("Validation Quality Ratios (Fake / Real)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quality_curve.png"), dpi=200)
    plt.close()


def get_id_loss_scale(epoch_idx: int, stage1_epochs: int, warmup_epochs: int) -> float:
    """
    Two-stage identity schedule:
    - stage1: disable identity loss to prioritize readability and expression alignment.
    - stage2: gradually enable identity loss to recover anonymization behavior.
    """
    # epoch_idx is 0-based.
    if epoch_idx < stage1_epochs:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    progress = (epoch_idx - stage1_epochs + 1) / float(warmup_epochs)
    return float(max(0.0, min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EAVGAN from scratch on FER2013 real dataset.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="auto", choices=["auto", "csv", "folder"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-recon", type=float, default=8.0)
    parser.add_argument("--lambda-edge", type=float, default=4.0)
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=40,
        help="Epochs to disable identity losses (D2/G2) for readability-first training.",
    )
    parser.add_argument(
        "--stage2-id-warmup-epochs",
        type=int,
        default=40,
        help="Warmup epochs to ramp identity losses from 0 to 1 after stage1.",
    )
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
    best_val_expr_fake = -1.0
    best_selection_score = -1.0

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
            id_loss_scale = get_id_loss_scale(
                epoch_idx=epoch,
                stage1_epochs=args.stage1_epochs,
                warmup_epochs=args.stage2_id_warmup_epochs,
            )

            fake_imgs, mu, logvar = net_g(real_imgs, c)
            net_d.zero_grad(set_to_none=True)
            err_d = compute_D_loss(
                net_d,
                real_imgs,
                fake_imgs,
                real_id_labels,
                real_exp_labels,
                id_loss_scale=id_loss_scale,
            )
            err_d.backward()
            optimizer_d.step()

            for _ in range(max(1, args.g_updates)):
                net_g.zero_grad(set_to_none=True)
                fake_imgs, mu, logvar = net_g(real_imgs, c)
                err_g = compute_G_loss(
                    net_d=net_d,
                    real_imgs=real_imgs,
                    fake_imgs=fake_imgs,
                    target_id_labels=target_id_idx,
                    original_exp_labels=real_exp_labels,
                    mu=mu,
                    logvar=logvar,
                    lambda_recon=args.lambda_recon,
                    lambda_edge=args.lambda_edge,
                    id_loss_scale=id_loss_scale,
                )
                err_g.backward()
                optimizer_g.step()

            running_d += err_d.item()
            running_g += err_g.item()
            steps += 1

        val_metrics = evaluate_metrics(net_g, net_d, val_loader, device)
        avg_d = running_d / max(1, steps)
        avg_g = running_g / max(1, steps)
        sharp_ratio = val_metrics["sharpness_fake_mean"] / max(1e-8, val_metrics["sharpness_real_mean"])
        contrast_ratio = val_metrics["contrast_fake_mean"] / max(1e-8, val_metrics["contrast_real_mean"])
        selection_score = val_metrics["expr_acc_fake"] * min(1.0, sharp_ratio)
        id_loss_scale = get_id_loss_scale(
            epoch_idx=epoch,
            stage1_epochs=args.stage1_epochs,
            warmup_epochs=args.stage2_id_warmup_epochs,
        )
        print(
            f"[{epoch + 1}/{args.epochs}] "
            f"Loss_D={avg_d:.4f} Loss_G={avg_g:.4f} "
            f"val_expr_fake={val_metrics['expr_acc_fake']:.4f} "
            f"val_id_fake={val_metrics['id_acc_fake']:.4f} "
            f"sharp_ratio={sharp_ratio:.4f} "
            f"contrast_ratio={contrast_ratio:.4f} "
            f"id_scale={id_loss_scale:.2f}"
        )

        torch.save(net_g.state_dict(), os.path.join(args.save_dir, f"netG_epoch_{epoch + 1}.pth"))
        torch.save(net_d.state_dict(), os.path.join(args.save_dir, f"netD_epoch_{epoch + 1}.pth"))
        if val_metrics["expr_acc_fake"] > best_val_expr_fake:
            best_val_expr_fake = val_metrics["expr_acc_fake"]

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save(net_g.state_dict(), os.path.join(args.save_dir, "netG_best.pth"))
            torch.save(net_d.state_dict(), os.path.join(args.save_dir, "netD_best.pth"))

        history.append(
            {
                "epoch": epoch + 1,
                "loss_d": avg_d,
                "loss_g": avg_g,
                "val_expr_acc_fake": val_metrics["expr_acc_fake"],
                "val_expr_acc_real": val_metrics["expr_acc_real"],
                "val_id_acc_fake": val_metrics["id_acc_fake"],
                "val_id_acc_real": val_metrics["id_acc_real"],
                "val_sharpness_fake_mean": val_metrics["sharpness_fake_mean"],
                "val_sharpness_real_mean": val_metrics["sharpness_real_mean"],
                "val_contrast_fake_mean": val_metrics["contrast_fake_mean"],
                "val_contrast_real_mean": val_metrics["contrast_real_mean"],
                "val_sharpness_ratio": sharp_ratio,
                "val_contrast_ratio": contrast_ratio,
                "selection_score": selection_score,
                "id_loss_scale": id_loss_scale,
            }
        )

    with open(os.path.join(args.save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_expr_acc_fake": best_val_expr_fake,
                "best_selection_score": best_selection_score,
                "final_epoch_metrics": history[-1] if history else {},
                "epochs": args.epochs,
                "data_root": args.data_root,
                "data_format": args.data_format,
                "save_dir": args.save_dir,
                "device": str(device),
                "lambda_recon": args.lambda_recon,
                "lambda_edge": args.lambda_edge,
                "stage1_epochs": args.stage1_epochs,
                "stage2_id_warmup_epochs": args.stage2_id_warmup_epochs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    save_training_curves(history, args.save_dir)
    if history:
        final_m = history[-1]
        print(
            "Final metrics: "
            f"expr_fake={final_m['val_expr_acc_fake']:.4f}, "
            f"id_fake={final_m['val_id_acc_fake']:.4f}, "
            f"expr_real={final_m['val_expr_acc_real']:.4f}, "
            f"id_real={final_m['val_id_acc_real']:.4f}, "
            f"sharp_ratio={final_m['val_sharpness_ratio']:.4f}, "
            f"contrast_ratio={final_m['val_contrast_ratio']:.4f}"
        )
    print(f"Training complete. Best val expr(fake) acc = {best_val_expr_fake:.4f}")


if __name__ == "__main__":
    main()
