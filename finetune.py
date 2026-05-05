import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

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
from data.ferg_dataset import FERGMultiLabelDataset
from data.real_expr_data import (
    MixedExpressionDataset,
    RealExpressionDataset,
    discover_expr_samples,
)
from models.VAE.Decoder import PPRL_VGAN_Decoder
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.discriminator import PPRL_VGAN_Discriminator
from models.generator import PPRL_VGAN_Generator


criterion_GAN = nn.BCELoss()
criterion_Class = nn.CrossEntropyLoss()


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

    loss_d = (
        lambda_D1 * (loss_d1_real + loss_d1_fake)
        + lambda_D2 * loss_d2
        + lambda_D3 * loss_d3
    )
    return loss_d


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

    loss_g = (
        lambda_G1 * loss_g1
        + lambda_G2 * loss_g2
        + lambda_G3 * loss_g3
        + lambda_G4 * loss_g4
    )
    return loss_g


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_models(
    resume_netg: Optional[str],
    resume_netd: Optional[str],
    device: torch.device,
) -> Tuple[PPRL_VGAN_Generator, PPRL_VGAN_Discriminator]:
    encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
    decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
    net_g = PPRL_VGAN_Generator(encoder, decoder)
    net_d = PPRL_VGAN_Discriminator()

    if resume_netg:
        state = torch.load(resume_netg, map_location=device)
        net_g.load_state_dict(state)
    if resume_netd:
        state = torch.load(resume_netd, map_location=device)
        net_d.load_state_dict(state)

    net_g.to(device)
    net_d.to(device)
    return net_g, net_d


def freeze_encoder_front_half(net_g: PPRL_VGAN_Generator) -> None:
    for name, param in net_g.named_parameters():
        if "encoder.conv_layers." in name:
            try:
                idx = int(name.split("encoder.conv_layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                idx = -1
            if idx in (0, 1, 2, 3, 4, 5):
                param.requires_grad = False
                continue
        param.requires_grad = True


def unfreeze_all(net_g: PPRL_VGAN_Generator, net_d: PPRL_VGAN_Discriminator) -> None:
    for p in net_g.parameters():
        p.requires_grad = True
    for p in net_d.parameters():
        p.requires_grad = True


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    transform = build_transform()

    real_train_samples = discover_expr_samples(
        root=args.real_data_root,
        split="train",
        default_id=args.default_real_id,
        parse_id_from_filename=args.parse_real_id_from_filename,
    )
    real_val_samples = discover_expr_samples(
        root=args.real_data_root,
        split="val",
        default_id=args.default_real_id,
        parse_id_from_filename=args.parse_real_id_from_filename,
    )

    real_train_ds = RealExpressionDataset(real_train_samples, transform=transform)
    real_val_ds = RealExpressionDataset(real_val_samples, transform=transform)

    if args.ferg_dataroot:
        ferg_train_ds = FERGMultiLabelDataset(args.ferg_dataroot, transform=transform)
        train_ds = MixedExpressionDataset(
            primary_dataset=real_train_ds,
            secondary_dataset=ferg_train_ds,
            primary_weight=args.real_mix_ratio,
            base_len=max(len(real_train_ds), len(ferg_train_ds)),
            seed=args.seed,
        )
    else:
        train_ds = real_train_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        real_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate_expression_consistency(
    net_g: PPRL_VGAN_Generator,
    net_d: PPRL_VGAN_Discriminator,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    net_g.eval()
    net_d.eval()
    total = 0
    correct = 0
    total_g_loss = 0.0

    with torch.no_grad():
        for imgs, _, exp_labels in loader:
            real_imgs = imgs.to(device)
            exp_labels = exp_labels.to(device)
            b_size = real_imgs.size(0)

            target_id_idx = torch.randint(0, n_id, (b_size,), device=device)
            c = torch.nn.functional.one_hot(target_id_idx, num_classes=n_id).float()
            fake_imgs, mu, logvar = net_g(real_imgs, c)
            _, _, d3_fake = net_d(fake_imgs)

            preds = torch.argmax(d3_fake, dim=1)
            correct += (preds == exp_labels).sum().item()
            total += exp_labels.size(0)

            total_g_loss += compute_G_loss(
                net_d=net_d,
                fake_imgs=fake_imgs,
                target_id_labels=target_id_idx,
                original_exp_labels=exp_labels,
                mu=mu,
                logvar=logvar,
            ).item() * b_size

    return {
        "expr_acc": correct / max(1, total),
        "avg_g_loss": total_g_loss / max(1, total),
    }


def run_stage(
    stage_name: str,
    num_epochs: int,
    net_g: PPRL_VGAN_Generator,
    net_d: PPRL_VGAN_Discriminator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    save_dir: str,
    best_metric: float,
    patience: int,
    history: List[Dict[str, float]],
) -> Tuple[float, int]:
    no_improve = 0
    for epoch in range(num_epochs):
        net_g.train()
        net_d.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
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
            loss_d = compute_D_loss(
                net_d=net_d,
                real_imgs=real_imgs,
                fake_imgs=fake_imgs,
                real_id_labels=real_id_labels,
                real_exp_labels=real_exp_labels,
            )
            loss_d.backward()
            optimizer_d.step()

            net_g.zero_grad(set_to_none=True)
            fake_imgs, mu, logvar = net_g(real_imgs, c)
            loss_g = compute_G_loss(
                net_d=net_d,
                fake_imgs=fake_imgs,
                target_id_labels=target_id_idx,
                original_exp_labels=real_exp_labels,
                mu=mu,
                logvar=logvar,
            )
            loss_g.backward()
            optimizer_g.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()
            steps += 1

        val_metrics = evaluate_expression_consistency(net_g, net_d, val_loader, device)
        epoch_stat = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "train_loss_d": running_loss_d / max(1, steps),
            "train_loss_g": running_loss_g / max(1, steps),
            "val_expr_acc": val_metrics["expr_acc"],
            "val_avg_g_loss": val_metrics["avg_g_loss"],
        }
        history.append(epoch_stat)
        print(
            f"[{stage_name}][{epoch + 1}/{num_epochs}] "
            f"D={epoch_stat['train_loss_d']:.4f} "
            f"G={epoch_stat['train_loss_g']:.4f} "
            f"val_expr_acc={epoch_stat['val_expr_acc']:.4f}"
        )

        torch.save(net_g.state_dict(), os.path.join(save_dir, f"netG_{stage_name}_epoch_{epoch + 1}.pth"))
        torch.save(net_d.state_dict(), os.path.join(save_dir, f"netD_{stage_name}_epoch_{epoch + 1}.pth"))

        if val_metrics["expr_acc"] > best_metric:
            best_metric = val_metrics["expr_acc"]
            no_improve = 0
            torch.save(net_g.state_dict(), os.path.join(save_dir, "netG_finetuned_best.pth"))
            torch.save(net_d.state_dict(), os.path.join(save_dir, "netD_finetuned_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop in {stage_name}: no improvement for {patience} epochs.")
                break

    return best_metric, no_improve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune VGAN on real expression dataset.")
    parser.add_argument("--real-data-root", type=str, required=True)
    parser.add_argument("--ferg-dataroot", type=str, default="")
    parser.add_argument("--real-mix-ratio", type=float, default=0.7)
    parser.add_argument("--default-real-id", type=int, default=0)
    parser.add_argument("--parse-real-id-from-filename", action="store_true")
    parser.add_argument("--resume-netg", type=str, required=True)
    parser.add_argument("--resume-netd", type=str, default="")
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-a", type=float, default=1e-4)
    parser.add_argument("--lr-b", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints/finetune")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(args)
    net_g, net_d = build_models(args.resume_netg, args.resume_netd or None, device)

    history: List[Dict[str, float]] = []
    best_metric = -1.0

    freeze_encoder_front_half(net_g)
    for p in net_d.parameters():
        p.requires_grad = True
    optimizer_g = optim.RMSprop((p for p in net_g.parameters() if p.requires_grad), lr=args.lr_a)
    optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.lr_a)
    best_metric, _ = run_stage(
        stage_name="stageA",
        num_epochs=args.stage_a_epochs,
        net_g=net_g,
        net_d=net_d,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        save_dir=args.save_dir,
        best_metric=best_metric,
        patience=args.patience,
        history=history,
    )

    unfreeze_all(net_g, net_d)
    optimizer_g = optim.RMSprop(net_g.parameters(), lr=args.lr_b)
    optimizer_d = optim.RMSprop(net_d.parameters(), lr=args.lr_b)
    best_metric, _ = run_stage(
        stage_name="stageB",
        num_epochs=args.stage_b_epochs,
        net_g=net_g,
        net_d=net_d,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        save_dir=args.save_dir,
        best_metric=best_metric,
        patience=args.patience,
        history=history,
    )

    with open(os.path.join(args.save_dir, "finetune_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_dir, "finetune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "best_val_expr_acc": best_metric,
                "real_data_root": args.real_data_root,
                "ferg_dataroot": args.ferg_dataroot,
                "real_mix_ratio": args.real_mix_ratio,
                "resume_netg": args.resume_netg,
                "resume_netd": args.resume_netd,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Finetuning complete. Best val expr acc = {best_metric:.4f}")


if __name__ == "__main__":
    main()
