import argparse
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from config import nc, nz, n_id, image_size
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.VAE.Decoder import PPRL_VGAN_Decoder
from models.generator import PPRL_VGAN_Generator


def build_transform(image_size_: int) -> transforms.Compose:
    """
    使用与训练阶段一致的预处理:
    - Resize + CenterCrop 到 64x64
    - 转为 Tensor
    - 归一化到 [-1, 1]
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size_),
            transforms.CenterCrop(image_size_),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def tensor_to_bgr_image(t: torch.Tensor) -> np.ndarray:
    """
    将生成的张量图像 ([-1,1]) 转换为 OpenCV 可显示的 BGR uint8。
    期望输入形状: (3, H, W)
    """
    t = t.detach().cpu()
    t = (t + 1.0) / 2.0  # [-1,1] -> [0,1]
    t = torch.clamp(t, 0.0, 1.0)
    np_img = t.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))  # CHW -> HWC
    np_img = (np_img * 255).astype(np.uint8)
    # 转为 BGR 以便用 OpenCV 显示
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img


def prepare_generator(ckpt_path: str, device: torch.device) -> PPRL_VGAN_Generator:
    """
    构建并加载训练好的生成器权重。
    """
    encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
    decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
    netG = PPRL_VGAN_Generator(encoder, decoder)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.to(device)
    netG.eval()
    return netG


def build_identity_code(target_id_idx: int, device: torch.device) -> torch.Tensor:
    """
    构造目标虚拟身份的 one-hot 代码 c。
    """
    if target_id_idx < 0 or target_id_idx >= n_id:
        raise ValueError(f"target_id_idx must be in [0, {n_id - 1}], got {target_id_idx}")
    c = torch.nn.functional.one_hot(
        torch.tensor([target_id_idx], dtype=torch.long), num_classes=n_id
    ).float()
    return c.to(device)


def stack_with_info(
    original_bgr: np.ndarray,
    generated_bgr: np.ndarray,
    step_texts: Tuple[str, str, str, str],
) -> np.ndarray:
    """
    将原始摄像头图像和生成的虚拟人图像并排展示，并在底部叠加
    VGAN 隐私保护流程的文字说明。
    """
    h = min(original_bgr.shape[0], generated_bgr.shape[0])
    w = min(original_bgr.shape[1], generated_bgr.shape[1])
    original_resized = cv2.resize(original_bgr, (w, h))
    generated_resized = cv2.resize(generated_bgr, (w, h))

    combined = np.hstack([original_resized, generated_resized])

    # 在底部增加一条说明条
    info_bar_height = 80
    info_bar = np.zeros((info_bar_height, combined.shape[1], 3), dtype=np.uint8)

    # 填充深灰背景
    info_bar[:] = (30, 30, 30)

    # 写入四步流程说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (255, 255, 255)
    thickness = 1
    y0 = 22
    dy = 18
    x = 10
    for i, line in enumerate(step_texts):
        y = y0 + i * dy
        cv2.putText(info_bar, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    final = np.vstack([combined, info_bar])

    # 在上方添加简单标题
    title = "PPRL-VGAN: 實時表情遷移與隱私保護演示 (左: 原始人臉 / 右: 虛擬角色)"
    cv2.putText(
        final,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return final


def run_demo(
    checkpoint: str,
    target_id_idx: int,
    camera_index: int = 0,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    netG = prepare_generator(checkpoint, device)
    transform = build_transform(image_size)
    c = build_identity_code(target_id_idx, device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={camera_index})")

    print("Webcam demo started. Press 'q' to quit.")

    # 说明文本：对应 PPRL-VGAN 隐私保护的 4 个核心步骤
    step_texts = (
        "1. 輸入人臉: 從攝像頭讀取當前人臉表情。",
        "2. 去身份編碼: 編碼器提取與身份無關的表情潛在向量。",
        "3. 注入虛擬身份: 將潛在向量與虛擬角色身份代碼拼接。",
        "4. 生成虛擬人: 解碼器輸出僅保留表情的匿名虛擬人圖像。",
    )

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # 转为 RGB 并构建 PIL Image 以使用 torchvision 的变换
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # 预处理到与训练一致的 64x64
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                fake_imgs, _, _ = netG(img_tensor, c)

            # 取 batch 中的第一张
            fake_img = fake_imgs[0]
            fake_bgr = tensor_to_bgr_image(fake_img)

            # 拼接展示 + 流程说明
            display_img = stack_with_info(frame_bgr, fake_bgr, step_texts)

            cv2.imshow("PPRL-VGAN Privacy-Preserving Expression Transfer Demo", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPRL-VGAN webcam demo: expression transfer to virtual identity with privacy-preserving visualization."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained generator checkpoint (e.g., checkpoints/netG_epoch_199.pth).",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=0,
        help="Target virtual identity index (0-based, in [0, n_id-1]).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        checkpoint=args.checkpoint,
        target_id_idx=args.target_id,
        camera_index=args.camera_index,
    )

