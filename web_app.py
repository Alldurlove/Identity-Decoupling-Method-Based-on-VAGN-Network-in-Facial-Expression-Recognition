import base64
import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from config import image_size, n_id, nc, nz
from models.VAE.Decoder import PPRL_VGAN_Decoder
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.generator import PPRL_VGAN_Generator


class InferenceRequest(BaseModel):
    image_base64: str = Field(..., description="data:image/...;base64,...")
    target_id: int = Field(0, ge=0, le=n_id - 1)


class InferenceEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.generator: Optional[PPRL_VGAN_Generator] = None
        self.model_ready = False
        self.model_status = "fallback"
        self._try_load_generator()

    def _try_load_generator(self) -> None:
        ckpt_path = os.environ.get("CHECKPOINT_PATH", "").strip()
        if not ckpt_path:
            self.model_status = "fallback: CHECKPOINT_PATH not set"
            return
        if not os.path.isfile(ckpt_path):
            self.model_status = f"fallback: checkpoint not found ({ckpt_path})"
            return
        try:
            encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
            decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
            net_g = PPRL_VGAN_Generator(encoder, decoder)
            state_dict = torch.load(ckpt_path, map_location=self.device)
            net_g.load_state_dict(state_dict)
            net_g.to(self.device)
            net_g.eval()
            self.generator = net_g
            self.model_ready = True
            self.model_status = f"ready: {ckpt_path}"
        except Exception as exc:
            self.model_status = f"fallback: load failed ({exc})"

    @staticmethod
    def _decode_data_uri(data_uri: str) -> np.ndarray:
        if "," not in data_uri:
            raise ValueError("image_base64 must be a data URI")
        _, b64data = data_uri.split(",", 1)
        image_bytes = base64.b64decode(b64data)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img)

    @staticmethod
    def _rgb_to_data_uri(rgb_img: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("failed to encode image")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _build_identity_code(self, target_id: int) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            torch.tensor([target_id], dtype=torch.long), num_classes=n_id
        ).float()
        return one_hot.to(self.device)

    def _run_model(self, rgb_img: np.ndarray, target_id: int) -> np.ndarray:
        if self.generator is None:
            raise RuntimeError("generator unavailable")
        pil_img = Image.fromarray(rgb_img)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        identity_code = self._build_identity_code(target_id)
        with torch.no_grad():
            fake_img, _, _ = self.generator(img_tensor, identity_code)
        out = fake_img[0].detach().cpu()
        out = (out + 1.0) / 2.0
        out = torch.clamp(out, 0.0, 1.0)
        out = (out.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return out

    @staticmethod
    def _run_fallback(rgb_img: np.ndarray) -> np.ndarray:
        # 无模型时使用轻量风格化，保证网页演示可运行。
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        stylized = cv2.stylization(bgr, sigma_s=60, sigma_r=0.5)
        return cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)

    def infer(self, data_uri: str, target_id: int) -> Tuple[str, str]:
        rgb = self._decode_data_uri(data_uri)
        if self.model_ready:
            out = self._run_model(rgb, target_id)
            mode = "model"
        else:
            out = self._run_fallback(rgb)
            mode = "fallback"
        return self._rgb_to_data_uri(out), mode


app = FastAPI(title="PPRL-VGAN Web Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = InferenceEngine()


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "device": str(engine.device),
            "model_ready": engine.model_ready,
            "model_status": engine.model_status,
        }
    )


@app.post("/api/infer")
def infer(payload: InferenceRequest) -> JSONResponse:
    try:
        image_data_uri, mode = engine.infer(payload.image_base64, payload.target_id)
        return JSONResponse({"ok": True, "mode": mode, "image_base64": image_data_uri})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc


app.mount("/", StaticFiles(directory="web", html=True), name="web")
