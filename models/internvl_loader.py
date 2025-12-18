import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


def _crop_sky_if_needed(img: Image.Image, sky_crop_top_pct: float) -> Image.Image:
    if sky_crop_top_pct <= 0.0:
        return img
    w, h = img.size
    crop_top = int(h * sky_crop_top_pct)
    crop_top = max(0, min(crop_top, h - 1))
    return img.crop((0, crop_top, w, h))


class InternVLVisionEncoder:
    """
    InternVL2 비전 인코더를 이용해 이미지 임베딩을 추출한다.
    - 입력: PIL 이미지 또는 파일 경로들
    - 출력: (num_images, hidden_dim) 임베딩(numpy)
    - 풀링: CLS 제외한 토큰 평균 풀링(기본)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_id: Optional[str] = "OpenGVLab/InternVL2-8B",
        precision: str = "bf16",
        image_size: int = 448,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.image_size = image_size

        if precision == "bf16" and torch.cuda.is_available():
            self.dtype = torch.bfloat16
        elif precision == "fp16" and torch.cuda.is_available():
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        path = model_path if (model_path and os.path.exists(model_path)) else (hf_id or "OpenGVLab/InternVL2-8B")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, torch_dtype=self.dtype, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        # 비전 인코더 핸들
        self.vision = None
        if hasattr(self.model, "vision_model"):
            self.vision = self.model.vision_model
        elif hasattr(self.model, "get_vision_tower"):
            try:
                self.vision = self.model.get_vision_tower()
            except Exception:
                self.vision = None
        elif hasattr(self.model, "intern_vit"):
            self.vision = self.model.intern_vit
        if self.vision is None:
            raise RuntimeError("InternVL2 vision encoder를 찾을 수 없습니다.")

    @torch.no_grad()
    def _encode_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values") or inputs.get("images") or None
        if pixel_values is None:
            raise RuntimeError("Processor가 pixel_values를 생성하지 못했습니다.")
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        # vision forward
        out = self.vision(pixel_values=pixel_values)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            # 일부 모델은 다른 키를 사용할 수 있음
            hidden = getattr(out, "last_hidden_states", None)
        if hidden is None:
            raise RuntimeError("vision forward 결과에서 last_hidden_state를 찾을 수 없습니다.")
        # shape: (B, seq_len, hidden_dim)
        # CLS(토큰 0) 제외 평균 풀링
        if hidden.size(1) > 1:
            pooled = hidden[:, 1:, :].mean(dim=1)
        else:
            pooled = hidden.mean(dim=1)
        return pooled  # (B, hidden_dim)

    def encode_filepaths(
        self,
        image_paths: List[str],
        batch_size: int = 8,
        sky_crop_top_pct: float = 0.0,
        normalize: bool = True,
    ) -> np.ndarray:
        embs: List[torch.Tensor] = []
        batch: List[Image.Image] = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = _crop_sky_if_needed(img, sky_crop_top_pct)
                batch.append(img)
                if len(batch) >= batch_size:
                    embs.append(self._encode_batch(batch))
                    batch.clear()
            except Exception:
                # 깨진 이미지 등은 건너뜀
                continue
        if batch:
            embs.append(self._encode_batch(batch))
        if not embs:
            return np.zeros((0, 0), dtype=np.float32)
        embs_t = torch.cat(embs, dim=0)  # (N, D)
        if normalize and embs_t.numel() > 0:
            embs_t = torch.nn.functional.normalize(embs_t, p=2, dim=1)
        return embs_t.float().cpu().numpy()


