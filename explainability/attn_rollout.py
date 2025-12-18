from typing import Optional, Tuple
import io
import base64
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def _to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import cv2
    img = np.array(pil_img.convert("RGB"))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    blended = (img * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def compute_attention_rollout_base64(
    image_path: str,
    model_path: Optional[str] = None,
    hf_id: Optional[str] = "OpenGVLab/InternVL2-8B",
    device: Optional[str] = None,
    image_size: int = 448,
) -> Tuple[Optional[str], Optional[str]]:
    """
    InternVL2-8B 비전 인코더의 self-attention rollout을 활용한 히트맵 오버레이를 생성.
    반환: (base64 PNG data URL, error_message)
    """
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        path = model_path if (model_path and os.path.exists(model_path)) else (hf_id or "OpenGVLab/InternVL2-8B")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
        model.to(device)
        model.eval()

        # 이미지 로드/전처리
        pil = Image.open(image_path).convert("RGB")
        inputs = processor(images=pil, return_tensors="pt")
        pixel_values = inputs.get("pixel_values") or inputs.get("images") or None
        if pixel_values is None:
            return None, "processor가 pixel_values를 생성하지 못했습니다."
        pixel_values = pixel_values.to(device, dtype=dtype)

        # 비전 인코더 추출
        vision = None
        if hasattr(model, "vision_model"):
            vision = model.vision_model
        elif hasattr(model, "get_vision_tower"):
            try:
                vision = model.get_vision_tower()
            except Exception:
                vision = None
        elif hasattr(model, "intern_vit"):
            vision = model.intern_vit

        if vision is None:
            return None, "모델에서 비전 인코더를 찾을 수 없습니다."

        with torch.no_grad():
            out = vision(pixel_values=pixel_values, output_attentions=True)

        atts = out.attentions if hasattr(out, "attentions") else None
        if atts is None or len(atts) == 0:
            return None, "attention을 출력하지 못했습니다."

        # attention rollout: head 평균 → 잔차 고려 간단 누적곱
        attn_maps = [a.mean(dim=1) for a in atts]  # (B, heads, N, N) -> (B, N, N)
        attn = attn_maps[0]
        for i in range(1, len(attn_maps)):
            attn = torch.matmul(attn, attn_maps[i])
        # CLS 토큰에서 패치 토큰으로 가중치
        attn = attn[0]  # B=1
        cls_to_patch = attn[0, 1:] if attn.shape[0] > 1 else attn[0]  # (N-1,)
        # 패치 그리드 추정(정사각형 근사)
        num_patches = cls_to_patch.shape[0]
        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            side = int(np.ceil(np.sqrt(num_patches)))
            # 패딩하여 reshape 가능하게
            pad = side * side - num_patches
            cls_to_patch = torch.nn.functional.pad(cls_to_patch, (0, pad), value=cls_to_patch.min().item())
        heat = cls_to_patch[: side * side].reshape(side, side).float().cpu().numpy()

        overlay = _overlay_heatmap(pil, heat)
        return _to_base64(overlay), None
    except Exception as e:
        return None, f"히트맵 생성 실패: {e}"


