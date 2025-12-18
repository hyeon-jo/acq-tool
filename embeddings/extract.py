import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import yaml

from models.internvl_loader import InternVLVisionEncoder


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_extract(split: str, cfg_path: str = "configs/config.yaml") -> Tuple[np.ndarray, list]:
    cfg = load_cfg(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    pre_cfg = cfg.get("preprocess", {})

    if split.upper() == "S":
        img_dir = data_cfg.get("S_dir")
    elif split.upper() == "U":
        img_dir = data_cfg.get("U_dir")
    else:
        raise ValueError("split must be 'S' or 'U'")

    outputs_dir = data_cfg.get("outputs_dir", "outputs")
    ensure_dir(outputs_dir)

    paths = list_images(img_dir)
    if not paths:
        raise RuntimeError(f"No images found in {img_dir}")

    enc = InternVLVisionEncoder(
        model_path=model_cfg.get("path") or os.environ.get("INTERNVL2_PATH", ""),
        hf_id=model_cfg.get("hf_id", "OpenGVLab/InternVL2-8B"),
        precision=model_cfg.get("precision", "bf16"),
        image_size=int(model_cfg.get("image_size", 448)),
    )

    embs = enc.encode_filepaths(
        image_paths=paths,
        batch_size=int(infer_cfg.get("batch_size", 8)),
        sky_crop_top_pct=float(pre_cfg.get("sky_crop_top_pct", 0.0)),
        normalize=bool(infer_cfg.get("normalize_embeddings", True)),
    )

    np.savez_compressed(os.path.join(outputs_dir, f"embed_{split.upper()}.npz"), embeddings=embs)
    with open(os.path.join(outputs_dir, f"paths_{split.upper()}.json"), "w") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    return embs, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["S", "U"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_extract(split=args.split, cfg_path=args.config)


if __name__ == "__main__":
    main()


