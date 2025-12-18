import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import yaml

from embeddings.extract import run_extract
from analytics.dim_reduce import reduce_umap_3d
from analytics.distances import min_distances_U_to_S, coverage_delta
from analytics.kcenter import kcenter_greedy
from viz.plotly_3d import build_figure, save_html


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_embeddings(outputs_dir: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    S_npz = np.load(os.path.join(outputs_dir, "embed_S.npz"))
    U_npz = np.load(os.path.join(outputs_dir, "embed_U.npz"))
    with open(os.path.join(outputs_dir, "paths_S.json"), "r") as f:
        paths_S = json.load(f)
    with open(os.path.join(outputs_dir, "paths_U.json"), "r") as f:
        paths_U = json.load(f)
    return S_npz["embeddings"], U_npz["embeddings"], paths_S, paths_U


def cmd_embed(args: argparse.Namespace) -> None:
    # S/U 각각 실행
    if args.split in ("S", "U"):
        run_extract(split=args.split, cfg_path=args.config)
    else:
        run_extract(split="S", cfg_path=args.config)
        run_extract(split="U", cfg_path=args.config)


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    data_cfg = cfg.get("data", {})
    umap_cfg = cfg.get("umap", {})
    dist_cfg = cfg.get("distance", {})
    k_cfg = cfg.get("kcenter", {})

    outputs_dir = data_cfg.get("outputs_dir", "outputs")
    ensure_dir(outputs_dir)

    # 로드
    S, U, paths_S, paths_U = load_embeddings(outputs_dir)

    # 거리/δ
    metric = args.metric or dist_cfg.get("metric", "cosine")
    d_U2S = min_distances_U_to_S(U, S, metric=metric)
    np.save(os.path.join(outputs_dir, "nn_U_to_S.npy"), d_U2S)
    delta = coverage_delta(d_U2S)
    with open(os.path.join(outputs_dir, "coverage.json"), "w") as f:
        json.dump({"delta": float(delta)}, f, ensure_ascii=False, indent=2)

    # UMAP 3D
    umap_params = dict(
        n_neighbors=int(umap_cfg.get("n_neighbors", 30)),
        min_dist=float(umap_cfg.get("min_dist", 0.1)),
        metric=umap_cfg.get("metric", "cosine"),
        use_pca_first=bool(umap_cfg.get("use_pca_first", False)),
        pca_dim=int(umap_cfg.get("pca_dim", 64)),
    )
    umap_S = reduce_umap_3d(S, **umap_params)
    umap_U = reduce_umap_3d(U, **umap_params)
    np.save(os.path.join(outputs_dir, "umap3d_S.npy"), umap_S)
    np.save(os.path.join(outputs_dir, "umap3d_U.npy"), umap_U)

    # K-Center
    k = int(args.k or k_cfg.get("num_select", 100))
    sel_idx = kcenter_greedy(U=U, S=S, k=k, metric=metric, precomputed_U_to_S=d_U2S)
    with open(os.path.join(outputs_dir, "kcenter_indices.json"), "w") as f:
        json.dump(sel_idx, f, ensure_ascii=False, indent=2)

    # Plotly HTML 저장(구형 모드 기본)
    fig = build_figure(umap_S=umap_S, umap_U=umap_U, dist_U_to_S=d_U2S, selected_U_indices=sel_idx, mode="spherical")
    save_html(fig, os.path.join(outputs_dir, "plot_3d.html"))

    # 경로 JSON이 없을 가능성도 있으므로 보장
    with open(os.path.join(outputs_dir, "paths_S.json"), "w") as f:
        json.dump(paths_S, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outputs_dir, "paths_U.json"), "w") as f:
        json.dump(paths_U, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--split", type=str, default="both", choices=["S", "U", "both"])
    p_embed.add_argument("--config", type=str, default="configs/config.yaml")

    p_ana = sub.add_parser("analyze")
    p_ana.add_argument("--config", type=str, default="configs/config.yaml")
    p_ana.add_argument("--k", type=int, default=None, help="K-Center 선택 개수")
    p_ana.add_argument("--metric", type=str, default=None, choices=["cosine", "euclidean"])

    args = parser.parse_args()
    if args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()


