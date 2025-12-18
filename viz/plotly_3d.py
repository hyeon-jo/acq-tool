from typing import List, Optional
import json
import numpy as np
import plotly.graph_objs as go
from viz.sphere_utils import l2_normalize_rows


def build_figure(
    umap_S: Optional[np.ndarray],
    umap_U: Optional[np.ndarray],
    dist_U_to_S: Optional[np.ndarray] = None,
    selected_U_indices: Optional[List[int]] = None,
    mode: str = "spherical",
) -> go.Figure:
    traces = []
    if umap_S is not None and umap_S.size > 0:
        pts = umap_S.copy()
        if mode == "spherical":
            pts = l2_normalize_rows(pts)
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name="S",
            marker=dict(size=2, color="rgba(0, 150, 255, 0.6)")
        ))
    if umap_U is not None and umap_U.size > 0:
        pts = umap_U.copy()
        if mode == "spherical":
            pts = l2_normalize_rows(pts)
        color_vals = None
        if dist_U_to_S is not None and len(dist_U_to_S) == pts.shape[0]:
            color_vals = dist_U_to_S
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name="U",
            marker=dict(
                size=2,
                color=color_vals if color_vals is not None else "rgba(255, 120, 0, 0.5)",
                colorscale="Turbo",
                opacity=0.9,
                colorbar=dict(title="dist(U→S)") if color_vals is not None else None
            )
        ))
        if selected_U_indices:
            sel = pts[np.array(selected_U_indices, dtype=int)]
            traces.append(go.Scatter3d(
                x=sel[:, 0], y=sel[:, 1], z=sel[:, 2],
                mode="markers",
                name="U* (K-Center)",
                marker=dict(size=5, color="red", symbol="diamond")
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")


