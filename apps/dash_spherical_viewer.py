import argparse
import base64
import io
import json
import os
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors

from viz.sphere_utils import l2_normalize_rows, generate_sphere_grid, maybe_sample_points
from explainability.attn_rollout import compute_attention_rollout_base64


def _load_numpy(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # take 'arr_0' by default
        if "embeddings" in arr.files:
            return arr["embeddings"]
        return arr[arr.files[0]]
    return arr


def _load_data(data_root: str) -> Dict[str, np.ndarray]:
    paths = {
        "umap_S": os.path.join(data_root, "umap3d_S.npy"),
        "umap_U": os.path.join(data_root, "umap3d_U.npy"),
        "dist_U_to_S": os.path.join(data_root, "nn_U_to_S.npy"),
        "coverage": os.path.join(data_root, "coverage.json"),
        "paths_S": os.path.join(data_root, "paths_S.json"),
        "paths_U": os.path.join(data_root, "paths_U.json"),
    }
    data = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            continue
        if p.endswith(".npy"):
            data[k] = np.load(p)
        elif p.endswith(".json"):
            with open(p, "r") as f:
                data[k] = json.load(f)
    return data


def _build_base_figure(points_S: np.ndarray,
                       points_U: np.ndarray,
                       dist_U_to_S: np.ndarray,
                       mode: str,
                       point_limit: int) -> go.Figure:
    # 투영: spherical이면 L2 normalize
    if points_S is not None and points_S.size > 0:
        pts_S = points_S.copy()
        if mode == "spherical":
            pts_S = l2_normalize_rows(pts_S)
        pts_S = maybe_sample_points(pts_S, point_limit)
    else:
        pts_S = None

    if points_U is not None and points_U.size > 0:
        pts_U = points_U.copy()
        if mode == "spherical":
            pts_U = l2_normalize_rows(pts_U)
        # 거리 컬러링을 위해 동일한 샘플링 인덱스 유지가 최선이나,
        # 간단히 좌표만 샘플링하고 컬러는 잘리도록 처리
        if pts_U.shape[0] > point_limit:
            idx = np.linspace(0, pts_U.shape[0] - 1, num=point_limit, dtype=int)
            pts_U = pts_U[idx]
            if dist_U_to_S is not None and len(dist_U_to_S) > 0:
                dist_U_to_S = dist_U_to_S[idx]
    else:
        pts_U = None

    traces = []
    if pts_S is not None:
        traces.append(go.Scatter3d(
            x=pts_S[:, 0], y=pts_S[:, 1], z=pts_S[:, 2],
            mode="markers",
            name="S",
            marker=dict(size=2, color="rgba(0, 150, 255, 0.6)")
        ))
    if pts_U is not None:
        color_vals = None
        if dist_U_to_S is not None and len(dist_U_to_S) == pts_U.shape[0]:
            color_vals = dist_U_to_S
        traces.append(go.Scatter3d(
            x=pts_U[:, 0], y=pts_U[:, 1], z=pts_U[:, 2],
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


def layout_app(app: dash.Dash) -> dash.Dash:
    controls = dbc.Card([
        html.Div("Controls", className="h6 mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("표시 데이터"),
                dcc.Dropdown(
                    id="display-split",
                    options=[{"label": v, "value": v} for v in ["Both", "S", "U"]],
                    value="Both",
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                dbc.Label("투영"),
                dcc.Dropdown(
                    id="projection-mode",
                    options=[{"label": v, "value": v} for v in ["spherical", "cartesian"]],
                    value="spherical",
                    clearable=False
                )
            ], width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("점 최대 수"),
                dcc.Input(id="point-limit", type="number", value=50000, min=1000, step=1000)
            ], width=6),
            dbc.Col([
                dbc.Label("top-K"),
                dcc.Input(id="top-k", type="number", value=8, min=1, step=1)
            ], width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("δ 임계값"),
                dcc.Input(id="delta-threshold", type="number", value=None)
            ], width=6),
            dbc.Col([
                dbc.Label("검색 대상"),
                dcc.Dropdown(
                    id="search-space",
                    options=[{"label": v, "value": v} for v in ["U", "S", "Both"]],
                    value="U",
                    clearable=False
                )
            ], width=6),
        ]),
        html.Hr(className="my-2"),
        dbc.Button("내보내기(선택 결과)", id="export-btn", color="primary", disabled=True),
        dcc.Download(id="export-download"),
        dcc.Store(id="memory-selection"),
    ], body=True)

    graph_card = dbc.Card([
        dcc.Loading(
            dcc.Graph(id="sphere-graph", style={"height": "75vh"}),
            type="default"
        )
    ], body=True)

    right_panel = dbc.Card([
        html.Div("결과", className="h6 mb-2"),
        html.Div(id="result-summary", className="mb-2"),
        html.Div(id="thumb-grid", className="d-flex flex-wrap gap-2"),
        html.Hr(className="my-2"),
        dbc.Button("히트맵 생성(선택 1개)", id="gen-heatmap-btn", color="secondary", disabled=False, className="mb-2"),
        html.Div(id="heatmap-view"),
    ], body=True)

    app.layout = dbc.Container([
        dcc.Store(id="store-data-root"),
        dcc.Store(id="store-umap-S"),
        dcc.Store(id="store-umap-U"),
        dcc.Store(id="store-dist-U2S"),
        dcc.Store(id="store-paths-S"),
        dcc.Store(id="store-paths-U"),
        dcc.Store(id="store-coverage"),
        dbc.Row([
            dbc.Col(controls, width=3),
            dbc.Col(graph_card, width=6),
            dbc.Col(right_panel, width=3),
        ], className="mt-2 gx-2"),
    ], fluid=True)
    return app


def encode_image_as_base64(path: str, max_side: int = 256) -> str:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # 실패 시 빈 투명 이미지
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--data-root", type=str, default="outputs")
    args = parser.parse_args()

    data = _load_data(args.data_root)
    umap_S = data.get("umap_S", None)
    umap_U = data.get("umap_U", None)
    dist_U2S = data.get("dist_U_to_S", None)
    coverage = data.get("coverage", {"delta": None})
    paths_S = data.get("paths_S", [])
    paths_U = data.get("paths_U", [])

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app = layout_app(app)

    # 데이터 저장
    @app.callback(
        Output("store-data-root", "data"),
        Output("store-umap-S", "data"),
        Output("store-umap-U", "data"),
        Output("store-dist-U2S", "data"),
        Output("store-paths-S", "data"),
        Output("store-paths-U", "data"),
        Output("store-coverage", "data"),
        Input("sphere-graph", "id"),
        prevent_initial_call=False
    )
    def init_store(_):
        return (args.data_root,
                umap_S.tolist() if umap_S is not None else [],
                umap_U.tolist() if umap_U is not None else [],
                dist_U2S.tolist() if dist_U2S is not None else [],
                paths_S, paths_U, coverage)

    @app.callback(
        Output("sphere-graph", "figure"),
        Input("store-umap-S", "data"),
        Input("store-umap-U", "data"),
        Input("store-dist-U2S", "data"),
        Input("projection-mode", "value"),
        Input("display-split", "value"),
        Input("point-limit", "value"),
    )
    def update_graph(umap_S_data, umap_U_data, dist_data, mode, display_split, point_limit):
        umap_S_arr = np.array(umap_S_data) if umap_S_data else np.empty((0, 3))
        umap_U_arr = np.array(umap_U_data) if umap_U_data else np.empty((0, 3))
        dist_arr = np.array(dist_data) if dist_data else None

        if display_split == "S":
            umap_U_arr = np.empty((0, 3))
            dist_arr = None
        elif display_split == "U":
            umap_S_arr = np.empty((0, 3))

        fig = _build_base_figure(umap_S_arr, umap_U_arr, dist_arr, mode, int(point_limit))
        # 참조용 구 메시(단순 원점에서 반투명 구): 시각적 가이드
        # Plotly에 직접 구 메시를 추가하려면 mesh3d를 생성해야 하나, 간단히 생략 가능.
        return fig

    @app.callback(
        Output("thumb-grid", "children"),
        Output("result-summary", "children"),
        Output("memory-selection", "data"),
        Output("export-btn", "disabled"),
        Input("sphere-graph", "clickData"),
        State("store-umap-S", "data"),
        State("store-umap-U", "data"),
        State("store-dist-U2S", "data"),
        State("store-paths-S", "data"),
        State("store-paths-U", "data"),
        State("store-coverage", "data"),
        State("projection-mode", "value"),
        State("search-space", "value"),
        State("top-k", "value"),
        State("delta-threshold", "value"),
        prevent_initial_call=True
    )
    def on_click(click_data, umap_S_data, umap_U_data, dist_data, paths_S, paths_U,
                 coverage, mode, search_space, top_k, delta_threshold):
        # 기본 파라미터
        top_k = int(top_k or 8)
        delta = coverage.get("delta", None)
        delta_thr = float(delta_threshold) if delta_threshold is not None else (float(delta) if delta else None)

        # 좌표 선택
        if not click_data or "points" not in click_data:
            return [], "클릭한 점이 없습니다.", None, True
        p = click_data["points"][0]
        q = np.array([[p["x"], p["y"], p["z"]]], dtype=np.float32)
        if mode == "spherical":
            q = l2_normalize_rows(q)

        # 검색 집합 구성
        umap_S_arr = np.array(umap_S_data) if umap_S_data else np.empty((0, 3))
        umap_U_arr = np.array(umap_U_data) if umap_U_data else np.empty((0, 3))
        dist_arr = np.array(dist_data) if dist_data else None

        candidates = []
        cand_paths = []
        # U 우선
        if search_space in ("U", "Both") and umap_U_arr.size > 0:
            candidates.append(umap_U_arr)
            cand_paths.append(paths_U)
        if search_space in ("S", "Both") and umap_S_arr.size > 0:
            candidates.append(umap_S_arr)
            cand_paths.append(paths_S)
        if not candidates:
            return [], "검색 대상이 비어 있습니다.", None, True

        pts = np.concatenate(candidates, axis=0)
        path_list = sum(cand_paths, [])

        # 최근접 탐색(3D 좌표 기준)
        nn = NearestNeighbors(n_neighbors=min(top_k, len(path_list)), metric="euclidean")
        nn.fit(pts)
        dist, idx = nn.kneighbors(q, return_distance=True)
        idx = idx[0].tolist()

        thumbs = []
        gap_flags = []
        for i in idx:
            path = path_list[i]
            img64 = encode_image_as_base64(path)
            # U의 경우에만 dist(U→S) 활용
            dist_to_S = None
            if i < len(paths_U) and dist_arr is not None and len(dist_arr) == len(paths_U):
                dist_to_S = float(dist_arr[i])
            gap = False
            if delta_thr is not None and dist_to_S is not None:
                gap = dist_to_S >= delta_thr
            gap_flags.append(gap)
            thumbs.append(
                html.Div([
                    html.Img(src=img64, style={"maxWidth": "120px", "maxHeight": "120px", "borderRadius": "4px"}),
                    html.Div(children=os.path.basename(path), style={"fontSize": "11px", "maxWidth": "120px", "overflow": "hidden", "textOverflow": "ellipsis"}),
                    html.Div(children=("dist_to_S: {:.3f}".format(dist_to_S) if dist_to_S is not None else ""), style={"fontSize": "11px"}),
                    html.Div(children=("GAP" if gap else ""), style={"color": "#d33" if gap else "#999", "fontWeight": "bold", "fontSize": "12px"}),
                ], style={"display": "inline-block", "marginRight": "8px"})
            )

        gap_count = sum(1 for g in gap_flags if g)
        summary = f"선택 지점 주변 top-{top_k}: GAP 후보 {gap_count}개" + (f" (δ={delta_thr:.3f})" if delta_thr is not None else "")

        selection_payload = {
            "indices": idx,
            "paths": [path_list[i] for i in idx],
        }
        return thumbs, summary, selection_payload, False

    @app.callback(
        Output("export-download", "data"),
        Input("export-btn", "n_clicks"),
        State("memory-selection", "data"),
        prevent_initial_call=True
    )
    def on_export(n, sel):
        if not sel or not sel.get("paths"):
            return dash.no_update
        # 간단한 HTML 리포트 생성(썸네일 + 가이드 문구)
        items_html = []
        for p in sel["paths"]:
            img64 = encode_image_as_base64(p, max_side=384)
            items_html.append(f"""
            <div style="display:inline-block;margin:6px;text-align:center;">
              <img src="{img64}" style="max-width:200px;border-radius:6px;border:1px solid #ddd"/>
              <div style="font-size:12px;color:#333;">{os.path.basename(p)}</div>
            </div>
            """)
        body = f"""
        <html><head><meta charset="utf-8"><title>Visual Spec</title></head>
        <body style="font-family:Arial,Helvetica,sans-serif;">
          <h2>Visual Collection Spec</h2>
          <p>가이드: 아래 예시 이미지들의 <b>붉은색 강조 영역(모델 주목)</b>과 유사한 시각적 패턴이 나타나는 환경을 수집하십시오. 
          텍스트로 정의하지 말고 눈에 보이는 질감을 따르십시오.</p>
          <div>{''.join(items_html)}</div>
        </body></html>
        """.encode("utf-8")
        return dict(content=body, filename="visual_spec.html", type="text/html")

    @app.callback(
        Output("heatmap-view", "children"),
        Input("gen-heatmap-btn", "n_clicks"),
        State("memory-selection", "data"),
        prevent_initial_call=True
    )
    def on_heatmap(n, sel):
        if not sel or not sel.get("paths"):
            return html.Div("선택된 항목이 없습니다.")
        path = sel["paths"][0]
        model_path = os.environ.get("INTERNVL2_PATH", "")
        data_url, err = compute_attention_rollout_base64(
            image_path=path,
            model_path=model_path if model_path else None,
            hf_id="OpenGVLab/InternVL2-8B"
        )
        if err:
            return html.Div(f"히트맵 오류: {err}", style={"color": "#c00"})
        if not data_url:
            return html.Div("히트맵을 생성하지 못했습니다.")
        return html.Img(src=data_url, style={"maxWidth": "100%"})

    app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()


