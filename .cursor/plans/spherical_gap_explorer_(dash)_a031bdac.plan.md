---
name: Spherical Gap Explorer (Dash)
overview: UMAP 3D를 단위 구로 투영해 회전·클릭 인터랙션을 제공하고, 클릭 지점 근방의 최근접 샘플과 수집 인사이트(프로토타입/히트맵/δ 기준)를 즉시 제시하는 Dash 앱을 추가합니다.
todos:
  - id: sphere-projection
    content: UMAP 3D→단위 구 투영 및 가상 클릭 격자 생성
    status: completed
  - id: dash-shell
    content: Dash 레이아웃/컨트롤/3D 그래프 베이스 구현
    status: completed
  - id: click-neighbors
    content: 클릭 좌표→최근접 top-K 질의와 δ 판정 로직
    status: completed
  - id: thumbs-panel
    content: 썸네일/메타 표시 및 시각적 강조 UI
    status: completed
  - id: attention-overlay
    content: InternViT attention rollout 히트맵 오버레이
    status: completed
  - id: export-spec
    content: 선택 샘플 기반 Visual Spec 내보내기(HTML/PNG ZIP)
    status: completed
  - id: perf-sampling
    content: 대규모 포인트 샘플링/FAISS 옵션/캐싱
    status: completed
---

# 대화형 구형 Gap Explorer 확장 플랜(Dash)

## 목표

- 단위 구 형태 시각화 + 회전/줌
- 빈 공간(희소 영역) 클릭 → 주변 최근접 이미지(이미지 전용) 추천 + δ 기반 Gap 여부 안내
- 선택 샘플로 즉시 시각적 가이드(Attention Rollout/Grad-CAM 유사 히트맵) 생성 및 다운로드

## 신규/변경 파일 구조

- `apps/dash_spherical_viewer.py`: Dash 앱 엔트리(서버/레이아웃/콜백)
- `viz/sphere_utils.py`: 3D→단위구 투영, 균일 샘플 그리드(가상 클릭 포인트) 생성
- `insights/neighbor_query.py`: 클릭 좌표→최근접 이웃 질의(top-K), δ 비교/라벨링
- `explainability/attn_rollout.py`: InternViT self-attention rollout 기반 히트맵(가능 시 Grad-CAM 대체/추가)
- `assets/`(Dash 정적): 기본 스타일
- 기존 파이프라인 산출물 재사용:
- 임베딩: `outputs/embed_S.npz`, `outputs/embed_U.npz`
- 좌표(UMAP 3D): `outputs/umap3d_S.npy`, `outputs/umap3d_U.npy`
- δ/거리: `outputs/coverage.json`, `outputs/nn_U_to_S.npy`

## UI 상호작용 사양(Dash)

- 상단 컨트롤 패널
- 데이터 표시: `S`, `U`, `Both` 토글
- 색상 맵: `최근접거리(→S)` or `클러스터 밀도`
- δ 임계 슬라이더: gap 판정 기준(기본 δcover)
- 최근접 탐색 파라미터: `topK`, `metric(cosine/euclid)`, `searchSpace(U/S/Both)`
- 투영 모드: `spherical`/`cartesian` 토글
- 3D 그래프(Plotly)
- 점: S/U 샘플(UMAP3D→단위구 정규화)
- 반투명 구 메시(참조)
- 가상 구 격자 포인트(희미한 점): 빈 공간 클릭을 위한 클릭 타깃
- 클릭 이벤트: (x,y,z) 좌표 확보
- 우측 패널: 결과/가이드
- 최근접 이미지 썸네일 그리드(top-K)
- 각 샘플의 `dist_to_S`, 파일 경로/ID
- 선택 샘플 히트맵 오버레이(Attention Rollout)
- Gap 여부/사유(δ 임계 초과 여부) 및 간단 가이드 문구
- "Visual Spec 내보내기"(HTML/PNG ZIP)

## 알고리즘/동작 흐름

1) 좌표 준비: `umap3d_S/U`를 L2 정규화 → 단위구 좌표 저장
2) 클릭 시: 최근 클릭 좌표 `q`(단위구)

- 후보 집합 선택(설정에 따라 S/U/Both)
- 3D 좌표에서 최근접 top-K 검색(KDTree/FAISS, 임시로 sklearn NearestNeighbors)
- 각 후보의 `dist_to_S`(미리 계산한 U→S 최근접거리 사용) 리포트
- δ 임계 비교로 Gap 여부 판단
3) 인사이트 생성:
- 선택 상위 N 이미지 썸네일
- Attention Rollout 히트맵(InternViT attention 가중치 활용) 오버레이
- 사양 문구 템플릿: "붉은 영역(모델 주목)과 유사한 시각적 패턴 수집"
4) 내보내기: 썸네일+히트맵 캔버스 묶음 / HTML 리포트 저장

## 성능/안정성

- 대규모 포인트: 샘플링/타일링, density aggregation 옵션
- 최근접 질의: 데이터 크기↑ 시 FAISS 선택(옵션)
- Attention Rollout: 미니배치/캐시, 실패 시 히트맵 생략/경고 처리

## 실행 방법(예시)

- 파이프라인으로 임베딩/UMAP/거리/δ 산출 완료 후:
- `python apps/dash_spherical_viewer.py --host 0.0.0.0 --port 8050 --data-root outputs/`

## 주요 파라미터

- `viewer.top_k`: 8
- `viewer.metric`: cosine
- `viewer.delta_threshold`: 기본 `coverage.json`의 δcover
- `viewer.point_limit`: 50k(초과 시 샘플링)
- `viewer.search_space`: U(기본)

## 주의/제약

- UMAP 3D 좌표 기반 이웃은 근사(원공간 거리와 불일치 가능). 필요 시 임시 후보(top-M)만 3D로 찾고, 원 임베딩 거리로 재정렬(re-rank)
- InternViT attention 구조 접근은 버전별 차이 존재(모델 로더에서 콜백 제공)

## 산출물

- 웹앱: 구형 시각화/클릭/인사이트 인터랙션
- 내보내기: 시각적 수집 가이드(HTML/PNG ZIP)