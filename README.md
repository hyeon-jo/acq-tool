## 자율주행 데이터 다양성 분석 · 구형 Gap Explorer (Dash)

시각 임베딩(InternVL2 비전 인코더 기반)을 3차원으로 축소 후 단위 구에 투영하여, 사용자가 구를 회전/확대/클릭하면서 잠재 공간 상의 빈 영역을 탐색하고, 클릭 지점 주변의 최근접 후보 이미지를 즉시 확인·내보내기 할 수 있는 대화형 웹앱입니다. 선택한 샘플에 대해 Attention Rollout 히트맵을 생성해 시각적 수집 가이드를 제공합니다.

### 핵심 개념
- **Geometric Coverage 관점**: 임베딩 공간에서 현재 보유 데이터셋(S)이 커버하지 못하는 희소 영역을 시각적으로 파악
- **UMAP 3D → 단위구 투영**: 시각적으로 안정된 구형 표면 위 산점도
- **클릭 → 주변 최근접 후보**: 클릭 지점 근방의 top-K 샘플과 δ 임계 기반 GAP 여부 표시
- **Attention Rollout 히트맵**: 선택된 샘플에 대해 모델의 주목 영역을 오버레이


## 구현 범위
- 구형 시각화와 인터랙션(Dash/Plotly)
- 클릭 지점 주변의 최근접 top-K 이미지 썸네일 및 경로/거리 표시
- U→S 최근접거리 기반 색상 매핑(부족도 척도)
- δ 임계(coverage.json) 기반 GAP 후보 식별
- Attention Rollout 히트맵 생성(InternVL2 비전 인코더의 self-attention 사용)
- 간단한 Visual Spec HTML 내보내기
- 대용량 대응을 위한 포인트 샘플링, FAISS(있으면 자동 사용) 기반 최근접 질의 가속


## 디렉터리 구조
- `apps/dash_spherical_viewer.py`: Dash 앱 엔트리(레이아웃/콜백/그래프/클릭 처리/내보내기/히트맵)
- `viz/sphere_utils.py`: 3D 좌표 정규화(단위구), 구 격자, 샘플링 유틸
- `insights/neighbor_query.py`: 클릭 좌표 → 최근접 이웃(top-K) 질의(FAISS 있으면 우선 사용)
- `explainability/attn_rollout.py`: Attention Rollout 히트맵 생성(모델 로더/전처리 내부 포함)
- `configs/config.yaml`: 기본 경로/파라미터(참고용)
- `requirements.txt`: 의존성 목록


## 설치
```bash
pip install -r requirements.txt
```

GPU 사용 권장. 히트맵은 모델 크기와 장치 메모리에 따라 시간이 소요됩니다.


## 데이터 준비(필수 산출물)
앱은 `--data-root`(기본값 `outputs/`) 디렉터리에서 다음 파일을 기대합니다.

- `umap3d_S.npy`: 보유 데이터셋 S의 3D 좌표 (shape: [NS, 3])
- `umap3d_U.npy`: 후보 데이터셋 U의 3D 좌표 (shape: [NU, 3])
- `nn_U_to_S.npy`: 각 U 포인트의 S까지의 최근접 거리 벡터 (len: NU)
- `coverage.json`: δ 커버리지 값
  - 예: `{"delta": 0.78}`
- `paths_S.json`: S 이미지의 절대 경로 배열
- `paths_U.json`: U 이미지의 절대 경로 배열

주의: 현재 저장소에는 임베딩/UMAP/거리 계산 파이프라인이 포함되어 있지 않습니다. 상기 파일들은 외부 파이프라인에서 생성해 주셔야 합니다. (차후 통합 가능)


## 실행
1) (선택) 로컬 모델 경로 설정: 히트맵 생성 시 우선 사용됩니다.
```bash
export INTERNVL2_PATH=/models/OpenGVLab/InternVL2-8B
```

2) 웹앱 실행:
```bash
python apps/dash_spherical_viewer.py --host 0.0.0.0 --port 8050 --data-root outputs/
```

3) 브라우저 접속: `http://<host>:8050`


## UI 사용법
### 좌측 컨트롤
- **표시 데이터**: S/U/Both 토글
- **투영**: spherical(단위 구) / cartesian(원래 3D)
- **점 최대 수**: 표시 포인트 상한(성능 튜닝)
- **top-K**: 클릭 지점 주변 최근접 후보의 개수
- **δ 임계값**: GAP 판정 기준. 비워두면 `coverage.json`의 δ 사용
- **검색 대상**: U/S/Both에서 후보를 찾음

### 중앙 그래프
- 마우스로 회전/확대/패닝
- 빈 영역으로 보이는 곳을 클릭하면 우측 패널에 해당 지점 주변의 top-K 후보가 나타납니다.

### 우측 패널
- **썸네일 그리드**: 파일명과 함께 표시, U 항목은 `dist_to_S`가 표기되며 δ 임계 초과 시 `GAP` 표시
- **히트맵 생성(선택 1개)**: 선택 결과의 첫 번째 이미지를 대상으로 Attention Rollout 히트맵 생성
- **내보내기**: 선택된 경로들을 포함한 간단한 Visual Spec HTML 다운로드


## 수집 가이드(Visual Spec) 내보내기
- 버튼 클릭 시 선택 목록을 HTML로 생성합니다.
- 가이드 문구 예시: “붉은색 강조 영역(모델 주목)과 유사한 시각적 패턴이 나타나는 환경을 수집하십시오.”
- 필요 시 PNG/ZIP 형식 확장은 추후 추가 가능합니다.


## 성능/확장 팁
- **점 최대 수**를 조정해 렌더링 부하를 낮출 수 있습니다.
- 후보가 매우 많다면 `insights/neighbor_query.py`가 FAISS를 자동 사용합니다(설치되어 있는 경우).
- 히트맵은 대용량 모델 로드가 필요합니다. GPU 메모리가 부족하면 해상도를 낮추거나(입력 리사이즈), CPU로 전환하십시오(속도 저하).
- δ 임계값을 낮추면 GAP 판정이 보수적으로 바뀌며, 높이면 보다 공격적으로 희소 영역을 표시합니다.


## 문제 해결
- 앱 실행 시 포인트가 보이지 않음
  - `--data-root` 하위 파일 유무/형식 확인(`umap3d_*.npy`, `paths_*.json`)
  - `umap3d_*.npy` shape가 (N, 3)인지 확인
- U 색상바가 나타나지 않음
  - `nn_U_to_S.npy` 길이가 `umap3d_U.npy`의 N과 동일한지 확인
- 히트맵 생성 실패
  - 환경변수 `INTERNVL2_PATH`가 올바른지, 혹은 HF ID 다운로드 권한/네트워크 확인
  - GPU 메모리 부족 시 CPU로 전환 또는 이미지 해상도 축소
- 브라우저에서 상호작용 지연
  - 점 수를 줄이거나, 브라우저 탭을 새로고침하여 메모리 확보


## 한계와 다음 단계
- 3D UMAP 이웃은 원 임베딩 공간의 이웃과 다를 수 있습니다. 필요 시 “3D에서 임시 후보→원 임베딩 거리 재정렬(re-rank)” 방식으로 보정하는 모드를 추가할 수 있습니다.
- Attention Rollout은 모델/버전별 구조 오차로 실패할 수 있습니다. 모델 로더를 공고히 하거나 Grad-CAM 옵션을 추가할 수 있습니다.
- 임베딩/UMAP/거리 산출 파이프라인을 통합해 원클릭 준비 기능을 제공하는 것이 차기 목표입니다.


## 참고 환경설정
- `configs/config.yaml`에 기본 파라미터가 정리되어 있습니다(참고용).
- 앱은 런타임 인자와 `outputs/` 파일들로 동작하므로, 설정 파일 없이도 실행 가능합니다.


## 라이선스/모델
- 모델: `OpenGVLab/InternVL2-8B` (Hugging Face). 로컬 경로 우선(`INTERNVL2_PATH`), 미지정 시 HF ID를 사용하여 히트맵 생성 시 로드합니다.


