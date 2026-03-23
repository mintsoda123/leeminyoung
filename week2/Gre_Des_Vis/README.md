# 04. 경사 하강법 × NumPy + TensorFlow
**폴더:** `week2/Gre_Des_Vis/` | **포트:** `localhost:8003`

> 산에서 공이 굴러 내려가듯, 기울기의 반대 방향으로 이동해 최솟값을 찾는다

손실 함수 `f(x) = x²`에서 경사 하강법이 최솟값 `x=0`을 향해 이동하는 과정을 시각화합니다.
학습률(lr)이 너무 크면 발산하는 것도 직접 체험할 수 있어 머신러닝의 핵심 최적화 개념을 가장 직관적으로 전달합니다.
나아가 이 원리를 **훅의 법칙(Hooke's Law)** 에 적용하여 TensorFlow가 스프링 상수 `k`를 학습하는 과정까지 시각화합니다.

---

## 업데이트 규칙

$$x_{t+1} = x_t - \alpha \cdot \nabla f(x_t) = x_t - \alpha \cdot 2x_t$$

---

## 학습률에 따른 동작 차이

| 학습률 (α) | 동작 | 결과 |
|-----------|------|------|
| 0.01 ~ 0.05 | 매우 조금씩 이동 | 수렴하지만 느림 |
| 0.1 ~ 0.4 | 적당히 이동 | 빠르고 안정적으로 수렴 ✓ |
| 0.8 ~ 0.9 | 최솟값을 넘어 진동 | 진동하며 느리게 수렴 |
| ≥ 1.0 | 반대 방향으로 튕김 | 발산 💥 |

---

## 기능

### ⚡ Tab 1 — 경사 하강법 시각화
- 시작점 x₀ 슬라이더 (-5 ~ 5)
- 학습률 α 슬라이더 (0.01 ~ 0.95) — 위험 구간 색상 경고
- 스텝 수 슬라이더 (5 ~ 60)
- 빠른 프리셋 4개: **기본 수렴 / 느린 수렴 / 빠른 수렴 / 💥 발산**
- Loss Landscape + 이동 경로 PNG 저장 (`output/gd_path.png`)
- 학습률 4종 비교 PNG 저장 (`output/lr_comparison.png`)
- 전체 Step 상세 테이블 (x, f(x), ∇f(x))

### 🔬 Tab 2 — Hooke's Law TensorFlow 학습
- TensorFlow `Dense(1)` 선형 회귀로 스프링 상수 `k = 10 N/m` 학습
- Epoch별 Loss 곡선 PNG 저장 (`output/loss_curve.png`)
- 회귀 적합선 PNG 저장 (`output/regression_fit.png`)
- 가중치 수렴 PNG 저장 (`output/weight_convergence.png`)
- 질량 입력 → 스프링 변위 예측 + SVG 스프링 애니메이션
- 예측 결과 PNG 저장 (`output/prediction_result.png`)

### 🖼️ Tab 3 — 결과 갤러리
- 생성된 PNG 파일 자동 표시 및 라이트박스 확대

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| Gradient Descent | 기울기 반대 방향으로 이동해 최솟값 탐색 |
| Learning Rate | 한 번에 이동하는 보폭, 너무 크면 발산 |
| 수렴 조건 | `\|x\| < 0.05` 이면 최솟값 도달로 판정 |
| 발산 조건 | 학습률 ≥ 1.0 이면 발산 위험 |
| Hooke's Law | `F = k·x`, `x = mg/k` — 스프링 탄성력 |
| Linear Regression | `y = Wx + b` — TF가 W(≈g/k)를 경사 하강으로 학습 |

---

## 정확도 검증 결과

| 항목 | 결과 |
|------|------|
| 훈련 정확도 | **99.88%** (목표 98% 초과) |
| R² Score | **0.9988** |
| 예측 정확도 (2.5 kg) | **99.96%** → 245.11 cm (이론: 245.0 cm) |
| k 추정값 | **10.004 N/m** (실제: 10 N/m) |

---

## 출력 파일 (`output/`)

| 파일 | 설명 |
|------|------|
| `gd_path.png` | 경사 하강 이동 경로 (Loss Landscape + Step별 Loss) |
| `lr_comparison.png` | 학습률 4종 비교 (Slow / Optimal / Fast / Diverging) |
| `loss_curve.png` | TF 훈련 Epoch별 MSE Loss 곡선 |
| `regression_fit.png` | 훅의 법칙 회귀 적합선 + 잔차 |
| `weight_convergence.png` | 가중치(W) · 편향(b) 수렴 과정 |
| `prediction_result.png` | 스프링 다이어그램 + 예측 vs 이론값 비교 |

---

## 프로젝트 구조

```
week2/Gre_Des_Vis/
├── main.py              # FastAPI 백엔드
├── gd_vis.py            # 경사 하강법 시각화 (NumPy + Matplotlib)
├── hooke_model.py       # TensorFlow 훅의 법칙 선형 회귀
├── requirements.txt
├── templates/
│   └── index.html       # Tailwind CSS 전문가 수준 UI
└── output/              # PNG 자동 저장
```

---

## 서버 실행

```bash
cd week2/Gre_Des_Vis
pip install -r requirements.txt
python main.py
# 또는
uvicorn main:app --reload --port 8003
```

→ **http://localhost:8003**

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 백엔드 | FastAPI + Uvicorn |
| 머신러닝 | TensorFlow 2.x (Keras) |
| 시각화 | Matplotlib (PNG 출력) |
| 수치 연산 | NumPy |
| 프론트엔드 | Tailwind CSS (CDN) + Vanilla JS |
| 폰트 | Inter + JetBrains Mono |
