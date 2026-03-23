# week2_submission
202312143 이민영 week2에 대한 과제 제출입니다. 

## 기술 스택

| 분류 | 기술 |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | Tailwind CSS, KaTeX |
| 시각화 | Matplotlib → PNG |
| ML / 수치계산 | TensorFlow, NumPy |

---

# 01. 훅의 법칙 × TensorFlow

**폴더:** `LinRegSpr/` | **포트:** `localhost:8000`

> 용수철에 질량을 달면 얼마나 늘어나는가? — AI가 물리 법칙을 스스로 학습

훅의 법칙 `F = kx`를 TensorFlow의 가장 단순한 신경망(`Dense(1)`)으로 학습합니다. 이론값 k=2, L₀=10에 AI가 스스로 수렴해가는 과정을 Epoch별 Loss 그래프로 확인할 수 있습니다.

sklearn 등 외부 ML 라이브러리 없이 **TensorFlow + NumPy만으로** 직접 구현했으며, 60개 샘플 데이터로 R² > 0.98 달성을 목표로 자동 재시도 학습을 수행합니다.


---

## 알고리즘 흐름

```
1. 데이터 생성  →  k=2 cm/kg, L₀=10 cm 기준 60개 샘플 (σ=0.45 노이즈 추가)
2. 모델 구성   →  Dense(1) 단층 신경망, Adam lr=0.08
3. 반복 학습   →  R² < 0.98이면 Epoch × 1.6 늘려서 최대 12회 재시도
4. 시각화      →  회귀선 / Loss 곡선 / 잔차 / Loss Landscape PNG 저장
```

---

## 기능

- Epoch 수 슬라이더 조절 (100~2000)
- 질량(kg) 입력 → 용수철 길이 예측 + 이론값 오차(%) 출력
- 학습 결과: R² / 학습된 k, b / 목표 달성 여부 표시
- Loss 곡선 / 회귀선 / 잔차 분석 / Loss Landscape PNG 저장

---

## 핵심 개념

| 개념 | 설명 |
|---|---|
| 선형회귀 | `y = wx + b` 형태로 데이터를 직선으로 근사 |
| Dense(1) | 입력 1개 → 출력 1개 선형 변환층 |
| MSE Loss | (예측 - 실제)² 평균, 작을수록 좋음 |
| Adam | 학습률 자동 조정 경사하강법 |
| R² Score | 예측 정확도 지표 (목표: 0.98 이상) |
| Loss Landscape | k, b 파라미터 공간에서의 손실 지형도 |

---

## 프로젝트 구조

```
LinRegSpr/
├── main.py             # FastAPI 서버 (학습 / 예측 API)
├── train_model.py      # TensorFlow 모델 학습 및 시각화
├── requirements.txt    # 의존성 패키지
├── static/
│   └── index.html      # 프론트엔드 UI
└── output/             # 생성된 PNG 저장 디렉토리
    ├── spring_fitting.png
    ├── loss_curve.png
    ├── residuals.png
    └── loss_landscape.png
```

---

## 서버 실행

```bash
pip install -r requirements.txt
python week2/LinRegSpr/main.py
```

또는

```bash
uvicorn main:app --reload --port 8000
```

---

# 02. K-Means 군집화 × NumPy

**폴더:** `UnSupClust/` | **포트:** `localhost:8001`

> 정답(레이블) 없이 데이터 스스로 3개 그룹을 찾아가는 과정

sklearn 등 외부 ML 라이브러리를 일절 사용하지 않고, **numpy 수식만으로** K-Means를 처음부터 구현했습니다. 90개 2D 데이터가 초기화 → 배정 → 이동 → 수렴 4단계를 반복하며 군집을 찾는 전 과정을 시각화합니다.

---

## 알고리즘 4단계

```
1. 초기화  →  데이터에서 K개(=3) 랜덤 중심점 선택
2. 배정    →  각 점을 가장 가까운 중심점에 배정 (유클리드 거리)
3. 이동    →  각 클러스터의 평균으로 중심점 갱신
4. 수렴    →  중심점 변화량 < 1e-6 이면 조기 종료
```

---

## 기능

- 최대 반복 횟수 슬라이더 (1~10회)
- (x, y) 입력 → 가장 가까운 클러스터 예측
- 군집 결과 산점도(★ 중심점) / WCSS Loss 곡선 / 예측 결과 PNG 저장

---

## 핵심 개념

| 개념 | 설명 |
|---|---|
| 비지도 학습 | 정답 없이 데이터 구조만으로 패턴 발견 |
| WCSS | Within-Cluster Sum of Squares, 군집 밀집도 지표 |
| 유클리드 거리 | 두 점 사이 직선 거리로 가장 가까운 클러스터 결정 |

---

## 서버 실행

```bash
python week2/UnSupClust/main.py
```
