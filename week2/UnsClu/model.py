"""
Hooke's Law + K-Means Unsupervised Learning Model
훅의 법칙: F = k * x  →  x = (m * g) / k
K-Means: 레이블 없이 스프링 종류를 자동 분류
TensorFlow: 질량 → 늘어남 예측 모델 학습
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state — 클러스터별 독립 모델
_models        = {}   # {cluster_id: tf.keras.Model}
_scalers_x     = {}   # {cluster_id: MinMaxScaler}
_scalers_y     = {}   # {cluster_id: MinMaxScaler}
_kmeans        = None
_is_trained    = False
_r2            = 0.0
_epochs_trained = 0
_loss_history  = []        # cluster 0 기준 대표 loss
_val_loss_history = []
_all_loss_history = {}     # {cluster_id: [loss...]}
_all_val_loss_history = {} # {cluster_id: [val_loss...]}
_cluster_labels  = None
_cluster_centers = None

G = 9.8  # gravitational acceleration (m/s²)

# Spring definitions
SPRINGS = [
    {"k": 10.0,  "name": "Soft Spring (k=10 N/m)",   "color": "#6366f1"},
    {"k": 30.0,  "name": "Medium Spring (k=30 N/m)", "color": "#10b981"},
    {"k": 60.0,  "name": "Hard Spring (k=60 N/m)",   "color": "#f59e0b"},
]
CLUSTER_COLORS = ["#6366f1", "#10b981", "#f59e0b"]
CENTROID_COLOR = "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_spring_data(n_per_spring: int = 50, noise_std: float = 0.005):
    """
    3종 스프링 데이터 생성 (K-Means에게는 레이블을 주지 않음)
    Returns:
        X      : shape (N, 2)  [mass_kg, extension_m]
        y_true : shape (N,)    [spring index 0/1/2]
        y_ext  : shape (N,)    [extension_m for TF training]
    """
    rng = np.random.default_rng(42)
    all_mass, all_ext, all_labels = [], [], []
    for idx, sp in enumerate(SPRINGS):
        masses = rng.uniform(0.1, 2.0, n_per_spring)
        ext = (masses * G) / sp["k"] + rng.normal(0, noise_std, n_per_spring)
        all_mass.append(masses)
        all_ext.append(ext)
        all_labels.append(np.full(n_per_spring, idx))

    X = np.column_stack([np.concatenate(all_mass), np.concatenate(all_ext)])
    y_true = np.concatenate(all_labels).astype(int)
    y_ext = X[:, 1]
    return X, y_true, y_ext


# ─────────────────────────────────────────────────────────────────────────────
# 2. K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int = 3):
    """
    K-Means 군집화 + 시각화 PNG 저장
    피처: extension/mass 비율 (= g/k, 스프링마다 고유한 상수)
    → K-Means가 같은 종류의 스프링을 완벽하게 분류
    시각화는 [mass, extension] 원래 공간에서 수행
    """
    global _kmeans, _cluster_labels, _cluster_centers

    # 클러스터링 피처: extension/mass = g/k (스프링 상수의 역수에 비례)
    # 같은 스프링은 이 값이 일정 → K-Means가 완벽히 분리
    ratio = (X[:, 1] / X[:, 0]).reshape(-1, 1)

    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(ratio)

    # 클러스터를 비율 내림차순 정렬 (soft spring = 0, hard spring = k-1)
    order = np.argsort(-km.cluster_centers_.ravel())
    remap = np.zeros(k, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    labels = np.array([remap[l] for l in labels])

    centers = np.array([
        X[labels == c].mean(axis=0) for c in range(k)
    ])

    _kmeans = km
    _cluster_labels = labels
    _cluster_centers = centers.tolist()

    _save_kmeans_plot(X, labels, centers, k)
    _save_hookes_law_plot(X, labels, k)

    return labels, centers


def _save_kmeans_plot(X, labels, centers, k):
    """PNG 1: K-Means 전/후 비교 (Before / After)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")

    masses, exts = X[:, 0], X[:, 1]

    # ── Before: 모든 데이터를 단색으로 (레이블 없음) ──
    ax0 = axes[0]
    ax0.set_facecolor("#1e293b")
    ax0.scatter(masses, exts, color="#94a3b8", alpha=0.6, s=40, edgecolors="none")
    ax0.set_title("Before K-Means\n(No Labels — Unlabeled Data)", color="white", fontsize=13, pad=12)
    ax0.set_xlabel("Mass (kg)", color="#94a3b8", fontsize=11)
    ax0.set_ylabel("Extension (m)", color="#94a3b8", fontsize=11)
    ax0.tick_params(colors="#64748b")
    for spine in ax0.spines.values():
        spine.set_color("#334155")
    ax0.grid(True, color="#1e293b", alpha=0.5)

    # ── After: 군집별 색상 + 중심점 ──
    ax1 = axes[1]
    ax1.set_facecolor("#1e293b")
    for c in range(k):
        mask = labels == c
        ax1.scatter(masses[mask], exts[mask],
                    color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    alpha=0.7, s=50, edgecolors="white", linewidths=0.4,
                    label=f"Cluster {c+1}")
    ax1.scatter(centers[:, 0], centers[:, 1],
                color=CENTROID_COLOR, marker="*", s=400, zorder=5,
                label="Centroids", edgecolors="white", linewidths=0.8)
    ax1.set_title("After K-Means\n(Auto-Grouped Without Labels)", color="white", fontsize=13, pad=12)
    ax1.set_xlabel("Mass (kg)", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Extension (m)", color="#94a3b8", fontsize=11)
    ax1.tick_params(colors="#64748b")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, color="#1e293b", alpha=0.5)
    legend = ax1.legend(facecolor="#1e293b", edgecolor="#334155",
                        labelcolor="white", fontsize=9)

    fig.suptitle("Unsupervised Learning: K-Means Clustering on Spring Data",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_kmeans_clustering.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PNG] Saved: {path}")


def _save_hookes_law_plot(X, labels, k):
    """PNG 2: 군집별 훅의 법칙 그래프 (mass vs extension)"""
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 6), sharey=False)
    fig.patch.set_facecolor("#0f172a")
    if k == 1:
        axes = [axes]

    for c in range(k):
        ax = axes[c]
        ax.set_facecolor("#1e293b")
        mask = labels == c
        m_c, e_c = X[mask, 0], X[mask, 1]

        # Scatter
        ax.scatter(m_c, e_c, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   alpha=0.7, s=50, edgecolors="white", linewidths=0.4, zorder=3)

        # Fit line (Hooke's Law: e = (g/k) * m)
        if len(m_c) > 1:
            coeffs = np.polyfit(m_c, e_c, 1)
            m_fit = np.linspace(m_c.min(), m_c.max(), 100)
            ax.plot(m_fit, np.polyval(coeffs, m_fit),
                    color="white", lw=2, ls="--", alpha=0.8, zorder=4,
                    label=f"slope≈{coeffs[0]:.3f}")
            # Estimate k from slope = g/k → k = g/slope
            k_est = G / coeffs[0] if coeffs[0] > 0 else 0
            ax.set_title(f"Cluster {c+1}\nEstimated k ≈ {k_est:.1f} N/m",
                         color="white", fontsize=12, pad=10)
            ax.legend(facecolor="#1e293b", edgecolor="#334155",
                      labelcolor="#94a3b8", fontsize=9)

        ax.set_xlabel("Mass (kg)", color="#94a3b8", fontsize=10)
        ax.set_ylabel("Extension (m)", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#1e293b", alpha=0.5)

    fig.suptitle("Hooke's Law: F = k·x  →  Extension per Cluster",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_hookes_law_groups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PNG] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TensorFlow Model
# ─────────────────────────────────────────────────────────────────────────────

def _build_tf_model(learning_rate: float = 0.001):
    """
    훅의 법칙 = 선형 관계 → Linear → SELU → Linear 구조
    SELU: 자기 정규화, dying ReLU 없음, 선형에 가까운 수렴
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="selu", input_shape=(1,),
                              kernel_initializer="lecun_normal"),
        tf.keras.layers.Dense(8,  activation="selu",
                              kernel_initializer="lecun_normal"),
        tf.keras.layers.Dense(1),
    ], name="HookesLawNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _train_one_cluster(masses_c, ext_c, epochs, lr, cluster_id):
    """
    클러스터 하나 독립 훈련
    - Mini-batch (16): Adam 옵티마이저 최적 동작, 빠른 수렴
    - 시드 고정: 재현 가능한 결과
    - EarlyStopping 제거: 충분한 epochs 동안 안정적 학습
    """
    tf.random.set_seed(42 + cluster_id)
    np.random.seed(42 + cluster_id)

    sx = MinMaxScaler()
    sy = MinMaxScaler()
    X = sx.fit_transform(masses_c.reshape(-1, 1))
    y = sy.fit_transform(ext_c.reshape(-1, 1)).ravel()

    idx   = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * 0.8)
    tr, val = idx[:split], idx[split:]

    m = _build_tf_model(lr)

    # Mini-batch 16: Adam이 설계된 방식대로 동작 → 빠른 수렴
    h = m.fit(
        X[tr], y[tr],
        validation_data=(X[val], y[val]),
        epochs=epochs,
        batch_size=16,
        verbose=0,
    )

    loss_h = [float(v) for v in h.history["loss"]]
    val_h  = [float(v) for v in h.history["val_loss"]]
    return m, sx, sy, loss_h, val_h


def train_tensorflow(masses: np.ndarray, extensions: np.ndarray,
                     epochs: int = 200, learning_rate: float = 0.001,
                     cluster_labels: np.ndarray = None):
    """
    클러스터별 독립 TF 모델 훈련
    — 각 모델이 하나의 스프링 종류만 학습 → R² >> 0.99 달성
    """
    global _models, _scalers_x, _scalers_y, _is_trained, _r2
    global _epochs_trained, _loss_history, _val_loss_history
    global _all_loss_history, _all_val_loss_history

    _models.clear(); _scalers_x.clear(); _scalers_y.clear()
    _all_loss_history.clear(); _all_val_loss_history.clear()
    tf.keras.backend.clear_session()

    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels is not None else 1
    all_pred = np.zeros_like(extensions)

    for c in range(n_clusters):
        mask = (cluster_labels == c) if cluster_labels is not None else np.ones(len(masses), bool)
        m_c, e_c = masses[mask], extensions[mask]

        model_c, sx_c, sy_c, lh, vlh = _train_one_cluster(m_c, e_c, epochs, learning_rate, c)
        _models[c]    = model_c
        _scalers_x[c] = sx_c
        _scalers_y[c] = sy_c
        _all_loss_history[c]     = lh
        _all_val_loss_history[c] = vlh

        # 이 클러스터의 예측값 수집
        X_all = sx_c.transform(m_c.reshape(-1, 1))
        p_norm = model_c.predict(X_all, verbose=0).ravel()
        all_pred[mask] = sy_c.inverse_transform(p_norm.reshape(-1, 1)).ravel()

    # 대표 loss = 가장 긴 클러스터의 history
    rep = max(_all_loss_history, key=lambda c: len(_all_loss_history[c]))
    _loss_history     = _all_loss_history[rep]
    _val_loss_history = _all_val_loss_history[rep]
    _epochs_trained   = max(len(v) for v in _all_loss_history.values())

    _r2 = float(r2_score(extensions, all_pred))
    _is_trained = True

    _save_loss_plot()
    _save_prediction_plot(masses, extensions, all_pred, cluster_labels)

    return _loss_history, _val_loss_history, _r2


def _save_loss_plot():
    """PNG 3: 클러스터별 Training Loss Curve (3개 서브플롯)"""
    n = len(_all_loss_history)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    fig.patch.set_facecolor("#0f172a")
    if n == 1:
        axes = [axes]

    clr_train = ["#6366f1", "#10b981", "#f59e0b"]
    clr_val   = ["#a5b4fc", "#6ee7b7", "#fcd34d"]
    k_names   = ["Soft (k≈10)", "Medium (k≈30)", "Hard (k≈60)"]

    for c, ax in enumerate(axes):
        ax.set_facecolor("#1e293b")
        lh  = _all_loss_history.get(c, [])
        vlh = _all_val_loss_history.get(c, [])
        ep  = range(1, len(lh) + 1)

        ax.plot(ep, lh,  color=clr_train[c % 3], lw=2,
                label=f"Train Loss", zorder=3)
        ax.plot(ep, vlh, color=clr_val[c % 3],   lw=2, ls="--",
                label=f"Val Loss",   zorder=3)

        if vlh:
            best_ep  = int(np.argmin(vlh)) + 1
            best_val = min(vlh)
            ax.axvline(best_ep, color="white", lw=1, ls=":", alpha=0.5)
            ax.text(best_ep + 1, best_val, f"E{best_ep}\n{best_val:.5f}",
                    color="white", fontsize=7, va="bottom")

        ax.set_title(f"Cluster {c+1} — {k_names[c % 3]}\n"
                     f"Epochs: {len(lh)}",
                     color="white", fontsize=11, pad=8)
        ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
        ax.set_ylabel("MSE Loss", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#334155", alpha=0.4, lw=0.5)
        ax.legend(facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="white", fontsize=9)
        ax.set_yscale("log")

    fig.suptitle(f"TensorFlow Training Loss — Per-Cluster Models  |  Overall R² = {_r2:.4f}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_training_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PNG] Saved: {path}")


def _save_prediction_plot(masses, extensions, y_pred, cluster_labels=None):
    """PNG 4: 실제값 vs 예측값 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")

    # ── Left: Scatter actual vs predicted ──
    ax0 = axes[0]
    ax0.set_facecolor("#1e293b")
    mn, mx = extensions.min(), extensions.max()
    ax0.plot([mn, mx], [mn, mx], color="#f59e0b", lw=2, ls="--",
             label="Perfect Prediction", zorder=2)
    ax0.scatter(extensions, y_pred, color="#6366f1", alpha=0.6, s=40,
                edgecolors="white", linewidths=0.3, zorder=3, label="Predictions")
    ax0.set_title(f"Actual vs Predicted Extension\nR² = {_r2:.4f}",
                  color="white", fontsize=12, pad=10)
    ax0.set_xlabel("Actual Extension (m)", color="#94a3b8", fontsize=10)
    ax0.set_ylabel("Predicted Extension (m)", color="#94a3b8", fontsize=10)
    ax0.tick_params(colors="#64748b")
    for spine in ax0.spines.values():
        spine.set_color("#334155")
    ax0.grid(True, color="#334155", alpha=0.4)
    ax0.legend(facecolor="#1e293b", edgecolor="#334155",
               labelcolor="white", fontsize=9)

    # ── Right: Mass vs Extension (actual + predicted line) ──
    ax1 = axes[1]
    ax1.set_facecolor("#1e293b")
    sort_idx = np.argsort(masses)
    ax1.scatter(masses, extensions, color="#94a3b8", alpha=0.5, s=30,
                label="Actual Data", zorder=2)
    ax1.plot(masses[sort_idx], y_pred[sort_idx], color="#10b981",
             lw=2, label="TF Prediction", zorder=3)

    # Theoretical lines
    m_line = np.linspace(masses.min(), masses.max(), 200)
    for sp in SPRINGS:
        e_theory = (m_line * G) / sp["k"]
        ax1.plot(m_line, e_theory, lw=1, ls=":", alpha=0.5,
                 label=f'Theory k={sp["k"]}')

    ax1.set_title("Mass → Extension Prediction\nvs Theoretical Curves",
                  color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Mass (kg)", color="#94a3b8", fontsize=10)
    ax1.set_ylabel("Extension (m)", color="#94a3b8", fontsize=10)
    ax1.tick_params(colors="#64748b")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, color="#334155", alpha=0.4)
    ax1.legend(facecolor="#1e293b", edgecolor="#334155",
               labelcolor="white", fontsize=8, ncol=2)

    fig.suptitle("TensorFlow Model: Hooke's Law Prediction Results",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_prediction_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PNG] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(mass_kg: float, cluster_id: int = 0) -> dict:
    """클러스터별 독립 모델로 예측"""
    if not _is_trained or not _models:
        raise RuntimeError("Model not trained yet.")

    n_clusters = len(_models)
    cluster_id = max(0, min(cluster_id, n_clusters - 1))

    def _pred(cid):
        sx, sy, m = _scalers_x[cid], _scalers_y[cid], _models[cid]
        xn = sx.transform([[mass_kg]])
        yn = m.predict(xn, verbose=0)
        return float(sy.inverse_transform(yn)[0][0])

    ext_m = _pred(cluster_id)
    all_predictions = [
        {"cluster_id": c,
         "extension_m":  round(_pred(c), 4),
         "extension_cm": round(_pred(c) * 100, 2)}
        for c in range(n_clusters)
    ]
    k_est = float((mass_kg * G) / ext_m) if ext_m > 0 else 0.0

    return {
        "mass_kg": mass_kg,
        "extension_m": round(ext_m, 4),
        "extension_cm": round(ext_m * 100, 2),
        "spring_constant_estimated": round(k_est, 2),
        "cluster_id": cluster_id,
        "all_predictions": all_predictions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Status
# ─────────────────────────────────────────────────────────────────────────────

def get_status() -> dict:
    return {
        "trained": _is_trained,
        "r2_score": round(_r2, 4),
        "epochs_trained": _epochs_trained,
        "loss_history": _loss_history[-20:] if _loss_history else [],
        "val_loss_history": _val_loss_history[-20:] if _val_loss_history else [],
        "cluster_centers": _cluster_centers,
    }


def get_plots() -> list:
    files = ["01_kmeans_clustering.png", "02_hookes_law_groups.png",
             "03_training_loss.png", "04_prediction_comparison.png"]
    return [f for f in files if os.path.exists(os.path.join(OUTPUT_DIR, f))]
