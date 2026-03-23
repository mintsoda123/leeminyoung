"""
hooke_model.py — TensorFlow Linear Regression for Hooke's Law
Learns F = k * x  →  x = (m * g) / k  using gradient descent.

Physics:
  F = k * x           (Hooke's Law)
  F = m * g           (Gravitational force)
  x = m * g / k       (Spring displacement)

TF Model: y = W * x + b  (Linear Regression via Adam / SGD)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Physics constants ─────────────────────────────────────────────────────────
G = 9.8          # m/s²
K_TRUE = 10.0    # N/m  (spring constant — what the model should learn)

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_train_result = None


def generate_data(n_samples: int = 500, noise_std: float = 0.05):
    """Generate (mass, displacement) pairs with Gaussian noise."""
    np.random.seed(42)
    mass = np.linspace(0.1, 5.0, n_samples).astype(np.float32)
    displacement = (mass * G / K_TRUE).astype(np.float32)
    displacement += np.random.normal(0, noise_std, n_samples).astype(np.float32)
    return mass, displacement


def train_model(
    epochs: int = 500,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    n_samples: int = 500,
    noise_std: float = 0.05,
) -> dict:
    """
    Train a TensorFlow linear regression model on Hooke's Law data.
    Returns training metrics and saves PNG plots to output/.
    """
    import tensorflow as tf

    global _model, _train_result

    mass, disp = generate_data(n_samples, noise_std)

    # ── Build model ───────────────────────────────────────────────────────────
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # ── Custom callback: record epoch-level weight history ────────────────────
    epoch_weights = []
    epoch_biases = []

    class WeightHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            w, b = self.model.layers[0].get_weights()
            epoch_weights.append(float(w[0][0]))
            epoch_biases.append(float(b[0]))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=30, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=15, min_lr=1e-6
    )

    history = model.fit(
        mass, disp,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[WeightHistory(), early_stop, reduce_lr],
        verbose=0,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    loss_vals = history.history["loss"]
    val_loss_vals = history.history.get("val_loss", [])
    epochs_run = len(loss_vals)

    w_learned = float(model.layers[0].get_weights()[0][0][0])
    b_learned = float(model.layers[0].get_weights()[1][0])
    k_inferred = G / w_learned if w_learned != 0 else float("inf")

    # R² score
    y_pred = model(mass.reshape(-1, 1), training=False).numpy().flatten()
    ss_res = np.sum((disp - y_pred) ** 2)
    ss_tot = np.sum((disp - np.mean(disp)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    # RMSE vs theoretical
    disp_true_clean = mass * G / K_TRUE
    rmse = float(np.sqrt(np.mean((y_pred - disp_true_clean) ** 2)))
    mean_true = float(np.mean(disp_true_clean))
    accuracy_pct = float((1 - rmse / mean_true) * 100)

    _model = model

    result = {
        "epochs_run": epochs_run,
        "final_loss": round(float(loss_vals[-1]), 8),
        "final_val_loss": round(float(val_loss_vals[-1]), 8) if val_loss_vals else None,
        "r2_score": round(r2, 6),
        "rmse": round(rmse, 6),
        "accuracy_pct": round(accuracy_pct, 4),
        "weight": round(w_learned, 6),
        "bias": round(b_learned, 6),
        "k_inferred": round(k_inferred, 4),
        "k_true": K_TRUE,
        "converged": accuracy_pct >= 98.0,
        "loss_history": [round(v, 8) for v in loss_vals],
        "val_loss_history": [round(v, 8) for v in val_loss_vals],
        "epoch_weights": [round(v, 6) for v in epoch_weights],
        "epoch_biases": [round(v, 6) for v in epoch_biases],
    }
    _train_result = result

    # ── Generate plots ────────────────────────────────────────────────────────
    plots = _generate_all_plots(result, mass, disp, y_pred)
    result["plots"] = plots
    return result


def predict(mass_kg: float) -> dict:
    """Predict spring displacement for a given mass."""
    if _model is None:
        raise RuntimeError("Model not trained yet. Call train_model() first.")

    import tensorflow as tf
    x_in = np.array([[mass_kg]], dtype=np.float32)
    # Use direct call instead of .predict() to avoid Keras shape inference issues
    pred = float(_model(x_in, training=False).numpy()[0][0])
    theoretical = float(mass_kg * G / K_TRUE)
    error_pct = abs(pred - theoretical) / theoretical * 100 if theoretical != 0 else 0

    # Save prediction plot
    plot_path = _plot_prediction(mass_kg, pred, theoretical)

    return {
        "mass_kg": round(mass_kg, 4),
        "predicted_m": round(pred, 6),
        "predicted_cm": round(pred * 100, 4),
        "theoretical_m": round(theoretical, 6),
        "theoretical_cm": round(theoretical * 100, 4),
        "force_N": round(mass_kg * G, 4),
        "error_pct": round(error_pct, 4),
        "accuracy_pct": round(100 - error_pct, 4),
        "plot": plot_path,
    }


# ── Private plotting helpers ──────────────────────────────────────────────────

def _dark_ax(ax):
    ax.set_facecolor("#0d1424")
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")


def _generate_all_plots(result, mass, disp, y_pred) -> list:
    paths = []

    # ── 1. Loss curve (epoch vs loss) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0f1e")
    _dark_ax(ax)

    loss_h = result["loss_history"]
    val_h = result["val_loss_history"]
    ep = list(range(1, len(loss_h) + 1))

    ax.fill_between(ep, loss_h, alpha=0.12, color="#6366f1")
    ax.plot(ep, loss_h, color="#6366f1", linewidth=2, label="Train Loss (MSE)", zorder=4)
    if val_h:
        ax.fill_between(list(range(1, len(val_h) + 1)), val_h, alpha=0.08, color="#f59e0b")
        ax.plot(list(range(1, len(val_h) + 1)), val_h,
                color="#f59e0b", linewidth=1.5, linestyle="--",
                label="Val Loss", zorder=4)

    # Convergence annotation
    ax.axhline(loss_h[-1], color="#34d399", linewidth=1,
               linestyle=":", alpha=0.8, label=f"Final: {loss_h[-1]:.2e}")

    ax.set_title("Training Loss — Hooke's Law Linear Regression",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("MSE Loss", color="#94a3b8")
    ax.set_yscale("log")
    ax.legend(facecolor="#0d1424", edgecolor="#334155",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # ── 2. Regression fit ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0f1e")
    _dark_ax(ax)

    # True law
    mass_line = np.linspace(0, 5.5, 200)
    true_disp = mass_line * G / K_TRUE
    ax.plot(mass_line, true_disp, color="#10b981", linewidth=2.5,
            linestyle="--", label=f"True Law: x = mg/{K_TRUE:.0f}", zorder=5)

    # Training data scatter
    ax.scatter(mass, disp, color="#6366f1", s=6, alpha=0.4,
               label="Training Data (with noise)", zorder=3)

    # Model prediction line
    sort_idx = np.argsort(mass)
    ax.plot(mass[sort_idx], y_pred[sort_idx],
            color="#f59e0b", linewidth=2.5, label="TF Prediction", zorder=6)

    # Residuals (first 50 points)
    for i in range(0, len(mass), 10):
        ax.plot([mass[i], mass[i]], [disp[i], y_pred[i]],
                color="#f87171", linewidth=0.5, alpha=0.4)

    ax.set_title(
        f"Hooke's Law — Linear Regression Fit\n"
        f"k_inferred = {result['k_inferred']:.3f} N/m  "
        f"(true: {K_TRUE} N/m)  |  R² = {result['r2_score']:.4f}",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Mass (kg)", color="#94a3b8")
    ax.set_ylabel("Displacement (m)", color="#94a3b8")
    ax.legend(facecolor="#0d1424", edgecolor="#334155",
              labelcolor="white", fontsize=9)

    # Text box
    textstr = (f"W (learned) = {result['weight']:.4f}\n"
               f"b (bias)    = {result['bias']:.4f}\n"
               f"Accuracy    = {result['accuracy_pct']:.2f}%")
    props = dict(boxstyle="round,pad=0.5",
                 facecolor="#1e2d4a", edgecolor="#334155", alpha=0.85)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props, color="#e2e8f0",
            fontfamily="monospace")

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "regression_fit.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # ── 3. Weight convergence ─────────────────────────────────────────────────
    if result["epoch_weights"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0a0f1e")

        w_hist = result["epoch_weights"]
        b_hist = result["epoch_biases"]
        ep2 = list(range(1, len(w_hist) + 1))

        # Weight convergence
        ax = axes[0]
        _dark_ax(ax)
        w_true = G / K_TRUE
        ax.axhline(w_true, color="#10b981", linewidth=1.5,
                   linestyle="--", alpha=0.8, label=f"True W = g/k = {w_true:.4f}")
        ax.fill_between(ep2, w_hist, w_true, alpha=0.1, color="#f59e0b")
        ax.plot(ep2, w_hist, color="#f59e0b", linewidth=2,
                label="Learned Weight W")
        ax.set_title("Weight Convergence", color="white",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="#94a3b8")
        ax.set_ylabel("W (kg·s²/m → m/kg)", color="#94a3b8")
        ax.legend(facecolor="#0d1424", edgecolor="#334155",
                  labelcolor="white", fontsize=9)

        # Bias convergence
        ax = axes[1]
        _dark_ax(ax)
        ax.axhline(0, color="#10b981", linewidth=1.5,
                   linestyle="--", alpha=0.8, label="True bias = 0")
        ax.fill_between(ep2, b_hist, 0, alpha=0.1, color="#c084fc")
        ax.plot(ep2, b_hist, color="#c084fc", linewidth=2,
                label="Learned Bias b")
        ax.set_title("Bias Convergence", color="white",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="#94a3b8")
        ax.set_ylabel("Bias b", color="#94a3b8")
        ax.legend(facecolor="#0d1424", edgecolor="#334155",
                  labelcolor="white", fontsize=9)

        fig.suptitle("Gradient Descent — Parameter Convergence",
                     color="#e2e8f0", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, "weight_convergence.png")
        plt.savefig(p, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(p)

    return paths


def _plot_prediction(mass_kg: float, predicted: float, theoretical: float) -> str:
    """Generate a prediction result visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0a0f1e")

    # ── Left: Spring diagram ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d1424")

    # Ceiling
    ax.fill_between([-0.6, 0.6], [0, 0], [0.05, 0.05], color="#475569")
    ax.text(0, 0.1, "Fixed Support", ha="center", color="#64748b", fontsize=9)

    # Natural spring (coils)
    natural_len = 0.3
    extended_len = min(predicted, 1.5)  # cap for display
    n_coils = 10
    t = np.linspace(0, 1, 200)
    spring_x = 0.12 * np.sin(t * n_coils * 2 * np.pi)
    spring_y = -t * (natural_len + extended_len)

    ax.plot(spring_x, spring_y, color="#6366f1", linewidth=2, zorder=4)

    # Mass block
    y_mass = -(natural_len + extended_len)
    rect_h = 0.08
    rect_w = 0.3
    ax.fill_between([-rect_w / 2, rect_w / 2],
                    [y_mass - rect_h, y_mass - rect_h],
                    [y_mass, y_mass], color="#3b82f6", alpha=0.9)
    ax.text(0, y_mass - rect_h / 2, f"{mass_kg:.1f} kg",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")

    # Annotations
    ax.annotate("", xy=(0.5, y_mass), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="<->", color="#f59e0b", lw=1.5))
    ax.text(0.55, y_mass / 2,
            f"x = {predicted * 100:.2f} cm\n(predicted)",
            color="#f59e0b", fontsize=9, va="center")

    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(y_mass - 0.2, 0.2)
    ax.set_title(f"Spring Extension Diagram\nm = {mass_kg:.1f} kg",
                 color="white", fontsize=12, fontweight="bold")
    ax.axis("off")

    # ── Right: Comparison bar ─────────────────────────────────────────────────
    ax2 = axes[1]
    _dark_ax(ax2)

    categories = ["Theoretical\n(True Law)", "ML Predicted\n(TF Model)"]
    values = [theoretical * 100, predicted * 100]
    colors = ["#10b981", "#6366f1"]

    bars = ax2.bar(categories, values, color=colors, width=0.4,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.3f} cm",
                 ha="center", va="bottom",
                 color="white", fontsize=11, fontweight="bold")

    error_pct = abs(predicted - theoretical) / theoretical * 100
    ax2.text(0.5, 0.95,
             f"Accuracy: {100 - error_pct:.2f}%",
             transform=ax2.transAxes, ha="center",
             color="#34d399", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#0d2010", edgecolor="#34d399", alpha=0.8))

    ax2.set_title(f"Prediction vs Ground Truth\nm = {mass_kg} kg  |  F = {mass_kg * G:.1f} N",
                  color="white", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Displacement (cm)", color="#94a3b8")
    ax2.set_ylim(0, max(values) * 1.25)

    fig.suptitle(
        f"Hooke's Law Prediction  —  k = {K_TRUE} N/m  |  g = {G} m/s²",
        color="#e2e8f0", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "prediction_result.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return p
