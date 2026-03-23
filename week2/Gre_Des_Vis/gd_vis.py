"""
gd_vis.py — Gradient Descent Visualization
Visualizes how the Gradient Descent algorithm finds the minimum of a loss function.
Based on week2/04_gradient_descent_vis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def loss_fn(x: float) -> float:
    return float(x ** 2)


def gradient_fn(x: float) -> float:
    return float(2 * x)


def run_gradient_descent(
    start_x: float = -4.0,
    learning_rate: float = 0.1,
    n_steps: int = 20,
) -> dict:
    """
    Run gradient descent on y = x² and return trajectory + stats.
    """
    x = float(start_x)
    history = []

    for i in range(n_steps):
        loss = loss_fn(x)
        grad = gradient_fn(x)
        history.append({"step": i, "x": x, "loss": loss, "grad": grad})
        x = x - learning_rate * grad

    # Final step
    history.append({"step": n_steps, "x": x, "loss": loss_fn(x), "grad": gradient_fn(x)})

    final_x = history[-1]["x"]
    final_loss = history[-1]["loss"]
    converged = abs(final_x) < 0.05

    return {
        "history": history,
        "final_x": round(final_x, 6),
        "final_loss": round(final_loss, 8),
        "converged": converged,
        "start_x": start_x,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
    }


def plot_gd_path(result: dict) -> str:
    """
    Generate the Gradient Descent path visualization PNG.
    Returns the file path.
    """
    history = result["history"]
    xs = [h["x"] for h in history]
    ys = [h["loss"] for h in history]

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0a0f1e")

    # ── Left: Loss landscape + path ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#0d1424")

    x_range = np.linspace(-5, 5, 400)
    y_range = x_range ** 2

    # Gradient fill under curve
    ax1.fill_between(x_range, y_range, alpha=0.08, color="#6366f1")
    ax1.plot(x_range, y_range, color="#6366f1", linewidth=2.5,
             label=r"$L(x) = x^2$", zorder=3)

    # Minimum marker
    ax1.axvline(0, color="#10b981", linewidth=1, linestyle="--", alpha=0.5)
    ax1.axhline(0, color="#10b981", linewidth=1, linestyle="--", alpha=0.5)
    ax1.scatter([0], [0], color="#10b981", s=200, zorder=8,
                marker="*", label="Global Minimum (0, 0)")

    # GD path — color gradient from red (start) to green (end)
    n = len(xs)
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n))

    for i in range(n - 1):
        ax1.annotate(
            "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1.5),
            zorder=6,
        )

    sc = ax1.scatter(xs, ys, c=np.linspace(0, 1, n),
                     cmap="RdYlGn", s=80, zorder=7, edgecolors="white",
                     linewidths=0.5)

    # Start / End labels
    ax1.annotate(f"Start\nx={xs[0]:.2f}", xy=(xs[0], ys[0]),
                 xytext=(xs[0] + 0.6, ys[0] + 2),
                 color="#f87171", fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#f87171", lw=1))
    ax1.annotate(f"End\nx={xs[-1]:.4f}", xy=(xs[-1], ys[-1]),
                 xytext=(xs[-1] + 0.6, ys[-1] + 1.5),
                 color="#34d399", fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34d399", lw=1))

    ax1.set_title("Gradient Descent Path on $L(x) = x^2$",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Parameter $x$", color="#94a3b8", fontsize=11)
    ax1.set_ylabel("Loss $L(x)$", color="#94a3b8", fontsize=11)
    ax1.tick_params(colors="#64748b")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e2d4a")
    ax1.legend(facecolor="#0d1424", edgecolor="#334155",
               labelcolor="white", fontsize=9)

    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("Step Progress", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#64748b")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#64748b")

    # ── Right: Loss curve per step ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0d1424")

    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    # Area under curve
    ax2.fill_between(steps, losses, alpha=0.15, color="#f59e0b")
    ax2.plot(steps, losses, color="#f59e0b", linewidth=2.5,
             label="Loss per Step", zorder=4)
    ax2.scatter(steps, losses, c=np.linspace(0, 1, len(steps)),
                cmap="RdYlGn", s=60, zorder=5, edgecolors="white",
                linewidths=0.4)

    # Convergence annotations
    ax2.axhline(losses[-1], color="#34d399", linewidth=1,
                linestyle=":", alpha=0.7, label=f"Final Loss: {losses[-1]:.2e}")
    ax2.text(steps[-1] * 0.5, losses[-1] * 1.3,
             f"Converged → {losses[-1]:.4f}", color="#34d399",
             fontsize=9, ha="center")

    ax2.set_title("Loss Reduction per Step",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Gradient Descent Step", color="#94a3b8", fontsize=11)
    ax2.set_ylabel("Loss $L(x) = x^2$", color="#94a3b8", fontsize=11)
    ax2.tick_params(colors="#64748b")
    ax2.set_yscale("log")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e2d4a")
    ax2.legend(facecolor="#0d1424", edgecolor="#334155",
               labelcolor="white", fontsize=9)

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"Gradient Descent  |  lr={result['learning_rate']}  |  "
        f"steps={result['n_steps']}  |  "
        f"start={result['start_x']}  →  end={result['final_x']:.4f}",
        color="#e2e8f0", fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "gd_path.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def plot_learning_rate_comparison() -> str:
    """
    Compare different learning rates side by side.
    """
    lrs = [0.05, 0.1, 0.3, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0a0f1e")
    axes = axes.flatten()

    x_range = np.linspace(-5, 5, 400)
    y_range = x_range ** 2

    titles_and_colors = [
        ("Slow (lr=0.05)", "#60a5fa"),
        ("Optimal (lr=0.1)", "#34d399"),
        ("Fast (lr=0.3)", "#f59e0b"),
        ("Diverging (lr=0.9)", "#f87171"),
    ]

    for idx, (lr, (title, color)) in enumerate(zip(lrs, titles_and_colors)):
        ax = axes[idx]
        ax.set_facecolor("#0d1424")
        ax.fill_between(x_range, y_range, alpha=0.06, color=color)
        ax.plot(x_range, y_range, color="#475569", linewidth=1.5, zorder=2)

        res = run_gradient_descent(start_x=-4.0, learning_rate=lr, n_steps=20)
        xs = [h["x"] for h in res["history"]]
        ys = [h["loss"] for h in res["history"]]

        for i in range(len(xs) - 1):
            ax.annotate(
                "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.8),
            )

        ax.scatter(xs, ys, color=color, s=50, zorder=5,
                   edgecolors="white", linewidths=0.3)
        ax.scatter([0], [0], color="#10b981", s=150, zorder=8, marker="*")

        converged_str = "✓ Converged" if res["converged"] else "✗ Not converged"
        ax.set_title(f"{title}\n{converged_str} | final x={res['final_x']:.4f}",
                     color=color, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("x", color="#64748b", fontsize=9)
        ax.set_ylabel("L(x)", color="#64748b", fontsize=9)
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-1, 26)

    fig.suptitle("Learning Rate Comparison — Gradient Descent on $L(x) = x^2$",
                 color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "lr_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path
