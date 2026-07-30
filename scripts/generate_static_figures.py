#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def vectors_2d() -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    a = np.array([2, 1])
    b = np.array([3, 4])
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", label="a")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", label="b")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("2D vectors")
    save(fig, "vectors-2d.svg")


def vectors_dot_projection() -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    a = np.array([2, 1])
    b = np.array([3, 4])
    proj = (a @ b) / (b @ b) * b
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", label="a")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", label="b")
    ax.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1, color="#2ca02c", label="proj_b(a)")
    ax.plot([a[0], proj[0]], [a[1], proj[1]], "k--", linewidth=1)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    ax.set_title("Dot product as projection")
    save(fig, "vectors-dot-projection.svg")


def vectors_cosine() -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    a = np.array([2, 1])
    b = np.array([3, 4])
    theta = np.arccos((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    t = np.linspace(0, theta, 100)
    arc_r = 1.2
    ax.plot(arc_r * np.cos(t), arc_r * np.sin(t), color="purple")
    ax.text(1.05, 0.4, r"$\theta$")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", label="a")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", label="b")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Cosine similarity angle")
    save(fig, "vectors-cosine-angle.svg")


def linear_transform_grid() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    A = np.array([[1.2, 0.6], [-0.4, 1.1]])
    xs = np.linspace(-2, 2, 9)
    for x in xs:
        y = np.linspace(-2, 2, 100)
        pts = np.stack([np.full_like(y, x), y])
        tpts = A @ pts
        axes[0].plot(pts[0], pts[1], color="gray", alpha=0.6)
        axes[0].plot(y, np.full_like(y, x), color="gray", alpha=0.6)
        axes[1].plot(tpts[0], tpts[1], color="#1f77b4", alpha=0.7)
        pts2 = np.stack([y, np.full_like(y, x)])
        tpts2 = A @ pts2
        axes[1].plot(tpts2[0], tpts2[1], color="#ff7f0e", alpha=0.7)
    axes[0].set_title("Original grid")
    axes[1].set_title("After linear map A")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    save(fig, "linear-transform-grid.svg")


def eigenvectors_invariant() -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    A = np.array([[2.0, 0.0], [0.0, 0.8]])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    Av1, Av2 = A @ v1, A @ v2
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", label="v1")
    ax.quiver(0, 0, Av1[0], Av1[1], angles="xy", scale_units="xy", scale=1, color="#1f77b4", linestyle="--", label="A v1")
    ax.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", label="v2")
    ax.quiver(0, 0, Av2[0], Av2[1], angles="xy", scale_units="xy", scale=1, color="#ff7f0e", linestyle="--", label="A v2")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.set_title("Eigenvectors stay on their span")
    save(fig, "eigenvectors-invariant.svg")


def svd_rotate_scale_rotate() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    th1, th2 = np.deg2rad(30), np.deg2rad(-20)
    R1 = np.array([[np.cos(th1), -np.sin(th1)], [np.sin(th1), np.cos(th1)]])
    S = np.diag([2.0, 0.6])
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    step1 = R1 @ circle
    step2 = S @ step1
    step3 = R2 @ step2
    axes[0].plot(circle[0], circle[1], label="unit circle")
    axes[0].plot(step1[0], step1[1], label="rotate")
    axes[1].plot(step1[0], step1[1], label="rotated")
    axes[1].plot(step2[0], step2[1], label="scale")
    axes[2].plot(step2[0], step2[1], label="scaled")
    axes[2].plot(step3[0], step3[1], label="rotate")
    titles = ["Rotate", "Scale", "Rotate"]
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(title)
        ax.legend(fontsize=8)
    save(fig, "svd-rotate-scale-rotate.svg")


def pca_projection() -> None:
    rng = np.random.default_rng(42)
    cov = np.array([[3.0, 2.0], [2.0, 2.0]])
    pts = rng.multivariate_normal([0, 0], cov, size=200)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    v1, v2 = vecs[:, 0], vecs[:, 1]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.4)
    for v, color, label in [(v1, "#d62728", "PC1"), (v2, "#2ca02c", "PC2")]:
        ax.arrow(0, 0, v[0] * 3, v[1] * 3, width=0.03, color=color, label=label)
    x = np.array([2.5, -0.5])
    proj = (x @ v1) * v1
    ax.scatter([x[0]], [x[1]], color="black", s=35)
    ax.scatter([proj[0]], [proj[1]], color="#d62728", s=35)
    ax.plot([x[0], proj[0]], [x[1], proj[1]], "k--", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("PCA projection onto principal axis")
    ax.legend(loc="upper left")
    save(fig, "pca-projection.svg")


def softmax_temperature() -> None:
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    Ts = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(logits))
    w = 0.25
    for i, T in enumerate(Ts):
        p = np.exp(logits / T)
        p = p / p.sum()
        ax.bar(x + (i - 1) * w, p, width=w, label=f"T={T}")
    ax.set_xticks(x)
    ax.set_xticklabels(["tok1", "tok2", "tok3", "tok4"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Softmax temperature controls sharpness")
    ax.legend()
    save(fig, "softmax-temperature.svg")


def cross_entropy_curve() -> None:
    p = np.linspace(0.001, 0.999, 500)
    ce = -np.log(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p, ce, color="#9467bd")
    ax.set_xlabel("Assigned probability to correct class")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(r"$\ell=-\log p_{\mathrm{correct}}$")
    save(fig, "cross-entropy-curve.svg")


def activations() -> None:
    z = np.linspace(-4, 4, 500)
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)
    gelu = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z, sigmoid, label="sigmoid")
    ax.plot(z, tanh, label="tanh")
    ax.plot(z, relu, label="ReLU")
    ax.plot(z, gelu, label="GELU")
    ax.set_ylim(-1.5, 4.2)
    ax.set_title("Activation functions")
    ax.legend()
    save(fig, "activation-functions.svg")


def loss_surface_gd() -> None:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(-2.5, 2.5, 80)
    y = np.linspace(-2.5, 2.5, 80)
    X, Y = np.meshgrid(x, y)
    Z = 0.8 * X**2 + 1.5 * Y**2
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.75, linewidth=0)
    w = np.array([2.0, -2.0])
    eta = 0.15
    path = [w.copy()]
    for _ in range(12):
        grad = np.array([1.6 * w[0], 3.0 * w[1]])
        w = w - eta * grad
        path.append(w.copy())
    path = np.array(path)
    z_path = 0.8 * path[:, 0] ** 2 + 1.5 * path[:, 1] ** 2
    ax.plot(path[:, 0], path[:, 1], z_path, color="red", marker="o")
    ax.set_title("Loss surface with gradient descent trajectory")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("L(w)")
    save(fig, "loss-surface-gd.svg")


def attention_heatmap() -> None:
    A = np.array([[0.401, 0.401, 0.198], [0.198, 0.401, 0.401], [0.248, 0.503, 0.248]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=0.6)
    ax.set_xticks([0, 1, 2], ["k1", "k2", "k3"])
    ax.set_yticks([0, 1, 2], ["q1", "q2", "q3"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{A[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Self-attention weights (3-token example)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "attention-weight-heatmap.svg")


def positional_encoding_heatmap() -> None:
    pos = np.arange(0, 64)[:, None]
    d_model = 32
    i = np.arange(0, d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros_like(angles)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pe, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Position")
    ax.set_title("Sinusoidal positional encoding heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "positional-encoding-heatmap.svg")


def scaling_law_loglog() -> None:
    N = np.logspace(5, 10, 200)
    L_inf, a, alpha = 1.2, 25.0, 0.12
    L = L_inf + a * N ** (-alpha)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N, L, color="#8c564b")
    ax.set_xlabel("Model size N (parameters)")
    ax.set_ylabel("Validation loss L(N)")
    ax.set_title("Scaling law on log-log axes")
    save(fig, "scaling-law-loglog.svg")


def main() -> None:
    vectors_2d()
    vectors_dot_projection()
    vectors_cosine()
    linear_transform_grid()
    eigenvectors_invariant()
    svd_rotate_scale_rotate()
    pca_projection()
    softmax_temperature()
    cross_entropy_curve()
    activations()
    loss_surface_gd()
    attention_heatmap()
    positional_encoding_heatmap()
    scaling_law_loglog()
    print(f"Generated static figures in {OUT}")


if __name__ == "__main__":
    main()
