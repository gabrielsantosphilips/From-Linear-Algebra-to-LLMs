#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "anim"
OUT.mkdir(parents=True, exist_ok=True)


def gradient_descent_gif() -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(-3, 3, 400)
    y = x**2 + 0.4 * np.sin(3 * x)
    ax.plot(x, y, color="steelblue")
    ax.set_title("Gradient descent on a loss curve")
    ax.set_xlabel("w")
    ax.set_ylabel("L(w)")

    w = [2.6]
    points = [w[0]]
    for _ in range(28):
        grad = 2 * w[-1] + 1.2 * np.cos(3 * w[-1])
        w.append(w[-1] - 0.08 * grad)
        points.append(w[-1])

    scat = ax.scatter([], [], c="crimson", s=55)
    trail, = ax.plot([], [], color="crimson", linewidth=1.5)

    def update(i: int):
        xi = np.array(points[: i + 1])
        yi = xi**2 + 0.4 * np.sin(3 * xi)
        scat.set_offsets(np.array([[xi[-1], yi[-1]]]))
        trail.set_data(xi, yi)
        return scat, trail

    ani = FuncAnimation(fig, update, frames=len(points), interval=120, blit=True)
    ani.save(OUT / "gradient-descent.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


def self_attention_gif() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    keys = np.array([[7, 4.8], [7, 3.0], [7, 1.2]])
    query = np.array([2, 3.0])
    value = np.array([8.8, 3.0])
    weights = np.array([0.2, 0.55, 0.25])

    ax.scatter(*query, s=180, c="#1f77b4")
    ax.text(query[0] - 0.35, query[1] + 0.5, "Query")
    ax.scatter(keys[:, 0], keys[:, 1], s=140, c="#ff7f0e")
    for idx, (kx, ky) in enumerate(keys, 1):
        ax.text(kx + 0.2, ky + 0.05, f"Key {idx}")
    ax.scatter(*value, s=180, c="#2ca02c")
    ax.text(value[0] - 0.6, value[1] + 0.5, "Weighted sum")

    lines = [ax.plot([], [], color="#d62728", alpha=0.4 + w, linewidth=1.2 + 4 * w)[0] for w in weights]
    label = ax.text(0.4, 5.4, "Attention weights forming...", fontsize=11)

    def update(i: int):
        phase = min(1.0, i / 18)
        for j, line in enumerate(lines):
            end = query + phase * (keys[j] - query)
            line.set_data([query[0], end[0]], [query[1], end[1]])
        if i > 18:
            p = min(1.0, (i - 18) / 10)
            end = keys.T @ weights
            mixed = query + p * (end - query)
            label.set_text(f"Weighted sum = {weights.round(2)}")
            ax.plot([query[0], mixed[0]], [query[1], mixed[1]], color="#2ca02c", linewidth=3, alpha=0.5)
        return lines + [label]

    ani = FuncAnimation(fig, update, frames=30, interval=130, blit=False)
    ani.save(OUT / "self-attention.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


def linear_transform_svd_gif() -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Linear map: rotate → scale → rotate")

    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    line, = ax.plot([], [], color="#1f77b4", linewidth=2)

    th1, th2 = np.deg2rad(35), np.deg2rad(-20)
    R1 = np.array([[np.cos(th1), -np.sin(th1)], [np.sin(th1), np.cos(th1)]])
    S = np.diag([2.0, 0.6])
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])

    def transform(alpha: float):
        if alpha < 1 / 3:
            a = alpha * 3
            M = (1 - a) * np.eye(2) + a * R1
        elif alpha < 2 / 3:
            a = (alpha - 1 / 3) * 3
            M = S @ R1
            M = (1 - a) * R1 + a * M
        else:
            a = (alpha - 2 / 3) * 3
            M0 = S @ R1
            M1 = R2 @ S @ R1
            M = (1 - a) * M0 + a * M1
        return M @ circle

    def update(i: int):
        alpha = i / 39
        shp = transform(alpha)
        line.set_data(shp[0], shp[1])
        return (line,)

    ani = FuncAnimation(fig, update, frames=40, interval=90, blit=True)
    ani.save(OUT / "linear-transform-svd.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def softmax_temperature_gif() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4))
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    bars = ax.bar(range(4), [0, 0, 0, 0], color="#9467bd")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(4), ["tok1", "tok2", "tok3", "tok4"])
    title = ax.set_title("Softmax temperature sweep")

    def update(i: int):
        T = 0.4 + 2.2 * (i / 39)
        p = np.exp(logits / T)
        p /= p.sum()
        for b, h in zip(bars, p):
            b.set_height(h)
        title.set_text(f"Softmax temperature sweep (T={T:.2f})")
        return (*bars,)

    ani = FuncAnimation(fig, update, frames=40, interval=100, blit=True)
    ani.save(OUT / "softmax-temperature.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


def main() -> None:
    gradient_descent_gif()
    self_attention_gif()
    linear_transform_svd_gif()
    softmax_temperature_gif()
    print(f"Generated GIFs in {OUT}")


if __name__ == "__main__":
    main()
