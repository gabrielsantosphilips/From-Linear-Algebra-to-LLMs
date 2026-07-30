from manim import *
import numpy as np


class GradientDescentScene(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[-1, 8, 1], x_length=8, y_length=4)
        curve = axes.plot(lambda x: x**2 + 0.4 * np.sin(3 * x), color=BLUE)
        self.add(axes, curve)

        x = 2.6
        dot = Dot(axes.c2p(x, x**2 + 0.4 * np.sin(3 * x)), color=RED)
        self.add(dot)

        for _ in range(22):
            grad = 2 * x + 1.2 * np.cos(3 * x)
            x_next = x - 0.08 * grad
            self.play(dot.animate.move_to(axes.c2p(x_next, x_next**2 + 0.4 * np.sin(3 * x_next))), run_time=0.12)
            x = x_next
