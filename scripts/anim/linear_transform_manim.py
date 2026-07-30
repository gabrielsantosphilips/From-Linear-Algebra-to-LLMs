from manim import *
import numpy as np


class LinearTransformSVDScene(Scene):
    def construct(self):
        plane = NumberPlane(x_range=[-3, 3, 1], y_range=[-3, 3, 1], background_line_style={"stroke_opacity": 0.4})
        circle = Circle(radius=1.0, color=BLUE)
        self.add(plane, circle)

        m1 = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
        m2 = np.array([[2.0, 0.0], [0.0, 0.6]])
        m3 = np.array([[np.cos(-0.35), -np.sin(-0.35)], [np.sin(-0.35), np.cos(-0.35)]])

        self.play(ApplyMatrix(m1, plane), ApplyMatrix(m1, circle), run_time=1.0)
        self.play(ApplyMatrix(m2, plane), ApplyMatrix(m2, circle), run_time=1.0)
        self.play(ApplyMatrix(m3, plane), ApplyMatrix(m3, circle), run_time=1.0)
