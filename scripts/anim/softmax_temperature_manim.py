from manim import *
import numpy as np


def softmax(logits, t):
    z = np.exp(np.array(logits) / t)
    return z / z.sum()


class SoftmaxTemperatureScene(Scene):
    def construct(self):
        logits = [2.0, 1.0, 0.5, 0.1]
        bars = VGroup(*[Rectangle(width=0.7, height=0.1, fill_color=PURPLE, fill_opacity=0.8) for _ in logits])
        bars.arrange(RIGHT, buff=0.2).to_edge(DOWN)
        self.add(bars)

        title = Text("Temperature sweep", font_size=28).to_edge(UP)
        self.add(title)

        for t in np.linspace(0.5, 2.0, 18):
            p = softmax(logits, t)
            anims = [bars[i].animate.stretch_to_fit_height(max(0.1, float(p[i]) * 3)).align_to(bars[i], DOWN) for i in range(4)]
            self.play(*anims, run_time=0.1)
