from manim import *


class SelfAttentionScene(Scene):
    def construct(self):
        query = Dot(LEFT * 4, color=BLUE)
        keys = VGroup(*[Dot(RIGHT * 2 + UP * y, color=ORANGE) for y in [2, 0, -2]])
        weights = [0.2, 0.55, 0.25]
        self.add(query, keys)

        lines = VGroup()
        for k, w in zip(keys, weights):
            line = Line(query.get_center(), k.get_center(), stroke_width=2 + 6 * w, color=RED)
            lines.add(line)
        self.play(Create(lines), run_time=1.5)

        output = Dot(RIGHT * 5, color=GREEN)
        self.add(output)
        self.play(TransformFromCopy(lines, Line(query.get_center(), output.get_center(), color=GREEN, stroke_width=5)), run_time=1.2)
