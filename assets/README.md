# Figures and animations

Generated media lives in:

- `assets/img/` — static SVG figures (matplotlib)
- `assets/anim/` — animated GIFs

Generation scripts:

- `scripts/generate_static_figures.py`
- `scripts/anim/generate_gifs.py`
- `scripts/anim/*_manim.py` (manim scene sources for key concepts)

## Regenerate locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_static_figures.py
python scripts/anim/generate_gifs.py
```

Optional manim renders (MP4):

```bash
manim -pql scripts/anim/gradient_descent_manim.py GradientDescentScene
manim -pql scripts/anim/self_attention_manim.py SelfAttentionScene
manim -pql scripts/anim/linear_transform_manim.py LinearTransformSVDScene
manim -pql scripts/anim/softmax_temperature_manim.py SoftmaxTemperatureScene
```
