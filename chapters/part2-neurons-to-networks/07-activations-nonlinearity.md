# 7. Activation Functions & Nonlinearity

[← Table of Contents](../../README.md)

## Why nonlinearity?

Stacking linear layers alone is pointless: a composition of linear maps is
*still linear* ($\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}$).
**Nonlinear activation functions** between layers let networks approximate any
function (the *universal approximation theorem*).

## The main activation functions

### Sigmoid
$$
\sigma(z) = \frac{1}{1+e^{-z}} \in (0,1)
$$
Smooth, squashes to $(0,1)$. Suffers from *vanishing gradients* for large $|z|$.

### Tanh
$$
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \in (-1,1)
$$
Zero-centered version of sigmoid.

### ReLU (Rectified Linear Unit)
$$
\text{ReLU}(z) = \max(0, z)
$$
Cheap, non-saturating for $z>0$; the default in many deep nets.

### GELU (Gaussian Error Linear Unit)
$$
\text{GELU}(z) = z\,\Phi(z) \approx 0.5\,z\left(1+\tanh\!\big[\sqrt{2/\pi}(z+0.044715 z^3)\big]\right)
$$
A smooth ReLU-like curve used in GPT/BERT-style transformers.

## Worked numeric evaluations at $z = -1,\ 0,\ 2$

| $z$ | $\sigma(z)$ | $\tanh(z)$ | $\text{ReLU}(z)$ | $\text{GELU}(z)$ |
|----|----------|---------|---------------|---------------|
| $-1$ | $0.269$ | $-0.762$ | $0$ | $-0.159$ |
| $0$  | $0.500$ | $0.000$  | $0$ | $0.000$ |
| $2$  | $0.881$ | $0.964$  | $2$ | $1.954$ |

**Sample calculation** ($\sigma(2)$):
$$
\sigma(2) = \frac{1}{1+e^{-2}} = \frac{1}{1+0.135} = \frac{1}{1.135} = 0.881.
$$

## Intuition for LLMs

Modern transformers use **GELU** (or variants like SwiGLU) in their
feed-forward blocks. Without these nonlinearities, an LLM — no matter how many
layers — would collapse into a single linear map and could not model the rich
structure of language.

---

[← The Perceptron](06-perceptron-linear-models.md) · [Next: Loss & Gradient Descent →](08-loss-gradient-descent.md)
