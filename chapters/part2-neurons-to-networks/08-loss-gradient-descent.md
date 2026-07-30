# 8. Loss Functions & Gradient Descent

[← Table of Contents](../../README.md)

## Loss functions

A **loss** measures how wrong the model is. Two common ones:

**Mean Squared Error (regression):**
$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2.
$$

**Cross-entropy (classification / next-token):** see [Chapter 4](../part1-foundations/04-probability-statistics.md).
$$
L_{\text{CE}} = -\sum_i y_i \log p_i.
$$

## Gradient descent

To minimize the loss, repeatedly step **against** the gradient:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta\, \nabla_\mathbf{w} L,
$$

where $\eta$ is the **learning rate**.

## Full worked example (two GD steps)

Fit $\hat{y} = w x$ to one data point $(x, y) = (2, 6)$ (so the ideal $w=3$).
Loss $L(w) = (\hat{y}-y)^2 = (wx - y)^2$. Its derivative:

$$
\frac{dL}{dw} = 2(wx - y)\cdot x.
$$

Start at $w_0 = 0$, learning rate $\eta = 0.1$.

**Step 1:**
$$
\hat{y} = 0\cdot 2 = 0,\quad \frac{dL}{dw} = 2(0-6)(2) = -24.
$$
$$
w_1 = 0 - 0.1(-24) = 2.4.
$$
Loss now: $(2.4\cdot 2 - 6)^2 = (4.8-6)^2 = 1.44$ (was $36$).

**Step 2:**
$$
\hat{y} = 2.4\cdot 2 = 4.8,\quad \frac{dL}{dw} = 2(4.8-6)(2) = -4.8.
$$
$$
w_2 = 2.4 - 0.1(-4.8) = 2.88.
$$
Loss now: $(2.88\cdot 2 - 6)^2 = (5.76-6)^2 = 0.0576.$

The weight marches toward the true value $w=3$, and the loss shrinks
$36 \to 1.44 \to 0.0576$. That is learning, made concrete.

## Variants used in practice

- **SGD**: gradient from a *mini-batch* instead of the whole dataset.
- **Momentum**: accumulate past gradients to accelerate.
- **Adam**: adaptive per-parameter learning rates — the default optimizer for LLMs.

## Intuition for LLMs

An LLM is trained by running gradient descent (with Adam) over cross-entropy
loss across trillions of tokens. The tiny example above is *exactly* what
happens — just with billions of weights instead of one.

---

[← Activations](07-activations-nonlinearity.md) · [Next: Embeddings →](../part3-sequences-attention/09-embeddings.md)
