# 8. Loss Functions & Gradient Descent

[← Table of Contents](../../README.md)

## Loss functions

A **loss** measures how wrong the model is.

$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2,
\qquad
L_{\text{CE}} = -\sum_i y_i \log p_i.
$$

## Gradient descent

$$
\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\nabla f(\mathbf{w}_t).
$$

**Definition 8.1 ($L$-smoothness).** $f$ is $L$-smooth if
$\nabla f$ is Lipschitz:
$$
\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|\le L\|\mathbf{x}-\mathbf{y}\|.
$$

**Lemma 8.2 (Descent lemma).** If $f$ is differentiable and $L$-smooth, then
$$
f(\mathbf{y})\le f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})+
\frac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2.
$$

**Proof sketch.** Define $\phi(t)=f(\mathbf{x}+t(\mathbf{y}-\mathbf{x}))$. Integrate
$\phi'(t)$ and use Lipschitz continuity of $\nabla f$ along the segment.
$\blacksquare$

**Theorem 8.3 (Monotone decrease for $\eta\le 1/L$).** If $f$ is $L$-smooth and
$\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\nabla f(\mathbf{w}_t)$ with
$0<\eta\le 1/L$, then
$$
f(\mathbf{w}_{t+1})\le f(\mathbf{w}_t)-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla f(\mathbf{w}_t)\|^2
\le f(\mathbf{w}_t).
$$

**Proof.** Apply Lemma 8.2 with
$\mathbf{x}=\mathbf{w}_t$,
$\mathbf{y}=\mathbf{w}_t-\eta\nabla f(\mathbf{w}_t)$. $\blacksquare$

**Definition 8.4 ($\mu$-strong convexity).** $f$ is $\mu$-strongly convex if
$$
f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})+
\frac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2.
$$

**Theorem 8.5 (Linear convergence under strong convexity).** If $f$ is
$L$-smooth and $\mu$-strongly convex, and $0<\eta\le 1/L$, then with optimizer
$\mathbf{w}_\star$,
$$
\|\mathbf{w}_{t}-\mathbf{w}_\star\|^2 \le (1-\eta\mu)^t\|\mathbf{w}_0-\mathbf{w}_\star\|^2,
$$
and therefore function values also converge geometrically.

**Proof sketch.** Combine cocoercivity/strong monotonicity inequalities for
$\nabla f$ with the GD update expansion of
$\|\mathbf{w}_{t+1}-\mathbf{w}_\star\|^2$. $\blacksquare$

## Full worked example (two GD steps)

Fit $\hat{y} = w x$ to one data point $(x, y) = (2, 6)$ (ideal $w=3$). Loss
$L(w) = (wx - y)^2$. Derivative:
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
Loss: $(2.4\cdot 2 - 6)^2 = 1.44$.

**Step 2:**
$$
\hat{y} = 2.4\cdot 2 = 4.8,\quad \frac{dL}{dw} = 2(4.8-6)(2) = -4.8.
$$
$$
w_2 = 2.4 - 0.1(-4.8) = 2.88.
$$
Loss: $(2.88\cdot 2 - 6)^2 = 0.0576$.

## Intuition for LLMs

Chapter 5 gave the gradients; this chapter adds guarantees that updates
actually reduce loss (Theorem 8.3), and explains why strongly convex toy models
converge rapidly.

---

[← Activations](07-activations-nonlinearity.md) · [Next: Embeddings →](../part3-sequences-attention/09-embeddings.md)
