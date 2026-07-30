# 13. The Transformer Architecture

[← Table of Contents](../../README.md)

We now assemble everything from Parts I–III into the full transformer.

## The big picture (decoder-only / GPT style)

```
         token IDs
            ↓
     [ Embedding lookup ]
            +
     [ Positional encoding ]
            ↓
  ┌──────── Transformer block (× N) ──────┐
  │   Multi-Head Self-Attention            │
  │   + residual, LayerNorm                │
  │   Feed-Forward Network                 │
  │   + residual, LayerNorm                │
  └────────────────────────────────┘
            ↓
     [ Final LayerNorm ]
            ↓
     [ Linear → vocabulary logits ]
            ↓
     [ softmax → next-token probs ]
```

## Layer normalization

**Definition 13.1 (LayerNorm).** For
$\mathbf{x}=(x_1,\dots,x_d)\in\mathbb{R}^d$,
$$
\mu(\mathbf{x})=\frac{1}{d}\sum_{i=1}^d x_i,
\qquad
\sigma^2(\mathbf{x})=\frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2,
$$
$$
\operatorname{LN}(\mathbf{x})_i=\gamma_i\frac{x_i-\mu(\mathbf{x})}{\sqrt{\sigma^2(\mathbf{x})+\epsilon}}+\beta_i.
$$

**Proposition 13.2 (Shift/scale invariance of normalized core).** Define
$N(\mathbf{x})=(\mathbf{x}-\mu(\mathbf{x})\mathbf{1})/\sigma(\mathbf{x})$ for
$\sigma(\mathbf{x})>0$. For any $a>0,b\in\mathbb{R}$,
$$
N(a\mathbf{x}+b\mathbf{1})=N(\mathbf{x}).
$$

**Proof.** $\mu(a\mathbf{x}+b\mathbf{1})=a\mu(\mathbf{x})+b$ and
$\sigma(a\mathbf{x}+b\mathbf{1})=a\sigma(\mathbf{x})$ for $a>0$, so
$$
\frac{a\mathbf{x}+b\mathbf{1}-(a\mu+b)\mathbf{1}}{a\sigma}=\frac{\mathbf{x}-\mu\mathbf{1}}{\sigma}.
$$
$\blacksquare$

### Worked example

$\mathbf{x}=[2,4,6]$. Mean $\mu=4$, variance $\sigma^2=8/3$, standard deviation
$\sigma\approx1.633$:
$$
\hat{\mathbf{x}}=[-1.225,0,1.225].
$$

## Residual connections and gradient flow

A residual block is
$$
F(\mathbf{x})=\mathbf{x}+G(\mathbf{x}).
$$
Its Jacobian is
$$
J_F(\mathbf{x})=\mathbf{I}+J_G(\mathbf{x}).
$$
So gradients backpropagate through an identity path even when $J_G$ is small,
mitigating vanishing gradients.

## Feed-forward network (FFN)

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2\,\text{GELU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2.
$$

## Intuition for LLMs

LayerNorm stabilizes statistics; residuals preserve trainable gradient flow.
Together they make deep transformer stacks practical.

---

[← Positional Encoding](../part3-sequences-attention/12-positional-encoding.md) · [Next: The Training Objective →](14-training-objective.md)
