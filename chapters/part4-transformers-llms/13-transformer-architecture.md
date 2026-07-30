# 13. The Transformer Architecture

[← Table of Contents](../../README.md)

We now assemble everything from Parts I–III into the full transformer.

## The big picture (decoder-only / GPT style)

```
         token IDs
            ↓
     [ Embedding lookup ]  (Ch. 9)
            +
     [ Positional encoding ]  (Ch. 12)
            ↓
  ┌──────── Transformer block (× N) ──────┐
  │   Multi-Head Self-Attention (Ch. 10–11)  │
  │   + residual, LayerNorm                  │
  │   Feed-Forward Network                   │
  │   + residual, LayerNorm                  │
  └────────────────────────────────┘
            ↓
     [ Final LayerNorm ]
            ↓
     [ Linear → vocabulary logits ]
            ↓
     [ softmax → next-token probs ]  (Ch. 4)
```

## Residual connections

Each sub-layer adds its input back to its output:

$$
\mathbf{x} \leftarrow \mathbf{x} + \text{SubLayer}(\mathbf{x}).
$$

This creates a “gradient highway” so very deep stacks (dozens of blocks) still
train (gradients don't vanish — recall [Chapter 5](../part1-foundations/05-calculus-gradients.md)).

## Layer normalization

Normalizes a vector to zero mean and unit variance, then rescales:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma\,\frac{\mathbf{x}-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
$$

where $\mu,\sigma^2$ are the mean/variance across features, and
$\gamma,\beta$ are learned.

### Worked example

$\mathbf{x} = [2, 4, 6]$. Mean $\mu = 4$. Variance
$\sigma^2 = \frac{(2-4)^2+(4-4)^2+(6-4)^2}{3} = \frac{4+0+4}{3} = 2.667$,
so $\sigma = 1.633$. With $\epsilon\approx 0$, $\gamma=1,\beta=0$:

$$
\hat{\mathbf{x}} = \left[\tfrac{2-4}{1.633},\ \tfrac{4-4}{1.633},\ \tfrac{6-4}{1.633}\right] = [-1.225,\ 0,\ 1.225].
$$

## Feed-forward network (FFN)

Applied to each position independently — two linear layers with a nonlinearity
(GELU, [Chapter 7](../part2-neurons-to-networks/07-activations-nonlinearity.md)):

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2\,\text{GELU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2.
$$

Typically the hidden size is $4\times d_{\text{model}}$ (e.g. $512 \to 2048 \to 512$).

## Encoder vs. decoder-only

- **Encoder-only** (BERT): bidirectional attention; good for understanding/classification.
- **Decoder-only** (GPT): causal (masked) attention; generates text left-to-right. **This is what most LLMs are.**
- **Encoder–decoder** (T5, original Transformer): translation-style tasks.

## Intuition for LLMs

A modern LLM is just this block repeated $N$ times (GPT-3: 96 blocks), with
millions of tokens flowing through. Attention mixes information *across*
positions; the FFN processes each position; residual + LayerNorm keep it
trainable. Simple parts, stacked deep, at massive scale.

---

[← Positional Encoding](../part3-sequences-attention/12-positional-encoding.md) · [Next: The Training Objective →](14-training-objective.md)
