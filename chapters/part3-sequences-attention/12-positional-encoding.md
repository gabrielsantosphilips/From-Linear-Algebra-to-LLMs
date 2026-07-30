# 12. Positional Encoding

[← Table of Contents](../../README.md)

## The problem

Self-attention is **permutation-invariant**: shuffle the tokens and the math is
unchanged. But “dog bites man” ≠ “man bites dog.” We must inject **position**
information into the embeddings.

## Sinusoidal positional encoding

The original transformer adds a fixed pattern of sines and cosines. For
position $pos$ and dimension index $i$ (in a $d$-dimensional embedding):

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).
$$

These vectors are **added** to the token embeddings:
$\mathbf{x}'_{pos} = \mathbf{e}_{pos} + PE_{pos}$.

## Worked example ($d = 4$)

With $d=4$, the dimension pairs use frequencies
$10000^{0/4}=1$ and $10000^{2/4}=100$.

**Position 0:**
$$
PE_0 = [\sin 0,\ \cos 0,\ \sin 0,\ \cos 0] = [0,\ 1,\ 0,\ 1].
$$

**Position 1:**
$$
\sin(1/1)=0.841,\ \cos(1/1)=0.540,\ \sin(1/100)=0.010,\ \cos(1/100)=1.000.
$$
$$
PE_1 = [0.841,\ 0.540,\ 0.010,\ 1.000].
$$

**Position 2:**
$$
\sin(2)=0.909,\ \cos(2)=-0.416,\ \sin(0.02)=0.020,\ \cos(0.02)=1.000.
$$
$$
PE_2 = [0.909,\ -0.416,\ 0.020,\ 1.000].
$$

Note how low dimensions oscillate quickly (encode fine position) while high
dimensions change slowly (encode coarse position).

## Why sinusoids?

For any fixed offset $k$, $PE_{pos+k}$ is a *linear function* of $PE_{pos}$, so
the model can learn to attend by **relative** position easily. They also
extrapolate to sequence lengths not seen in training.

## Modern alternatives

- **Learned positional embeddings**: a trainable vector per position (used by GPT-2/BERT).
- **RoPE (Rotary Position Embedding)**: rotates query/key vectors by an
  angle proportional to position — dominant in recent LLMs (LLaMA, etc.).
- **ALiBi**: adds a distance-based bias directly to attention scores.

## Intuition for LLMs

Positional encoding is what lets an otherwise order-blind attention mechanism
understand word order, grammar, and sequence structure.

---

[← Multi-Head Attention](11-multi-head-attention.md) · [Next: The Transformer Architecture →](../part4-transformers-llms/13-transformer-architecture.md)
