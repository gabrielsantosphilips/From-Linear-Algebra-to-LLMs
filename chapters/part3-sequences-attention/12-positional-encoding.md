# 12. Positional Encoding

[← Table of Contents](../../README.md)

## The problem

Self-attention is permutation-equivariant (see
[Proposition 10.1](10-self-attention.md)), so token order must be injected.

## Sinusoidal positional encoding

For position $pos$ and pair index $i$:

$$
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).
$$

**Proposition 12.1 (Relative shift is linear rotation).** For fixed $i$, let
$\omega_i=10000^{-2i/d}$ and

$$
\mathbf{u}_{pos}^{(i)}=
\begin{bmatrix}
\sin(\omega_i pos) \\
\cos(\omega_i pos)
\end{bmatrix}.
$$

Then for any integer offset $k$,

$$
\mathbf{u}_{pos+k}^{(i)}=\mathbf{R}_i(k)\,\mathbf{u}_{pos}^{(i)},
$$

where

$$
\mathbf{R}_i(k)=
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}.
$$

**Proof.** Using angle-addition formulas,

$$
\sin(\omega_i(pos+k))=\sin(\omega_i pos)\cos(\omega_i k)+\cos(\omega_i pos)\sin(\omega_i k),
$$

$$
\cos(\omega_i(pos+k))=\cos(\omega_i pos)\cos(\omega_i k)-\sin(\omega_i pos)\sin(\omega_i k).
$$

This is exactly the stated matrix multiplication. $\blacksquare$

So each sine/cosine pair advances by a fixed linear transform, enabling
relative-position reasoning.

## Worked example ($d = 4$)

With $d=4$, frequencies are $1$ and $1/100$.

$$
PE_0 = [0,1,0,1],
$$

$$
PE_1 = [0.841,0.540,0.010,1.000],
$$

$$
PE_2 = [0.909,-0.416,0.020,1.000].
$$

## Modern alternatives

- Learned positional embeddings
- RoPE
- ALiBi

## Intuition for LLMs

The rotation identity in Proposition 12.1 is the key reason sinusoidal features
encode relative shifts cleanly.

---

[← Multi-Head Attention](11-multi-head-attention.md) · [Next: The Transformer Architecture →](../part4-transformers-llms/13-transformer-architecture.md)
