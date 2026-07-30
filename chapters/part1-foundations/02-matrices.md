# 2. Matrices & Matrix Operations

[← Table of Contents](../../README.md)

## Definition

A **matrix** is a rectangular grid of numbers with $m$ rows and $n$ columns,
$\mathbf{W}\in\mathbb{R}^{m\times n}$. In neural networks, matrices are the
**learned parameters (weights)** that transform vectors.

## Matrix multiplication

If $\mathbf{A}\in\mathbb{R}^{m\times p}$ and $\mathbf{B}\in\mathbb{R}^{p\times n}$,
their product $\mathbf{C}=\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times n}$ has entries

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}.
$$

The inner dimensions must match ($p=p$).

### Worked example ($2\times 3$ times $3\times 2$)

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \qquad
\mathbf{B} = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}.
$$

Element by element:

$$
C_{11} = 1\cdot 7 + 2\cdot 9 + 3\cdot 11 = 7+18+33 = 58
$$
$$
C_{12} = 1\cdot 8 + 2\cdot 10 + 3\cdot 12 = 8+20+36 = 64
$$
$$
C_{21} = 4\cdot 7 + 5\cdot 9 + 6\cdot 11 = 28+45+66 = 139
$$
$$
C_{22} = 4\cdot 8 + 5\cdot 10 + 6\cdot 12 = 32+50+72 = 154
$$

$$
\mathbf{C} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}.
$$

## Other operations

- **Transpose:** $(\mathbf{A}^\top)_{ij} = A_{ji}$ (flip across the diagonal).
- **Identity** $\mathbf{I}$: ones on the diagonal, zeros elsewhere; $\mathbf{I}\mathbf{x}=\mathbf{x}$.
- **Inverse** $\mathbf{A}^{-1}$: satisfies $\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}$ (only for square, full-rank matrices).
- **Rank:** number of linearly independent rows/columns — how much "information" the matrix carries.

## The linear layer

The fundamental building block of neural networks:

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}.
$$

### Worked example

With
$$
\mathbf{W} = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix},\;
\mathbf{x} = \begin{bmatrix} 3 \\ 5 \end{bmatrix},\;
\mathbf{b} = \begin{bmatrix} 1 \\ -2 \end{bmatrix}:
$$

$$
\mathbf{W}\mathbf{x} = \begin{bmatrix} 1\cdot 3 + (-1)\cdot 5 \\ 0\cdot 3 + 2\cdot 5 \end{bmatrix} = \begin{bmatrix} -2 \\ 10 \end{bmatrix},
\quad
\mathbf{y} = \begin{bmatrix} -2 \\ 10 \end{bmatrix} + \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} -1 \\ 8 \end{bmatrix}.
$$

## Intuition for LLMs

Every transformation inside a transformer — projecting embeddings into
queries/keys/values, the feed-forward layers, the final vocabulary projection —
is matrix multiplication. Understanding $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$
means you understand ~80% of the arithmetic in an LLM.

---

[← Vectors](01-vectors.md) · [Next: Linear Transformations, SVD & PCA →](03-linear-transformations-svd.md)
