# Notation & Conventions

[← Table of Contents](../README.md)

Before we begin, here is the notation used throughout the booklet. Keep this
page handy.

## Symbols

| Symbol | Meaning |
|---|---|
| $a, b, x$ | Scalars (single numbers), lowercase italic |
| $\mathbf{x}, \mathbf{v}$ | Vectors (bold lowercase) |
| $\mathbf{W}, \mathbf{A}$ | Matrices (bold uppercase) |
| $x_i$ | The $i$-th component (entry) of vector $\mathbf{x}$ |
| $W_{ij}$ | Entry in row $i$, column $j$ of matrix $\mathbf{W}$ |
| $\mathbf{x}^\top$ | Transpose of $\mathbf{x}$ |
| $\mathbb{R}^n$ | The set of real vectors with $n$ components |
| $\mathbb{R}^{m\times n}$ | The set of real $m\times n$ matrices |
| $\|\mathbf{x}\|$ | The (L2/Euclidean) norm of $\mathbf{x}$ |
| $\langle \mathbf{a}, \mathbf{b}\rangle$ or $\mathbf{a}\cdot\mathbf{b}$ | Dot (inner) product |
| $\nabla f$ | Gradient of a function $f$ |
| $\sigma(\cdot)$ | Sigmoid or, in context, an activation function |
| $\text{softmax}(\cdot)$ | The softmax function |

## Conventions

- **Vectors are columns** by default: $\mathbf{x}\in\mathbb{R}^{n}$ is an $n\times 1$ column.
- **Indexing starts at 1** in the math, but we note when code would use 0.
- A **linear layer** is written $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.
- Dimensions are stated explicitly, e.g. $\mathbf{W}\in\mathbb{R}^{m\times n}$.
- $d_{\text{model}}$ is the model (embedding) dimension; $d_k$ the key dimension.

---

[← Table of Contents](../README.md) · [Next: Vectors →](part1-foundations/01-vectors.md)
