# Appendix A — Linear Algebra Cheat Sheet

[← Table of Contents](../../README.md)

## Vectors

| Operation | Formula |
|---|---|
| Dot product | $\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i$ |
| L2 norm | $\lVert\mathbf{x}\rVert = \sqrt{\sum_i x_i^2}$ |
| L1 norm | $\lVert\mathbf{x}\rVert_1 = \sum_i \lvert x_i\rvert$ |
| Cosine similarity | $\dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$ |
| Unit vector | $\hat{\mathbf{x}} = \mathbf{x}/\lVert\mathbf{x}\rVert$ |

## Matrices

| Operation | Formula / Note |
|---|---|
| Multiplication | $C_{ij} = \sum_k A_{ik}B_{kj}$ (inner dims must match) |
| Transpose | $(\mathbf{A}^\top)_{ij} = A_{ji}$ |
| Identity | $\mathbf{I}\mathbf{x} = \mathbf{x}$ |
| Inverse | $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ (square, full rank) |
| Linear layer | $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ |
| $(\mathbf{AB})^\top$ | $= \mathbf{B}^\top\mathbf{A}^\top$ |

## Decompositions

| Name | Formula |
|---|---|
| Eigen | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ |
| SVD | $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ |
| Char. equation | $\det(\mathbf{A}-\lambda\mathbf{I}) = 0$ |
| $2\times2$ determinant | $\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad-bc$ |

## Probability & calculus

| Name | Formula |
|---|---|
| Softmax | $\text{softmax}(\mathbf{z})_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Cross-entropy | $-\sum_i y_i \log p_i$ |
| Sigmoid | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ |
| ReLU | $\max(0, z)$ |
| Chain rule | $\dfrac{dz}{dx} = \dfrac{dz}{dg}\dfrac{dg}{dx}$ |
| GD update | $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla L$ |

## Attention (one-liner)

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

---

[← Scaling & Emergence](../part4-transformers-llms/16-scaling-emergence.md) · [Next: Glossary →](B-glossary.md)
