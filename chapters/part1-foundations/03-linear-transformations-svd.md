# 3. Linear Transformations, Eigenvectors, SVD & PCA

[← Table of Contents](../../README.md)

## Linear transformations

A matrix $\mathbf{A}$ acting on a vector, $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$,
is a **linear transformation**: it can rotate, scale, and shear space while
keeping the origin fixed and lines straight.

## Eigenvectors & eigenvalues

An **eigenvector** $\mathbf{v}$ of $\mathbf{A}$ is a special direction that the
transformation only *scales* (does not rotate):

$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v},
$$

where the scalar $\lambda$ is its **eigenvalue**.

### Worked example ($2\times 2$)

$$
\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.
$$

Solve $\det(\mathbf{A}-\lambda\mathbf{I})=0$:

$$
\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}
= (2-\lambda)^2 - 1 = 0.
$$

So $(2-\lambda)^2 = 1 \Rightarrow 2-\lambda = \pm 1 \Rightarrow \lambda_1 = 3,\ \lambda_2 = 1.$

For $\lambda_1=3$: $(\mathbf{A}-3\mathbf{I})\mathbf{v}=0$ gives $-v_1+v_2=0$, so
$\mathbf{v}_1=\begin{bmatrix}1\\1\end{bmatrix}$.

For $\lambda_2=1$: gives $v_1+v_2=0$, so
$\mathbf{v}_2=\begin{bmatrix}1\\-1\end{bmatrix}$.

**Check:** $\mathbf{A}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}3\\3\end{bmatrix} = 3\begin{bmatrix}1\\1\end{bmatrix}.$ ✓

## Singular Value Decomposition (SVD)

**Any** matrix factorizes as

$$
\mathbf{A} = \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}^\top,
$$

where $\mathbf{U}, \mathbf{V}$ are orthogonal (rotations) and
$\mathbf{\Sigma}$ is diagonal with non-negative **singular values**
$\sigma_1\ge\sigma_2\ge\dots\ge 0$. Intuitively: *rotate → scale → rotate*. Large
singular values capture the dominant structure; small ones can be discarded to
compress the matrix (low-rank approximation).

## Principal Component Analysis (PCA)

PCA finds the directions of **maximum variance** in data — the top eigenvectors
of the covariance matrix (equivalently, top singular vectors). Projecting onto
the first $k$ of them reduces dimensionality while preserving most information.

**Sketch example:** if 3D embeddings mostly lie on a tilted plane, PCA finds
that plane's two axes; projecting onto them gives faithful 2D coordinates.

## Intuition for LLMs

- Embedding spaces are high-dimensional; PCA/SVD let us **visualize** and
  **compress** them.
- Low-rank ideas power efficient fine-tuning methods such as **LoRA**, which
  add a low-rank update $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ to frozen
  weights.
- Eigen-thinking explains how repeated linear maps amplify some directions and
  shrink others — relevant to training stability.

---

[← Matrices](02-matrices.md) · [Next: Probability & Statistics →](04-probability-statistics.md)
