# 2. Matrices & Matrix Operations

[← Table of Contents](../../README.md)

## Formal definitions

**Definition 2.1 (Matrix as a linear map).** A matrix
$\mathbf{A}\in\mathbb{R}^{m\times n}$ defines a linear map
$T_{\mathbf{A}}:\mathbb{R}^n\to\mathbb{R}^m$ by
$T_{\mathbf{A}}(\mathbf{x})=\mathbf{A}\mathbf{x}$.

**Definition 2.2 (Matrix multiplication as composition).** If
$\mathbf{A}\in\mathbb{R}^{m\times p}$ and
$\mathbf{B}\in\mathbb{R}^{p\times n}$, then
$\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times n}$ is the unique matrix such that

$$
T_{\mathbf{A}\mathbf{B}} = T_{\mathbf{A}}\circ T_{\mathbf{B}}.
$$

Entry-wise,

$$
(\mathbf{A}\mathbf{B})_{ij}=\sum_{k=1}^p A_{ik}B_{kj}.
$$

**Theorem 2.3 (Associativity of matrix multiplication).** For conformable
$\mathbf{A},\mathbf{B},\mathbf{C}$,

$$
(\mathbf{A}\mathbf{B})\mathbf{C}=\mathbf{A}(\mathbf{B}\mathbf{C}).
$$

**Proof.** For any $i,j$,

$$
\begin{aligned}
((\mathbf{A}\mathbf{B})\mathbf{C})_{ij}
&=\sum_{\ell}(\mathbf{A}\mathbf{B})_{i\ell}C_{\ell j}
=\sum_{\ell}\left(\sum_k A_{ik}B_{k\ell}\right)C_{\ell j} \\
&=\sum_k A_{ik}\left(\sum_{\ell}B_{k\ell}C_{\ell j}\right)
=(\mathbf{A}(\mathbf{B}\mathbf{C}))_{ij}.
\end{aligned}
$$

Thus all entries are equal. $\blacksquare$

**Proposition 2.4 (Transpose of a product).**

$$
(\mathbf{A}\mathbf{B})^\top=\mathbf{B}^\top\mathbf{A}^\top.
$$

**Proof.**

$$
((\mathbf{A}\mathbf{B})^\top)_{ij}=(\mathbf{A}\mathbf{B})_{ji}
=\sum_k A_{jk}B_{ki}
=\sum_k (\mathbf{B}^\top)_{ik}(\mathbf{A}^\top)_{kj}
=(\mathbf{B}^\top\mathbf{A}^\top)_{ij}.
$$

$\blacksquare$

**Definition 2.5 (Column space, null space, rank).** For
$\mathbf{A}\in\mathbb{R}^{m\times n}$:
- $\operatorname{Col}(\mathbf{A})=\{\mathbf{A}\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}\subseteq\mathbb{R}^m$.
- $\operatorname{Null}(\mathbf{A})=\{\mathbf{x}\in\mathbb{R}^n:\mathbf{A}\mathbf{x}=\mathbf{0}\}$.
- $\operatorname{rank}(\mathbf{A})=\dim\operatorname{Col}(\mathbf{A})$.

**Theorem 2.6 (Rank–nullity).** If $\mathbf{A}\in\mathbb{R}^{m\times n}$, then

$$
\dim\operatorname{Null}(\mathbf{A})+\operatorname{rank}(\mathbf{A})=n.
$$

**Proof sketch.** Row-reduce $\mathbf{A}$ to RREF. Let $r$ be the number of pivot
columns. Then $r=\operatorname{rank}(\mathbf{A})$. There are $n-r$ free variables
in solutions of $\mathbf{A}\mathbf{x}=\mathbf{0}$, producing a basis of
$\operatorname{Null}(\mathbf{A})$ with $n-r$ vectors. Hence
$\dim\operatorname{Null}(\mathbf{A})=n-r$, so the sum is $n$. $\blacksquare$

**Definition 2.7 (Invertibility).** A square matrix
$\mathbf{A}\in\mathbb{R}^{n\times n}$ is invertible if there exists
$\mathbf{B}$ such that
$\mathbf{A}\mathbf{B}=\mathbf{B}\mathbf{A}=\mathbf{I}$.

**Proposition 2.8 (Uniqueness of inverse).** If $\mathbf{A}$ is invertible, its
inverse is unique.

**Proof.** Suppose $\mathbf{B}$ and $\mathbf{C}$ both satisfy inverse equations.
Then

$$
\mathbf{B}=\mathbf{B}\mathbf{I}=\mathbf{B}(\mathbf{A}\mathbf{C})=(\mathbf{B}\mathbf{A})\mathbf{C}=\mathbf{I}\mathbf{C}=\mathbf{C}.
$$

So $\mathbf{B}=\mathbf{C}$. $\blacksquare$

## Worked example ($2\times 3$ times $3\times 2$)

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

Every transformation inside a transformer is matrix composition. Theorems 2.3
and 2.4 justify matrix reordering/transposition identities used constantly in
attention and backprop.

---

[← Vectors](01-vectors.md) · [Next: Linear Transformations, SVD & PCA →](03-linear-transformations-svd.md)
