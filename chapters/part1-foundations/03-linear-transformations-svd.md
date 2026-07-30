# 3. Linear Transformations, Eigenvectors, SVD & PCA

[← Table of Contents](../../README.md)

## Linear transformations

**Definition 3.1 (Linear transformation).** A map
$T:\mathbb{R}^n\to\mathbb{R}^m$ is linear if for all
$\mathbf{x},\mathbf{y}\in\mathbb{R}^n$ and $a,b\in\mathbb{R}$,

$$
T(a\mathbf{x}+b\mathbf{y})=aT(\mathbf{x})+bT(\mathbf{y}).
$$

**Theorem 3.2 (Matrix representation theorem).** Every linear map
$T:\mathbb{R}^n\to\mathbb{R}^m$ corresponds to a unique matrix
$\mathbf{A}\in\mathbb{R}^{m\times n}$ such that
$T(\mathbf{x})=\mathbf{A}\mathbf{x}$ for all $\mathbf{x}$.

**Proof.** Let $\mathbf{e}_1,\dots,\mathbf{e}_n$ be the standard basis of
$\mathbb{R}^n$. Define the $i$-th column of $\mathbf{A}$ as $T(\mathbf{e}_i)$.
For $\mathbf{x}=\sum_i x_i\mathbf{e}_i$,

$$
T(\mathbf{x})=\sum_i x_iT(\mathbf{e}_i)=\sum_i x_i\mathbf{a}_i=\mathbf{A}\mathbf{x}.
$$

Uniqueness: if $\mathbf{A}\mathbf{x}=\mathbf{B}\mathbf{x}$ for all $\mathbf{x}$,
then in particular on each $\mathbf{e}_i$ the columns match, so $\mathbf{A}=\mathbf{B}$.
$\blacksquare$

## Eigenvalues and eigenvectors

**Definition 3.3 (Eigenpair).** For square $\mathbf{A}$, a nonzero
$\mathbf{v}$ and scalar $\lambda$ satisfy

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
$$

iff $(\lambda,\mathbf{v})$ is an eigenvalue/eigenvector pair.

**Proposition 3.4 (Characteristic polynomial).** Eigenvalues are exactly roots of

$$
p_{\mathbf{A}}(\lambda)=\det(\mathbf{A}-\lambda\mathbf{I}).
$$

**Proof.** $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ with $\mathbf{v}\neq\mathbf{0}$ iff
$(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=\mathbf{0}$ has nontrivial solution iff
$\mathbf{A}-\lambda\mathbf{I}$ is singular iff
$\det(\mathbf{A}-\lambda\mathbf{I})=0$. $\blacksquare$

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

## Spectral theorem for symmetric matrices

**Lemma 3.5 (Real eigenvalues).** If $\mathbf{A}=\mathbf{A}^\top\in\mathbb{R}^{n\times n}$,
all eigenvalues are real.

**Proof.** Let $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ with complex
$\mathbf{v}\neq 0$. Then

$$
\lambda\,\mathbf{v}^*\mathbf{v}=\mathbf{v}^*\mathbf{A}\mathbf{v}.
$$

Taking conjugate transpose and using $\mathbf{A}^\top=\mathbf{A}$ (hence
$\mathbf{A}^*=\mathbf{A}$) gives

$$
\overline{\lambda}\,\mathbf{v}^*\mathbf{v}=\mathbf{v}^*\mathbf{A}\mathbf{v}.
$$

So $(\lambda-\overline{\lambda})\mathbf{v}^*\mathbf{v}=0$. Since
$\mathbf{v}^*\mathbf{v}>0$, $\lambda=\overline{\lambda}$ is real. $\blacksquare$

**Lemma 3.6 (Orthogonality of distinct eigenvectors).** If
$\mathbf{A}=\mathbf{A}^\top$ and
$\mathbf{A}\mathbf{u}=\lambda\mathbf{u}$,
$\mathbf{A}\mathbf{v}=\mu\mathbf{v}$ with $\lambda\neq\mu$, then
$\mathbf{u}^\top\mathbf{v}=0$.

**Proof.**

$$
\lambda\,\mathbf{u}^\top\mathbf{v}=(\mathbf{A}\mathbf{u})^\top\mathbf{v}
=\mathbf{u}^\top\mathbf{A}^\top\mathbf{v}=\mathbf{u}^\top\mathbf{A}\mathbf{v}
=\mu\,\mathbf{u}^\top\mathbf{v}.
$$

Hence $(\lambda-\mu)\mathbf{u}^\top\mathbf{v}=0$, so $\mathbf{u}^\top\mathbf{v}=0$.
$\blacksquare$

**Theorem 3.7 (Spectral theorem, real symmetric case).** If
$\mathbf{A}=\mathbf{A}^\top\in\mathbb{R}^{n\times n}$, then there exists an
orthogonal matrix $\mathbf{Q}$ and diagonal $\mathbf{\Lambda}$ such that

$$
\mathbf{A}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top.
$$

All diagonal entries of $\mathbf{\Lambda}$ are real eigenvalues of $\mathbf{A}$.

**Proof sketch.** By the fundamental theorem of algebra, $\mathbf{A}$ has a complex
eigenvalue; by Lemma 3.5 it is real, so there exists real eigenvector
$\mathbf{q}_1$. Normalize $\mathbf{q}_1$. Consider the orthogonal complement
$\mathbf{q}_1^\perp$, which is invariant under $\mathbf{A}$ by symmetry. Restrict
$\mathbf{A}$ to this subspace and iterate inductively to get an orthonormal basis
of eigenvectors. Stack them as columns of $\mathbf{Q}$, giving
$\mathbf{Q}^\top\mathbf{A}\mathbf{Q}=\mathbf{\Lambda}$ diagonal. Rearranging yields
$\mathbf{A}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$. $\blacksquare$

## SVD and low-rank approximation

**Theorem 3.8 (Existence and essential uniqueness of SVD).** For any
$\mathbf{A}\in\mathbb{R}^{m\times n}$, there exist orthogonal
$\mathbf{U}\in\mathbb{R}^{m\times m}$, $\mathbf{V}\in\mathbb{R}^{n\times n}$ and a
diagonal rectangular matrix $\mathbf{\Sigma}$ with singular values
$\sigma_1\ge\cdots\ge\sigma_r>0$ such that

$$
\mathbf{A}=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top.
$$

Singular values are unique; singular vectors are unique up to signs (and
orthogonal rotations within equal-singular-value subspaces).

**Proof sketch.** $\mathbf{A}^\top\mathbf{A}$ is symmetric positive semidefinite,
so by Theorem 3.7 it diagonalizes:
$\mathbf{A}^\top\mathbf{A}=\mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$ with
$\Lambda_{ii}=\sigma_i^2\ge 0$. For $\sigma_i>0$, define
$\mathbf{u}_i=(1/\sigma_i)\mathbf{A}\mathbf{v}_i$. Complete to orthonormal bases
for $\mathbf{U},\mathbf{V}$. Then
$\mathbf{A}=\sum_i \sigma_i\mathbf{u}_i\mathbf{v}_i^\top=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$.
$\blacksquare$

**Theorem 3.9 (Eckart–Young).** Let
$\mathbf{A}_k=\sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. Among all rank-$k$ matrices
$\mathbf{B}$,

$$
\|\mathbf{A}-\mathbf{B}\|_F \ge \|\mathbf{A}-\mathbf{A}_k\|_F,
$$

and similarly for operator norm. So truncated SVD is the best low-rank
approximation.

## PCA and Rayleigh quotient

Let centered data vectors be $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(N)}\in\mathbb{R}^d$.
Define covariance

$$
\mathbf{S}=\frac{1}{N}\sum_{i=1}^N \mathbf{x}^{(i)}(\mathbf{x}^{(i)})^\top.
$$

**Proposition 3.10 (Variance along a direction).** For unit vector
$\mathbf{u}$, variance of projected data $\mathbf{u}^\top\mathbf{x}$ equals

$$
\mathbf{u}^\top\mathbf{S}\mathbf{u}.
$$

**Proof.** With centered data, empirical variance is

$$
\frac{1}{N}\sum_{i=1}^N (\mathbf{u}^\top\mathbf{x}^{(i)})^2
=\frac{1}{N}\sum_{i=1}^N \mathbf{u}^\top\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^\top\mathbf{u}
=\mathbf{u}^\top\mathbf{S}\mathbf{u}.
$$

$\blacksquare$

**Theorem 3.11 (First principal component maximizes variance).** The solution of

$$
\max_{\|\mathbf{u}\|=1}\mathbf{u}^\top\mathbf{S}\mathbf{u}
$$

is any unit eigenvector of $\mathbf{S}$ associated to its largest eigenvalue.

**Proof.** By Theorem 3.7,
$\mathbf{S}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$, $\lambda_1\ge\cdots\ge\lambda_d\ge0$.
Write $\mathbf{u}=\mathbf{Q}\mathbf{y}$ with $\|\mathbf{y}\|=1$. Then

$$
\mathbf{u}^\top\mathbf{S}\mathbf{u}=\mathbf{y}^\top\mathbf{\Lambda}\mathbf{y}
=\sum_{i=1}^d \lambda_i y_i^2 \le \lambda_1\sum_i y_i^2=\lambda_1,
$$

with equality when $\mathbf{y}=\mathbf{e}_1$. So maximizers are top
eigenvectors. $\blacksquare$

## Intuition for LLMs

PCA rigorously reduces to the spectral theorem; SVD extends the same idea to
nonsquare matrices. These are the mathematical foundations of low-rank methods
such as LoRA and of embedding-space analysis.

---

[← Matrices](02-matrices.md) · [Next: Probability & Statistics →](04-probability-statistics.md)
