# 1. Vectors & Vector Spaces

[← Table of Contents](../../README.md)

## Formal foundations

**Definition 1.1 (Vector space over a field).** Let $\mathbb{F}$ be a field. A set
$V$ with operations $+:V\times V\to V$ and scalar multiplication
$\mathbb{F}\times V\to V$ is a vector space if for all
$\mathbf{u},\mathbf{v},\mathbf{w}\in V$ and $a,b\in\mathbb{F}$:

1. $\mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}$ (commutativity)
2. $(\mathbf{u}+\mathbf{v})+\mathbf{w}=\mathbf{u}+(\mathbf{v}+\mathbf{w})$ (associativity)
3. $\exists\,\mathbf{0}\in V$ such that $\mathbf{u}+\mathbf{0}=\mathbf{u}$ (additive identity)
4. $\forall\,\mathbf{u}\,\exists\,(-\mathbf{u})$ with $\mathbf{u}+(-\mathbf{u})=\mathbf{0}$ (additive inverse)
5. $a(\mathbf{u}+\mathbf{v})=a\mathbf{u}+a\mathbf{v}$ (distributive over vector addition)
6. $(a+b)\mathbf{u}=a\mathbf{u}+b\mathbf{u}$ (distributive over scalar addition)
7. $a(b\mathbf{u})=(ab)\mathbf{u}$ (compatibility)
8. $1\mathbf{u}=\mathbf{u}$ (scalar identity)

$\mathbb{R}^n$ with componentwise addition/scalar multiplication is the canonical
example used throughout this booklet.

**Definition 1.2 (Span).** For vectors $\mathbf{v}_1,\dots,\mathbf{v}_k\in V$,

$$
\operatorname{span}\{\mathbf{v}_1,\dots,\mathbf{v}_k\}=
\left\{\sum_{i=1}^k a_i\mathbf{v}_i : a_i\in\mathbb{F}\right\}.
$$

**Definition 1.3 (Linear independence).** $\mathbf{v}_1,\dots,\mathbf{v}_k$ are
linearly independent if

$$
\sum_{i=1}^k a_i\mathbf{v}_i=\mathbf{0} \Rightarrow a_1=\cdots=a_k=0.
$$

**Definition 1.4 (Basis and dimension).** A basis of $V$ is a linearly independent
set that spans $V$. The number of elements in any basis is the dimension,
written $\dim V$.

## Dot product, norm, and angle

In $\mathbb{R}^n$:

$$
\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^n a_i b_i,
\qquad
\|\mathbf{x}\|_2=\sqrt{\mathbf{x}\cdot\mathbf{x}}.
$$

**Proposition 1.5 (Dot product is an inner product on $\mathbb{R}^n$).**
$\langle \mathbf{a},\mathbf{b}\rangle:=\mathbf{a}\cdot\mathbf{b}$ is bilinear,
symmetric, and positive-definite.

**Proof.** Bilinearity follows from distributivity of real numbers:
$(\mathbf{a}+\mathbf{b})\cdot\mathbf{c}=\mathbf{a}\cdot\mathbf{c}+\mathbf{b}\cdot\mathbf{c}$ and
$(\alpha\mathbf{a})\cdot\mathbf{b}=\alpha(\mathbf{a}\cdot\mathbf{b})$; similarly in the second argument.
Symmetry is immediate: $\sum_i a_i b_i=\sum_i b_i a_i$.
Positive-definiteness: $\mathbf{x}\cdot\mathbf{x}=\sum_i x_i^2\ge 0$, and it is $0$
iff each $x_i=0$, i.e. $\mathbf{x}=\mathbf{0}$. $\blacksquare$

**Theorem 1.6 (Cauchy–Schwarz).** For all $\mathbf{a},\mathbf{b}\in\mathbb{R}^n$,

$$
|\mathbf{a}\cdot\mathbf{b}|\le \|\mathbf{a}\|\,\|\mathbf{b}\|.
$$

**Proof.** If $\mathbf{b}=\mathbf{0}$ it is trivial. Otherwise consider
$f(t)=\|\mathbf{a}-t\mathbf{b}\|^2\ge 0$ for all $t\in\mathbb{R}$:

$$
f(t)=\|\mathbf{a}\|^2-2t(\mathbf{a}\cdot\mathbf{b})+t^2\|\mathbf{b}\|^2.
$$

A quadratic nonnegative for all $t$ has discriminant $\le 0$:

$$
(-2\mathbf{a}\cdot\mathbf{b})^2-4\|\mathbf{b}\|^2\|\mathbf{a}\|^2\le 0
\Rightarrow
(\mathbf{a}\cdot\mathbf{b})^2\le \|\mathbf{a}\|^2\|\mathbf{b}\|^2.
$$

Take square roots. $\blacksquare$

**Corollary 1.7 (Triangle inequality).** For all $\mathbf{a},\mathbf{b}\in\mathbb{R}^n$,

$$
\|\mathbf{a}+\mathbf{b}\|\le \|\mathbf{a}\|+\|\mathbf{b}\|.
$$

**Proof.** Expand and apply Theorem 1.6:

$$
\|\mathbf{a}+\mathbf{b}\|^2
=\|\mathbf{a}\|^2+2\mathbf{a}\cdot\mathbf{b}+\|\mathbf{b}\|^2
\le \|\mathbf{a}\|^2+2\|\mathbf{a}\|\|\mathbf{b}\|+\|\mathbf{b}\|^2
=(\|\mathbf{a}\|+\|\mathbf{b}\|)^2.
$$

Take square roots. $\blacksquare$

**Proposition 1.8 (Dot product and angle).** For nonzero
$\mathbf{a},\mathbf{b}\in\mathbb{R}^n$, define

$$
\theta = \arccos\left(\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}\right)
\in[0,\pi].
$$

Then

$$
\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta.
$$

**Proof.** By Theorem 1.6, the ratio is in $[-1,1]$, so $\theta$ is well-defined.
Applying $\cos$ to both sides of the definition gives the identity. $\blacksquare$

## Worked example

Let two token embeddings be

$$
\mathbf{a} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \qquad
\mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}.
$$

**Dot product:**

$$
\mathbf{a}\cdot\mathbf{b} = 2\cdot 3 + 1\cdot 4 = 6 + 4 = 10.
$$

**Norms:**

$$
\|\mathbf{a}\| = \sqrt{2^2+1^2} = \sqrt{5} \approx 2.236, \quad
\|\mathbf{b}\| = \sqrt{3^2+4^2} = \sqrt{25} = 5.
$$

**Cosine similarity:**

$$
\cos\theta = \frac{10}{2.236 \times 5} = \frac{10}{11.18} \approx 0.894.
$$

So the angle is $\theta \approx \arccos(0.894) \approx 26.6^\circ$ — the two
tokens point in fairly similar directions, i.e. they are "semantically close."

## Intuition for LLMs

Embeddings place words in a high-dimensional space where **direction encodes
meaning**. Similar words (king/queen) have small angles between them. Cosine
similarity is a normalized inner product, and inner products are the core
operation in [Chapter 10](../part3-sequences-attention/10-self-attention.md).

---

[← Notation](../00-notation.md) · [Next: Matrices →](02-matrices.md)
