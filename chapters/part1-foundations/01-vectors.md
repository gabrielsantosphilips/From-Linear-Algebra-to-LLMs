# 1. Vectors & Vector Spaces

[← Table of Contents](../../README.md)

## Definition

A **vector** is an ordered list of numbers. In an LLM, every token (roughly, a
word or word-piece) is represented by a vector of numbers called an
**embedding**. Geometrically a vector is an arrow from the origin to a point.

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

A **vector space** $\mathbb{R}^n$ is the set of all such $n$-tuples, closed under:

- **Addition:** $(\mathbf{a}+\mathbf{b})_i = a_i + b_i$
- **Scalar multiplication:** $(c\,\mathbf{a})_i = c\,a_i$

## Core operations

### Dot product

$$
\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

### Norm (length)

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}, \qquad \|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|
$$

### Cosine similarity

Measures the *angle* between two vectors, ignoring magnitude:

$$
\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}
$$

A value of $1$ means same direction, $0$ means orthogonal (unrelated), $-1$
means opposite.

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
similarity is *the* workhorse for comparing meaning, and — as we'll see in
[Chapter 10](../part3-sequences-attention/10-self-attention.md) — the dot
product is the heart of the attention mechanism.

---

[← Notation](../00-notation.md) · [Next: Matrices →](02-matrices.md)
