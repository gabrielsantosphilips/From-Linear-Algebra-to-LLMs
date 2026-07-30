# Appendix E — Mathematical Prerequisites

[← Table of Contents](../../README.md)

This appendix is a compact recap of background used in the proofs.

## Sets and functions

- A **set** is a collection of elements.
- Notation: $x\in A$, $A\subseteq B$, $A\cap B$, $A\cup B$, $A\setminus B$.
- Set-builder notation: $\{x\in A : P(x)\}$.
- A **function** $f:A\to B$ assigns each $x\in A$ exactly one $f(x)\in B$.

## Logic symbols

- $\forall x$: for all $x$.
- $\exists x$: there exists $x$.
- $P\Rightarrow Q$: if $P$ then $Q$.
- $P\iff Q$: equivalent statements.
- $\neg P$: not $P$.

## Fields and vector-space scalars

A **field** $\mathbb{F}$ is a set with addition/multiplication where:
- $\mathbb{F}$ is an abelian group under addition,
- nonzero elements form an abelian group under multiplication,
- multiplication distributes over addition.

In this booklet, scalars are usually in $\mathbb{R}$ (real numbers).

## Useful inequalities

- Cauchy–Schwarz:
  $|\mathbf{a}\cdot\mathbf{b}|\le\|\mathbf{a}\|\|\mathbf{b}\|$.
- Triangle inequality:
  $\|\mathbf{a}+\mathbf{b}\|\le\|\mathbf{a}\|+\|\mathbf{b}\|$.
- Jensen (convex $\varphi$):
  $\varphi(\mathbb{E}[X])\le\mathbb{E}[\varphi(X)]$.

## Matrix facts used repeatedly

- $(\mathbf{A}\mathbf{B})^\top=\mathbf{B}^\top\mathbf{A}^\top$.
- Symmetric matrices have orthonormal eigenbases (Theorem 3.7).
- $\mathbf{A}^\top\mathbf{A}$ is symmetric positive semidefinite.

## Differential notation map

- Scalar-to-scalar: derivative $f'(x)$.
- Vector-to-scalar: gradient $\nabla f$.
- Vector-to-vector: Jacobian $J_F$.
- Second order: Hessian $\nabla^2 f$.

---

[← Appendix D](D-theorems-and-proofs.md) · [Back to Table of Contents →](../../README.md)
