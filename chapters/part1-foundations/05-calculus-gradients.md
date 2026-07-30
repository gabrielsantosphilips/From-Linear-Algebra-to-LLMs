# 5. Calculus & Gradients

[← Table of Contents](../../README.md)

## Formal definitions

**Definition 5.1 (Limit and derivative, one variable).**
$$
\lim_{x\to a} f(x)=L
$$
means: $\forall\varepsilon>0\,\exists\delta>0$ such that
$0<|x-a|<\delta\Rightarrow |f(x)-L|<\varepsilon$.
The derivative at $a$ is
$$
f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}
$$
if this limit exists.

**Definition 5.2 (Partial derivative, gradient, Jacobian).**
For $f:\mathbb{R}^n\to\mathbb{R}$,
$$
\frac{\partial f}{\partial x_i}(\mathbf{x})=
\lim_{h\to0}\frac{f(\mathbf{x}+h\mathbf{e}_i)-f(\mathbf{x})}{h},
\qquad
\nabla f(\mathbf{x})=
\begin{bmatrix}
\partial f/\partial x_1 \\
\vdots \\
\partial f/\partial x_n
\end{bmatrix}.
$$
For $F:\mathbb{R}^n\to\mathbb{R}^m$, the Jacobian is
$J_F(\mathbf{x})\in\mathbb{R}^{m\times n}$ with
$(J_F)_{ij}=\partial F_i/\partial x_j$.

**Definition 5.3 (Differentiability / total derivative).**
$F:\mathbb{R}^n\to\mathbb{R}^m$ is differentiable at $\mathbf{x}$ if there exists
a linear map $L$ such that
$$
\lim_{\mathbf{h}\to\mathbf{0}}
\frac{\|F(\mathbf{x}+\mathbf{h})-F(\mathbf{x})-L\mathbf{h}\|}{\|\mathbf{h}\|}=0.
$$
Then $L=J_F(\mathbf{x})$.

## Chain rule and backpropagation

**Theorem 5.4 (Single-variable chain rule).** If $g$ is differentiable at $x$ and
$f$ differentiable at $g(x)$, then
$$
\frac{d}{dx}f(g(x))=f'(g(x))g'(x).
$$

**Proof.** Write
$$
f(g(x+h))-f(g(x))
=f'(g(x))(g(x+h)-g(x))+r(h),
$$
where $r(h)/(g(x+h)-g(x))\to 0$. Divide by $h$:
$$
\frac{f(g(x+h))-f(g(x))}{h}
=f'(g(x))\frac{g(x+h)-g(x)}{h}+\frac{r(h)}{h}.
$$
As $h\to0$, first term goes to $f'(g(x))g'(x)$ and second to $0$. $\blacksquare$

**Theorem 5.5 (Multivariate chain rule).** If
$F:\mathbb{R}^n\to\mathbb{R}^m$ is differentiable at $\mathbf{x}$ and
$G:\mathbb{R}^m\to\mathbb{R}^p$ differentiable at $F(\mathbf{x})$, then
$$
J_{G\circ F}(\mathbf{x})=J_G(F(\mathbf{x}))\,J_F(\mathbf{x}).
$$

**Proof sketch.** Compose first-order expansions from Definition 5.3:
$F(\mathbf{x}+\mathbf{h})=F(\mathbf{x})+J_F\mathbf{h}+o(\|\mathbf{h}\|)$,
$G(\mathbf{y}+\mathbf{k})=G(\mathbf{y})+J_G\mathbf{k}+o(\|\mathbf{k}\|)$,
then substitute $\mathbf{k}=J_F\mathbf{h}+o(\|\mathbf{h}\|)$ and collect linear
terms. $\blacksquare$

**Proposition 5.6 (Backpropagation recurrence).** For a feed-forward network
$$
\mathbf{a}^{(\ell)}=\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}+\mathbf{b}^{(\ell)},
\quad
\mathbf{h}^{(\ell)}=\phi^{(\ell)}(\mathbf{a}^{(\ell)}),
$$
with scalar loss $L$, define
$\boldsymbol\delta^{(\ell)}=\partial L/\partial \mathbf{a}^{(\ell)}$. Then
$$
\boldsymbol\delta^{(\ell)}=
\left((\mathbf{W}^{(\ell+1)})^\top\boldsymbol\delta^{(\ell+1)}\right)
\odot \phi'^{(\ell)}(\mathbf{a}^{(\ell)}),
$$
and
$$
\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}=
\boldsymbol\delta^{(\ell)}(\mathbf{h}^{(\ell-1)})^\top,
\qquad
\frac{\partial L}{\partial \mathbf{b}^{(\ell)}}=\boldsymbol\delta^{(\ell)}.
$$

**Proof.** Apply Theorem 5.5 layer-by-layer to
$L\circ h^{(L)}\circ\cdots\circ h^{(1)}$. The Jacobian of affine map gives
transpose multiplication by $\mathbf{W}^{(\ell+1)}$, and elementwise nonlinearity
gives Hadamard product by $\phi'$. $\blacksquare$

## Worked example

Let $g(x) = 2x + 1$ and $f(g) = g^2$, so $z = (2x+1)^2$.

$$
\frac{dz}{dg} = 2g = 2(2x+1), \qquad \frac{dg}{dx} = 2.
$$
$$
\frac{dz}{dx} = 2(2x+1)\cdot 2 = 4(2x+1).
$$

At $x=1$: $\frac{dz}{dx} = 4(3) = 12$.

## Convexity and optimality

**Definition 5.7 (Convex function).** A function
$f:\mathbb{R}^n\to\mathbb{R}$ is convex if
$$
f(t\mathbf{x}+(1-t)\mathbf{y})\le tf(\mathbf{x})+(1-t)f(\mathbf{y})
\quad \forall\mathbf{x},\mathbf{y},\ t\in[0,1].
$$

**Theorem 5.8 (Stationary point of differentiable convex function is global
minimum).** If $f$ is differentiable and convex, and $\nabla f(\mathbf{x}_\star)=0$,
then
$$
f(\mathbf{x})\ge f(\mathbf{x}_\star)\quad\forall\mathbf{x}.
$$

**Proof.** Differentiable convex functions satisfy first-order condition:
$$
f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}).
$$
Set $\mathbf{x}=\mathbf{x}_\star$ and use $\nabla f(\mathbf{x}_\star)=0$. $\blacksquare$

**Definition 5.9 (Hessian and second-order conditions).** If $f$ is twice
 differentiable, Hessian is $\nabla^2 f(\mathbf{x})$.
- First-order necessary condition for local optimum: $\nabla f(\mathbf{x}_\star)=0$.
- Second-order sufficient condition for strict local minimum:
  $\nabla f(\mathbf{x}_\star)=0$ and $\nabla^2 f(\mathbf{x}_\star)\succ 0$.
- If $\nabla^2 f(\mathbf{x})\succeq 0$ for all $\mathbf{x}$, then $f$ is convex.

## Intuition for LLMs

Backpropagation is just Theorem 5.5 applied repeatedly. This is the rigorous
bridge from calculus to practical training in
[Chapter 8](../part2-neurons-to-networks/08-loss-gradient-descent.md).

---

[← Probability & Statistics](04-probability-statistics.md) · [Next: The Perceptron →](../part2-neurons-to-networks/06-perceptron-linear-models.md)
