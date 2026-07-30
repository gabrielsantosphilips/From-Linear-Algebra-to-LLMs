# 5. Calculus & Gradients

[← Table of Contents](../../README.md)

## Why calculus?

Training means *adjusting weights to reduce loss*. To know **which way** to
adjust each weight, we need the **gradient** — the vector of partial derivatives
of the loss with respect to every parameter.

## Derivatives

The derivative measures how a function changes as its input changes:

$$
f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}.
$$

Example: $f(x)=x^2 \Rightarrow f'(x)=2x$. At $x=3$, the slope is $6$.

## Partial derivatives & gradient

For $f(x,y)$, the **gradient** collects both partials:

$$
\nabla f = \begin{bmatrix} \partial f/\partial x \\ \partial f/\partial y \end{bmatrix}.
$$

Example: $f(x,y) = x^2 + 3xy$.
$$
\frac{\partial f}{\partial x} = 2x + 3y, \qquad \frac{\partial f}{\partial y} = 3x.
$$
At $(x,y)=(1,2)$: $\nabla f = \begin{bmatrix} 2+6 \\ 3 \end{bmatrix} = \begin{bmatrix} 8 \\ 3 \end{bmatrix}.$

The gradient points in the direction of **steepest increase**; we step in the
*opposite* direction to minimize (see [Chapter 8](../part2-neurons-to-networks/08-loss-gradient-descent.md)).

## The chain rule

If $z = f(g(x))$, then

$$
\frac{dz}{dx} = \frac{dz}{dg}\cdot\frac{dg}{dx}.
$$

### Worked example

Let $g(x) = 2x + 1$ and $f(g) = g^2$, so $z = (2x+1)^2$.

$$
\frac{dz}{dg} = 2g = 2(2x+1), \qquad \frac{dg}{dx} = 2.
$$
$$
\frac{dz}{dx} = 2(2x+1)\cdot 2 = 4(2x+1).
$$

At $x=1$: $\frac{dz}{dx} = 4(3) = 12$. (Check: $z=(2x+1)^2$, $\frac{dz}{dx}=4(2x+1)$ ✓.)

## Backpropagation (the big idea)

A neural network is a giant composition of functions. **Backpropagation** is
just the chain rule applied systematically, from the loss backward to every
weight, reusing intermediate results. Each layer receives the gradient flowing
back, multiplies by its local derivative, and passes it further back.

$$
\underbrace{\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}}_{\text{what to update}}
= \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{W}^{(\ell)}}
$$

## Intuition for LLMs

Every one of an LLM's billions of parameters is nudged using its gradient,
computed by backpropagation. Calculus is *how the model learns*. The next parts
put vectors, matrices, probability, and calculus together into actual networks.

---

[← Probability & Statistics](04-probability-statistics.md) · [Next: The Perceptron →](../part2-neurons-to-networks/06-perceptron-linear-models.md)
