# 6. The Perceptron & Linear Models

[← Table of Contents](../../README.md)

## Definition

The **perceptron** (an artificial neuron) is the atom of neural networks. It
takes an input vector $\mathbf{x}$, computes a weighted sum plus a bias, and
passes the result through an **activation function** $f$:

$$
y = f(\mathbf{w}\cdot\mathbf{x} + b) = f\!\left(\sum_{i=1}^n w_i x_i + b\right).
$$

A whole **layer** stacks many neurons, giving the familiar matrix form:

$$
\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b}).
$$

## Worked example (forward pass)

Input $\mathbf{x} = [1, 2]$, weights $\mathbf{w} = [0.5, -1]$, bias $b = 0.3$,
activation = step function ($f(z)=1$ if $z\ge 0$, else $0$).

$$
z = 0.5\cdot 1 + (-1)\cdot 2 + 0.3 = 0.5 - 2 + 0.3 = -1.2.
$$

$$
y = f(-1.2) = 0.
$$

With a sigmoid activation instead (see [Chapter 7](07-activations-nonlinearity.md)):

$$
y = \sigma(-1.2) = \frac{1}{1+e^{1.2}} = \frac{1}{1+3.32} = 0.231.
$$

## Limitations of a single neuron

A single perceptron can only separate data with a **straight line/plane** (it
is a *linear* classifier). Famously, it cannot solve **XOR**. The fix is to
stack layers and insert **nonlinear** activations — the topic of the next
chapter.

## Intuition for LLMs

Every transformer contains thousands of these neurons in its feed-forward
sub-layers. The perceptron is the conceptual seed from which the entire model
grows: `linear transform → nonlinearity → repeat`.

---

[← Calculus & Gradients](../part1-foundations/05-calculus-gradients.md) · [Next: Activations →](07-activations-nonlinearity.md)
