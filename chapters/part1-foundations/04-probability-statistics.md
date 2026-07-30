# 4. Probability & Statistics

[← Table of Contents](../../README.md)

## Formal foundations

**Definition 4.1 (Probability space).** A probability space is a triple
$(\Omega,\mathcal{F},\mathbb{P})$ where:
- $\Omega$ is the sample space,
- $\mathcal{F}\subseteq 2^\Omega$ is a $\sigma$-algebra of events,
- $\mathbb{P}:\mathcal{F}\to[0,1]$ satisfies Kolmogorov axioms:
  1. $\mathbb{P}(A)\ge0$,
  2. $\mathbb{P}(\Omega)=1$,
  3. For pairwise disjoint $A_i$, $\mathbb{P}(\cup_i A_i)=\sum_i\mathbb{P}(A_i)$.

**Definition 4.2 (Random variable, expectation, variance).** A random variable is
a measurable function $X: \Omega\to\mathbb{R}$. Its expectation is
$\mathbb{E}[X]$ (sum/integral form as appropriate), and variance is
$\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2]$.

## Softmax and its properties

For logits $\mathbf{z}\in\mathbb{R}^n$, define
$$
p_i=\text{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}.
$$

**Proposition 4.3 (Softmax gives a probability distribution).**
$p_i>0$ for all $i$, and $\sum_i p_i=1$.

**Proof.** Exponentials are positive, so $p_i>0$. Also
$$
\sum_i p_i=\frac{\sum_i e^{z_i}}{\sum_j e^{z_j}}=1.
$$
$\blacksquare$

**Proposition 4.4 (Shift invariance).** For any $c\in\mathbb{R}$,
$$
\text{softmax}(\mathbf{z}+c\mathbf{1})=\text{softmax}(\mathbf{z}).
$$

**Proof.**
$$
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}=
\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}=
\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$
$\blacksquare$

**Theorem 4.5 (Softmax Jacobian).**
$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
$$

**Proof.** Let $S=\sum_k e^{z_k}$, so $p_i=e^{z_i}/S$.
If $i=j$:
$$
\frac{\partial p_i}{\partial z_i}=
\frac{e^{z_i}S-e^{z_i}e^{z_i}}{S^2}=p_i(1-p_i).
$$
If $i\neq j$:
$$
\frac{\partial p_i}{\partial z_j}=
\frac{0\cdot S-e^{z_i}e^{z_j}}{S^2}=-p_i p_j.
$$
Both cases unify as $p_i(\delta_{ij}-p_j)$. $\blacksquare$

### Worked example

Logits $\mathbf{z} = [2.0,\ 1.0,\ 0.1]$.

$$
e^{2.0}=7.389,\quad e^{1.0}=2.718,\quad e^{0.1}=1.105.
$$
$$
\text{sum} = 7.389 + 2.718 + 1.105 = 11.212.
$$
$$
\text{softmax} = \left[\tfrac{7.389}{11.212},\ \tfrac{2.718}{11.212},\ \tfrac{1.105}{11.212}\right] = [0.659,\ 0.242,\ 0.099].
$$

## Entropy, cross-entropy, KL divergence

**Definition 4.6 (Entropy).** For a discrete distribution $p$,
$$
H(p)=-\sum_i p_i\log p_i.
$$

**Definition 4.7 (Cross-entropy).**
$$
H(q,p)=-\sum_i q_i\log p_i.
$$

**Definition 4.8 (KL divergence).**
$$
\operatorname{KL}(q\|p)=\sum_i q_i\log\frac{q_i}{p_i}.
$$

**Theorem 4.9 (Gibbs' inequality).**
$$
\operatorname{KL}(q\|p)\ge 0,
$$
with equality iff $q=p$ (on support of $q$).

**Proof.** Since $\log$ is concave, $-\log$ is convex. By Jensen with random
variable $Y=p_i/q_i$ under weights $q_i$,
$$
\sum_i q_i\left(-\log\frac{p_i}{q_i}\right)
\ge -\log\left(\sum_i q_i\frac{p_i}{q_i}\right)
=-\log\left(\sum_i p_i\right)= -\log 1 = 0.
$$
Left side is $\operatorname{KL}(q\|p)$. Equality condition is the Jensen equality
case, equivalent to $p_i/q_i$ constant on support of $q$, giving $p=q$. $\blacksquare$

**Corollary 4.10 (Cross-entropy decomposition).**
$$
H(q,p)=H(q)+\operatorname{KL}(q\|p)\ge H(q).
$$

**Proof.** Rearrange Definition 4.8. Then apply Theorem 4.9. $\blacksquare$

## Cross-entropy and maximum likelihood

For one-hot label $q=\mathbf{e}_c$, cross-entropy is $-\log p_c$. Over a dataset,
minimizing average cross-entropy is equivalent to maximizing
$\sum_t \log p_\theta(y_t\mid x_t)$, i.e. maximum likelihood.

**Proposition 4.11 (Convexity in logits for fixed target distribution).**
For fixed $q$, the function
$$
\ell(\mathbf{z})=-\sum_i q_i z_i + \log\sum_j e^{z_j}
$$
is convex in $\mathbf{z}$.

**Proof sketch.** Its gradient is $\nabla\ell=\text{softmax}(\mathbf{z})-q$. Its
Hessian is the softmax Jacobian matrix
$\mathbf{H}=\operatorname{diag}(p)-pp^\top$. For any $\mathbf{v}$,
$$
\mathbf{v}^\top\mathbf{H}\mathbf{v}=\sum_i p_i v_i^2-\left(\sum_i p_i v_i\right)^2
=\operatorname{Var}_{i\sim p}(v_i)\ge0.
$$
So $\mathbf{H}\succeq 0$, hence convex. $\blacksquare$

## Intuition for LLMs

Softmax and cross-entropy appear both in output-token prediction and attention.
The precise gradient formula in Theorem 4.5 is reused directly in
[Chapter 14](../part4-transformers-llms/14-training-objective.md).

---

[← Linear Transformations & SVD](03-linear-transformations-svd.md) · [Next: Calculus & Gradients →](05-calculus-gradients.md)
