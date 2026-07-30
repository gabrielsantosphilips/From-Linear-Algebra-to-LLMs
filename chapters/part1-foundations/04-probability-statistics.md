# 4. Probability & Statistics

[← Table of Contents](../../README.md)

## Why probability?

An LLM does not output a single word — it outputs a **probability distribution**
over the entire vocabulary. Two tools make this possible: the **softmax**
function (to build distributions) and **cross-entropy** (to train them).

## Basics

- A **distribution** over outcomes assigns probabilities $p_i \ge 0$ with $\sum_i p_i = 1$.
- **Expectation:** $\mathbb{E}[X] = \sum_i p_i x_i$.

## The softmax function

Given a vector of raw scores (**logits**) $\mathbf{z}\in\mathbb{R}^n$, softmax
turns them into a probability distribution:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}.
$$

Larger logits get exponentially more probability, but everything stays positive
and sums to 1.

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

They sum to $1.0$. The model would predict the first token with 65.9% probability.

## Cross-entropy loss

To train, we compare the predicted distribution $\mathbf{p}$ against the true
(one-hot) label $\mathbf{y}$:

$$
H(\mathbf{y},\mathbf{p}) = -\sum_{i} y_i \log p_i.
$$

For a one-hot label at the correct class $c$, this simplifies to
$-\log p_c$.

### Worked example

If the correct token was class 1 (probability $0.659$ above):

$$
\text{loss} = -\log(0.659) = 0.417.
$$

If the model had instead assigned it only $0.099$:

$$
\text{loss} = -\log(0.099) = 2.313.
$$

Being confident **and correct** yields low loss; being wrong is punished heavily.

## Maximum likelihood

Training an LLM = maximizing the likelihood of the observed text = minimizing
the average cross-entropy (equivalently, **perplexity** $= e^{\text{loss}}$).

## Intuition for LLMs

Softmax is the *last step* of the model (over the vocabulary) **and** the core
of attention weights. Cross-entropy is the training signal that shapes every
parameter. These two functions appear again in
[Chapter 14](../part4-transformers-llms/14-training-objective.md).

---

[← Linear Transformations & SVD](03-linear-transformations-svd.md) · [Next: Calculus & Gradients →](05-calculus-gradients.md)
