# 14. The Training Objective

[← Table of Contents](../../README.md)

## Next-token prediction

An LLM is trained on one deceptively simple task: **given the previous tokens,
predict the next one.** Over enough text, mastering this task forces the model
to learn grammar, facts, reasoning patterns, and style.

For a sequence $t_1, t_2, \dots, t_T$, the model maximizes

$$
P(t_1,\dots,t_T) = \prod_{i=1}^{T} P(t_i \mid t_1,\dots,t_{i-1}).
$$

## The loss

Taking the negative log turns the product into a sum — the **cross-entropy**
over the vocabulary at each position (recall [Chapter 4](../part1-foundations/04-probability-statistics.md)):

$$
L = -\frac{1}{T}\sum_{i=1}^{T} \log P(t_i \mid t_{<i}).
$$

### Worked example

Suppose at three positions the model assigned the *correct* next token
probabilities $0.6, 0.3, 0.8$. The loss is

$$
L = -\tfrac{1}{3}\big(\log 0.6 + \log 0.3 + \log 0.8\big)
= -\tfrac{1}{3}(-0.511 - 1.204 - 0.223) = \tfrac{1.938}{3} = 0.646.
$$

**Perplexity** $= e^{L} = e^{0.646} = 1.91$ — loosely, the model is “as confused
as if choosing between ~1.9 options” at each step.

## Teacher forcing

During training we feed the **true** previous tokens (not the model's own
guesses) as context. This stabilizes and parallelizes training: every position
in the sequence is predicted simultaneously, each conditioned on the real prefix
(with causal masking from [Chapter 10](../part3-sequences-attention/10-self-attention.md)).

```mermaid
flowchart LR
    A[Token sequence] --> B[Shifted input tokens]
    B --> C[Transformer forward pass]
    C --> D[Vocabulary logits]
    D --> E[Softmax probabilities]
    E --> F[Cross-entropy vs true next token]
    F --> G[Backpropagation]
    G --> H[Parameter update]
```

*Diagram: Next-token training loop with teacher forcing.*

## Tokenization (brief)

Text is split into subword tokens (e.g. Byte-Pair Encoding). “unbelievable”
might become `un` + `believ` + `able`. This keeps the vocabulary ($V\approx 30\text{k}–100\text{k}$) manageable while covering any word, including unseen ones.

## Intuition for LLMs

There is no special “intelligence module.” The entire capability of an LLM
emerges from relentlessly minimizing next-token cross-entropy over vast text,
using gradient descent ([Chapter 8](../part2-neurons-to-networks/08-loss-gradient-descent.md))
and backpropagation ([Chapter 5](../part1-foundations/05-calculus-gradients.md)).

---

[← Transformer Architecture](13-transformer-architecture.md) · [Next: Inference & Sampling →](15-inference-sampling.md)
