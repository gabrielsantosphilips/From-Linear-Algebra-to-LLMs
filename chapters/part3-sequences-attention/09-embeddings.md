# 9. Embeddings

[← Table of Contents](../../README.md)

## From words to vectors

Computers cannot process raw text, so we convert tokens into vectors. The
pipeline is:

```
text → tokenizer → token IDs (integers) → embedding lookup → vectors
```

## The embedding matrix

An **embedding matrix** $\mathbf{E}\in\mathbb{R}^{V\times d}$ has one row per
vocabulary token ($V$ tokens) and $d$ columns (the embedding dimension).
Looking up token ID $t$ simply selects row $t$:

$$
\mathbf{e}_t = \mathbf{E}[t, :].
$$

This is mathematically equivalent to multiplying a one-hot vector by $\mathbf{E}$.

## Worked example

Suppose a tiny vocabulary of 4 tokens and $d=3$:

$$
\mathbf{E} = \begin{bmatrix}
0.1 & 0.3 & -0.2 \\
0.9 & -0.1 & 0.4 \\
-0.5 & 0.8 & 0.2 \\
0.0 & 0.6 & -0.7
\end{bmatrix}
\begin{matrix}
\leftarrow \text{“the” (id 0)} \\
\leftarrow \text{“cat” (id 1)} \\
\leftarrow \text{“sat” (id 2)} \\
\leftarrow \text{“mat” (id 3)}
\end{matrix}
$$

The sentence “the cat sat” = token IDs $[0, 1, 2]$ becomes the sequence of
vectors:

$$
\mathbf{e}_0 = [0.1, 0.3, -0.2],\quad
\mathbf{e}_1 = [0.9, -0.1, 0.4],\quad
\mathbf{e}_2 = [-0.5, 0.8, 0.2].
$$

Stacked, the input to the transformer is a $3\times 3$ matrix (sequence length × $d$).

## Learned vs. fixed

The entries of $\mathbf{E}$ are **learned** during training by gradient descent,
just like any other weights. Over training, tokens with similar meaning drift to
nearby vectors (small cosine distance — recall [Chapter 1](../part1-foundations/01-vectors.md)).

## Intuition for LLMs

Embeddings are the bridge from discrete language to the continuous world of
linear algebra. Everything after this point — attention, feed-forward layers —
operates on these vectors. The famous analogy
$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$
lives in this space.

---

[← Loss & Gradient Descent](../part2-neurons-to-networks/08-loss-gradient-descent.md) · [Next: Self-Attention →](10-self-attention.md)
