# 11. Multi-Head Attention

[← Table of Contents](../../README.md)

## The idea

One attention computation captures **one kind** of relationship. Language has
many simultaneously (syntax, coreference, tense...). **Multi-head attention**
runs several attention operations (“heads”) in parallel, each with its own
$\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V$, then combines them.

## The mechanism

For $h$ heads, each of dimension $d_k = d_{\text{model}}/h$:

$$
\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}^Q_i,\ \mathbf{X}\mathbf{W}^K_i,\ \mathbf{X}\mathbf{W}^V_i).
$$

Concatenate the heads and apply an output projection $\mathbf{W}^O$:

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,\mathbf{W}^O.
$$

## Dimensions worked out

Suppose $d_{\text{model}} = 512$ and $h = 8$ heads. Then each head works in
$d_k = 512/8 = 64$ dimensions.

- Each $\mathbf{W}^Q_i,\mathbf{W}^K_i,\mathbf{W}^V_i \in \mathbb{R}^{512\times 64}$.
- Each head outputs a matrix of shape (seq_len $\times$ 64).
- Concatenating 8 heads: seq_len $\times$ (8×64) = seq_len $\times$ 512.
- $\mathbf{W}^O \in \mathbb{R}^{512\times 512}$ maps back to $d_{\text{model}}$.

## Small numeric illustration

Using our [Chapter 10](10-self-attention.md) output as **head 1** (seq_len 3,
$d_k=2$):

$$
\text{head}_1 = \begin{bmatrix} 1.000 & 1.401 \\ 0.797 & 1.604 \\ 0.744 & 1.757 \end{bmatrix}.
$$

Suppose a second head produced

$$
\text{head}_2 = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.9 \\ 0.3 & 0.3 \end{bmatrix}.
$$

Concatenation gives a $3\times 4$ matrix:

$$
\text{Concat} = \begin{bmatrix} 1.000 & 1.401 & 0.5 & 0.2 \\ 0.797 & 1.604 & 0.1 & 0.9 \\ 0.744 & 1.757 & 0.3 & 0.3 \end{bmatrix},
$$

which $\mathbf{W}^O \in \mathbb{R}^{4\times d_{\text{model}}}$ then projects to the
output dimension.

## Intuition for LLMs

Different heads specialize — some track subject–verb agreement, others resolve
pronouns, others attend to nearby words. Multi-head attention gives the model
several “perspectives” on the sequence at once, then fuses them.

---

[← Self-Attention](10-self-attention.md) · [Next: Positional Encoding →](12-positional-encoding.md)
