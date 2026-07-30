# 10. Self-Attention (the centerpiece)

[← Table of Contents](../../README.md)

This is the mechanism that makes transformers work. Read slowly — we compute a
complete example by hand.

## The idea

Each token should be able to **look at** every other token and decide how much
to pay attention to it. “It” in a sentence should attend to whatever noun it
refers to. Self-attention computes, for every token, a weighted average of all
tokens' information, where weights depend on relevance.

## Queries, Keys, Values

From each input embedding $\mathbf{x}$ we produce three vectors via learned
matrices $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}^K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}^V.
$$

- **Query**: what this token is looking for.
- **Key**: what this token offers.
- **Value**: the information this token carries.

## Scaled dot-product attention

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}.
$$

The dot product $\mathbf{Q}\mathbf{K}^\top$ scores query–key relevance; dividing by
$\sqrt{d_k}$ keeps scores from getting too large; softmax turns them into
weights that sum to 1; multiplying by $\mathbf{V}$ produces the weighted output.

---

## Complete worked example (3 tokens, $d_k = 2$)

Assume we already have $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ for a 3-token
sequence (to keep the arithmetic clean):

$$
\mathbf{Q} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix},\quad
\mathbf{K} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix},\quad
\mathbf{V} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 1 & 1 \end{bmatrix}.
$$

### Step 1 — scores $\mathbf{Q}\mathbf{K}^\top$ (a $3\times 3$ matrix)

Row 1 ($\mathbf{q}_1=[1,0]$) dotted with each key row $[1,0],[1,1],[0,1]$:
$$
[1,\ 1,\ 0].
$$
Row 2 ($\mathbf{q}_2=[0,1]$): $[0,\ 1,\ 1]$.
Row 3 ($\mathbf{q}_3=[1,1]$): $[1,\ 2,\ 1]$.

$$
\mathbf{Q}\mathbf{K}^\top = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 2 & 1 \end{bmatrix}.
$$

### Step 2 — scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$

$$
\frac{1}{1.414}\begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 2 & 1 \end{bmatrix}
= \begin{bmatrix} 0.707 & 0.707 & 0 \\ 0 & 0.707 & 0.707 \\ 0.707 & 1.414 & 0.707 \end{bmatrix}.
$$

### Step 3 — softmax each row

**Row 1** $[0.707, 0.707, 0]$: $e^{0.707}=2.028,\ e^{0.707}=2.028,\ e^{0}=1$, sum $=5.056$.
$$
\to [0.401,\ 0.401,\ 0.198].
$$

**Row 2** $[0, 0.707, 0.707]$: $1, 2.028, 2.028$, sum $=5.056$.
$$
\to [0.198,\ 0.401,\ 0.401].
$$

**Row 3** $[0.707, 1.414, 0.707]$: $2.028, 4.113, 2.028$, sum $=8.169$.
$$
\to [0.248,\ 0.503,\ 0.248].
$$

So the attention weight matrix is
$$
\mathbf{A} = \begin{bmatrix} 0.401 & 0.401 & 0.198 \\ 0.198 & 0.401 & 0.401 \\ 0.248 & 0.503 & 0.248 \end{bmatrix}.
$$

### Step 4 — multiply by $\mathbf{V}$

Each output row is a weighted sum of value rows $[2,0],[0,3],[1,1]$.

**Output row 1:**
$$
0.401[2,0] + 0.401[0,3] + 0.198[1,1] = [0.802+0.198,\ 1.203+0.198] = [1.000,\ 1.401].
$$

**Output row 2:**
$$
0.198[2,0] + 0.401[0,3] + 0.401[1,1] = [0.396+0.401,\ 1.203+0.401] = [0.797,\ 1.604].
$$

**Output row 3:**
$$
0.248[2,0] + 0.503[0,3] + 0.248[1,1] = [0.496+0.248,\ 1.509+0.248] = [0.744,\ 1.757].
$$

$$
\boxed{\ \text{Attention output} = \begin{bmatrix} 1.000 & 1.401 \\ 0.797 & 1.604 \\ 0.744 & 1.757 \end{bmatrix}\ }
$$

Each token now carries a blended representation that mixes in information from
the tokens it attended to most.

## Causal masking (for GPT-style models)

In text generation, a token may only attend to **earlier** tokens. We enforce
this by setting future scores to $-\infty$ *before* softmax, so their weights
become 0.

## Intuition for LLMs

Self-attention lets every token gather context from the whole sequence in
parallel — no recurrence needed. This is why transformers train efficiently and
model long-range dependencies. It is, quite literally, “Attention Is All You
Need.”

---

[← Embeddings](09-embeddings.md) · [Next: Multi-Head Attention →](11-multi-head-attention.md)
