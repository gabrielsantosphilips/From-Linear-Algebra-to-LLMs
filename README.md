# From Linear Algebra to LLMs

> A self-contained mathematical booklet that builds up, step by step, from
> vectors and matrices to the inner workings of modern Large Language Models
> (LLMs). Every concept is followed by a **concrete numeric worked example**.

## Who is this for?

Anyone who wants to *really* understand how an LLM works — not just at the
"it predicts the next word" level, but at the level of the actual matrix
operations happening inside. We assume only high-school mathematics at the
start and build everything else from there.

## How to read this booklet

Each chapter follows the same rhythm:

1. **Definition** — what the object/operation is.
2. **Derivation** — the mathematics, kept as simple as possible but no simpler.
3. **Worked example** — real numbers, arithmetic shown step by step.
4. **Intuition** — why it matters for LLMs.

Read in order the first time — each chapter builds on the previous ones.

> **Math rendering:** This booklet uses LaTeX (`$...$` inline and `$$...$$`
> display). GitHub renders these natively in Markdown.

---

## Table of Contents

### Front matter
- [Notation & Conventions](chapters/00-notation.md)

### Part I — Mathematical Foundations
1. [Vectors & Vector Spaces](chapters/part1-foundations/01-vectors.md)
2. [Matrices & Matrix Operations](chapters/part1-foundations/02-matrices.md)
3. [Linear Transformations, Eigenvectors, SVD & PCA](chapters/part1-foundations/03-linear-transformations-svd.md)
4. [Probability & Statistics (softmax, cross-entropy)](chapters/part1-foundations/04-probability-statistics.md)
5. [Calculus & Gradients (backpropagation)](chapters/part1-foundations/05-calculus-gradients.md)

### Part II — From Neurons to Networks
6. [The Perceptron & Linear Models](chapters/part2-neurons-to-networks/06-perceptron-linear-models.md)
7. [Activation Functions & Nonlinearity](chapters/part2-neurons-to-networks/07-activations-nonlinearity.md)
8. [Loss Functions & Gradient Descent](chapters/part2-neurons-to-networks/08-loss-gradient-descent.md)

### Part III — Sequences & Attention
9. [Embeddings](chapters/part3-sequences-attention/09-embeddings.md)
10. [Self-Attention (the centerpiece)](chapters/part3-sequences-attention/10-self-attention.md)
11. [Multi-Head Attention](chapters/part3-sequences-attention/11-multi-head-attention.md)
12. [Positional Encoding](chapters/part3-sequences-attention/12-positional-encoding.md)

### Part IV — Transformers & LLMs
13. [The Transformer Architecture](chapters/part4-transformers-llms/13-transformer-architecture.md)
14. [The Training Objective](chapters/part4-transformers-llms/14-training-objective.md)
15. [Inference & Sampling](chapters/part4-transformers-llms/15-inference-sampling.md)
16. [Scaling & Emergence](chapters/part4-transformers-llms/16-scaling-emergence.md)

### Appendices
- [A — Linear Algebra Cheat Sheet](chapters/appendix/A-linear-algebra-cheatsheet.md)
- [B — Glossary](chapters/appendix/B-glossary.md)
- [C — Further Reading](chapters/appendix/C-further-reading.md)

---

*Built as an educational resource. Contributions and corrections welcome.*
