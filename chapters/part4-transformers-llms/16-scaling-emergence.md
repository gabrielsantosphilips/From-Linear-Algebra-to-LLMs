# 16. Scaling & Emergence

[← Table of Contents](../../README.md)

## Counting parameters

Let's estimate a small transformer. With $d_{\text{model}} = 512$, $N = 6$
blocks, vocabulary $V = 30{,}000$, FFN hidden $=4d=2048$:

- Per block attention params: $4d_{\text{model}}^2=1{,}048{,}576$.
- Per block FFN params: $2d_{\text{model}}(4d_{\text{model}})=2{,}097{,}152$.
- Total per block $\approx 3.15$M.
- Total blocks: $N\cdot 3.15\text{M}\approx 18.9$M.
- Embedding matrix: $Vd_{\text{model}}=15.36$M.
- Grand total: $\approx 34$M parameters.

## Scaling laws (empirical functional forms)

Define:
- $N$: parameter count,
- $D$: number of training tokens,
- $C$: training compute,
- $L$: validation loss,
- $L_\infty$: irreducible loss floor.

A common empirical model family is
$$
L(N,D,C)\approx L_\infty + a_N N^{-\alpha_N} + a_D D^{-\alpha_D} + a_C C^{-\alpha_C},
$$
with constants $a_\bullet>0$ and exponents $\alpha_\bullet\in(0,1)$ fit from
data. Holding two variables fixed yields one-dimensional power laws, e.g.
$$
L(N)\approx L_\infty + a_N N^{-\alpha_N}.
$$

Compute-optimal training laws are often written as
$$
N^\star(C)\propto C^{\beta_N},\qquad D^\star(C)\propto C^{\beta_D},
$$
capturing the empirical finding that model size and data should scale together.

## Emergent abilities

Certain benchmark capabilities appear sharply only past scale thresholds. These
are empirical observations, not proven mathematical phase transitions.

## Beyond pre-training

- Supervised fine-tuning (SFT)
- RLHF
- Parameter-efficient fine-tuning (LoRA; see [Theorem 3.8](../part1-foundations/03-linear-transformations-svd.md))

## The whole journey

> vectors → matrices → linear maps → nonlinearities → gradient descent →
> embeddings → attention → transformer blocks → next-token training → sampling
> → scale → an LLM.

---

[← Inference & Sampling](15-inference-sampling.md) · [Appendix A: Cheat Sheet →](../appendix/A-linear-algebra-cheatsheet.md)
