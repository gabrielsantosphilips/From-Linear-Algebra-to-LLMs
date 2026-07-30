# 16. Scaling & Emergence

[← Table of Contents](../../README.md)

## Counting parameters

Let's estimate the parameter count of a small transformer to build intuition.
With $d_{\text{model}} = 512$, $N = 6$ blocks, vocabulary $V = 30{,}000$,
FFN hidden $= 4d = 2048$.

**Per block:**
- Attention: 4 matrices ($\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V,\mathbf{W}^O$),
  each $512\times 512$ $\Rightarrow 4 \times 512^2 = 1{,}048{,}576$.
- FFN: $512\times 2048$ + $2048\times 512 = 2 \times 1{,}048{,}576 = 2{,}097{,}152$.
- Total per block $\approx 3.15$M.

**6 blocks:** $6 \times 3.15\text{M} \approx 18.9$M.

**Embeddings:** $V \times d = 30{,}000 \times 512 = 15.36$M.

**Grand total** $\approx 34$M parameters (ignoring biases/LayerNorm, which are
small). Real LLMs scale each knob up enormously: GPT-3 has 175 **billion**
parameters, 96 blocks, $d_{\text{model}}=12{,}288$.

## Scaling laws

Empirically, loss falls **predictably** as a power law in model size $N$,
dataset size $D$, and compute $C$:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + L_\infty,
$$

with small exponents (e.g. $\alpha_N \approx 0.076$). More parameters + more
data + more compute → reliably lower loss. The Chinchilla result refined this:
for a compute budget, model and data size should grow **together**.

## Emergent abilities

Some capabilities (multi-step arithmetic, in-context learning, chain-of-thought
reasoning) are near-absent in small models and appear relatively **suddenly**
past a scale threshold. Whether these are true phase transitions or artifacts
of metrics is debated — but the practical effect is real: scale unlocks
qualitatively new behavior.

## Beyond pre-training

Raw next-token models are then **aligned**:

- **Supervised fine-tuning (SFT):** train on curated instruction–response pairs.
- **RLHF:** a reward model learns human preferences; the LLM is optimized (e.g.
  with PPO) to maximize that reward — making it helpful, harmless, and honest.
- **Parameter-efficient fine-tuning (LoRA):** low-rank updates
  ([Chapter 3](../part1-foundations/03-linear-transformations-svd.md)) adapt a
  frozen model cheaply.

## The whole journey

You now have the full chain:

> vectors → matrices → linear layers → nonlinearities → gradient descent →
> embeddings → attention → transformer blocks → next-token training → sampling
> → scale → an LLM.

Everything an LLM does reduces to the linear algebra, probability, and calculus
in this booklet — executed billions of times, at enormous scale.

---

[← Inference & Sampling](15-inference-sampling.md) · [Appendix A: Cheat Sheet →](../appendix/A-linear-algebra-cheatsheet.md)
