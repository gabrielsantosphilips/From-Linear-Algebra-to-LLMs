# Appendix B — Glossary

[← Table of Contents](../../README.md)

**Activation function** — A nonlinearity (ReLU, GELU, sigmoid) applied after a
linear layer so networks can model complex functions.

**Attention** — Mechanism where each token computes a weighted average of other
tokens' values, weighted by query–key relevance.

**Backpropagation** — Algorithm that computes gradients of the loss w.r.t. every
parameter by applying the chain rule backward through the network.

**Cross-entropy** — Loss measuring the difference between predicted and true
distributions; the training objective for next-token prediction.

**Decoder-only** — Transformer variant (GPT-style) using causal masking to
generate text left-to-right.

**Embedding** — A learned vector representing a token; direction encodes meaning.

**Eigenvector / eigenvalue** — A direction a matrix only scales, and its scale
factor $\lambda$.

**Feed-forward network (FFN)** — Two linear layers with a nonlinearity, applied
per position inside each transformer block.

**Gradient descent** — Optimization that steps parameters opposite the gradient
to reduce loss.

**Layer normalization** — Normalizes a vector to zero mean / unit variance, then
rescales; stabilizes deep-network training.

**Logits** — Raw model output scores before softmax.

**LoRA** — Low-Rank Adaptation; efficient fine-tuning via a low-rank weight
update.

**Multi-head attention** — Several attention operations in parallel, each
capturing a different relationship, then concatenated.

**Norm** — The length/magnitude of a vector.

**Perceptron** — A single artificial neuron: $y = f(\mathbf{w}\cdot\mathbf{x}+b)$.

**Perplexity** — $e^{\text{loss}}$; how “confused” the model is on average.

**Positional encoding** — Position information added to embeddings so order
matters.

**Query / Key / Value** — The three projections of each token used in attention.

**Residual connection** — Adds a sub-layer's input to its output, easing gradient
flow.

**RLHF** — Reinforcement Learning from Human Feedback; aligns model outputs with
human preferences.

**Scaling laws** — Empirical power-law relationship between loss and
model/data/compute size.

**Self-attention** — Attention where queries, keys, and values all come from the
same sequence.

**Softmax** — Turns a vector of logits into a probability distribution.

**SVD** — Singular Value Decomposition; factorizes any matrix as rotate–scale–
rotate.

**Token** — A subword unit of text; the atomic input to an LLM.

**Transformer** — The architecture (attention + FFN + residual + LayerNorm,
stacked) underlying modern LLMs.

**Vector space** — Set of vectors closed under addition and scalar
multiplication.

---

[← Cheat Sheet](A-linear-algebra-cheatsheet.md) · [Next: Further Reading →](C-further-reading.md)
