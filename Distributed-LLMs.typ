#set document(title: "Distributed LLMs")
#set page(margin: 2cm)

= Distributed LLMs

== Background Knowledge

=== LoRA: Low-Rank Adaptation

LoRA@huLoRALowRankAdaptation2021 is a memory-efficient way to fine-tune a pretrained neural network,
especially a large language model or image-generation model.

Instead of updating billions of existing model parameters, LoRA freezes the original model and
trains a comparatively small set of additional parameters called an adapter. This belongs to the
broader family of parameter-efficient fine-tuning methods, or PEFT.

==== How it works

Suppose a layer contains a large weight matrix $W$. Full fine-tuning learns a complete update:

$
  W' = W + Delta W
$

LoRA assumes that the useful change $Delta W$ can be approximated by multiplying two much smaller
matrices:

$
  W' = W + (alpha / r) B A
$

Here:

- $W$ remains frozen.
- $A$ and $B$ are trained.
- $r$, the rank, is small compared with the dimensions of $W$.
- $alpha$ controls the strength of the adapter.

For example, modifying a $4096 times 4096$ matrix directly involves about 16.8 million parameters. A
rank-16 LoRA update needs:

$
  4096 times 16 + 16 times 4096 = 131,072
$

That is about 128 times fewer trainable parameters for that matrix.

A useful analogy is that full fine-tuning rewrites an entire textbook, while LoRA attaches a compact
set of amendments telling the model how to behave differently.

==== Why it is popular

LoRA generally offers:

- Lower training-memory requirements, especially for optimizer states and gradients.
- Small adapter files, often far smaller than a complete model checkpoint.
- Multiple specializations from one base model—for example, separate legal, customer-support,
  coding, or writing-style adapters.
- Little or no added inference latency when merged, because the learned update can be incorporated
  into the original weights before deployment.

In its original experiments, the LoRA paper reported up to $10,000$ times fewer trainable parameters
and roughly three times lower GPU-memory requirements than full fine-tuning for GPT-3, while
achieving comparable or better results on the tested tasks. These are experimental results, not
universal guarantees.

==== Important settings

/ Rank, $r$: Determines adapter capacity. A larger rank can learn more complex changes but consumes
  more memory and storage.
/ Alpha, $alpha$: Scales the LoRA update. It is often selected relative to the rank.
/ Target modules: Specifies which model layers receive adapters. Transformer attention
  projections—such as query, key, value, and output matrices—are common targets, although modern
  setups may adapt additional linear layers.
/ LoRA dropout: Regularizes the adapter during training and can help when the dataset is small.

There is no universally best configuration. Small ranks such as $8$, $16$, or $32$ are common
starting points, but the right value depends on the model, dataset, task, and targeted layers.

==== Limitations

LoRA does not make the base model small—you still need to load the base model unless it is
separately quantized.

It may underperform full fine-tuning when a task requires broad, high-capacity changes. A rank that
is too low can underfit, while a rank that is unnecessarily high weakens the efficiency advantage.

Adapters are also tied to their base-model architecture and usually to a particular checkpoint. A
LoRA trained for one model version generally cannot simply be attached to a different model.

Finally, LoRA cannot compensate for poor training data. Small, repetitive, mislabeled, or
low-quality datasets can make a model less reliable regardless of training efficiency.

==== One terminology note

LoRA usually means Low-Rank Adaptation in machine learning.

LoRa, with a lowercase “a,” commonly means Long Range, a low-power wireless radio technology used in
IoT systems.

== Papers

== References

#bibliography("reference.bib", style: "ieee", title: none)
