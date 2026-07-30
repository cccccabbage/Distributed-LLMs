#set document(title: "Distributed LLMs")
#set page(margin: 2cm)
#show heading.where(level: 1): set text(size: 22pt, weight: "bold")
#show heading.where(level: 2): set text(size: 18pt, weight: "bold")
#show heading.where(level: 3): set text(size: 14pt, weight: "bold")

#align(center)[#text(size: 28pt, weight: "bold")[Distributed LLMs]]

#outline(
  title: [Contents],
  depth: 2,
)

= Background Knowledge

== LoRA: Low-Rank Adaptation

LoRA@huLoRALowRankAdaptation2021 is a memory-efficient way to fine-tune a pretrained neural network,
especially a large language model or image-generation model.

Instead of updating billions of existing model parameters, LoRA freezes the original model and
trains a comparatively small set of additional parameters called an adapter. This belongs to the
broader family of parameter-efficient fine-tuning methods, or PEFT.

=== How it works

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

=== Why it is popular

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

=== Important settings

/ Rank, $r$: Determines adapter capacity. A larger rank can learn more complex changes but consumes
  more memory and storage.
/ Alpha, $alpha$: Scales the LoRA update. It is often selected relative to the rank.
/ Target modules: Specifies which model layers receive adapters. Transformer attention
  projections—such as query, key, value, and output matrices—are common targets, although modern
  setups may adapt additional linear layers.
/ LoRA dropout: Regularizes the adapter during training and can help when the dataset is small.

There is no universally best configuration. Small ranks such as $8$, $16$, or $32$ are common
starting points, but the right value depends on the model, dataset, task, and targeted layers.

=== Limitations

LoRA does not make the base model small—you still need to load the base model unless it is
separately quantized.

It may underperform full fine-tuning when a task requires broad, high-capacity changes. A rank that
is too low can underfit, while a rank that is unnecessarily high weakens the efficiency advantage.

Adapters are also tied to their base-model architecture and usually to a particular checkpoint. A
LoRA trained for one model version generally cannot simply be attached to a different model.

Finally, LoRA cannot compensate for poor training data. Small, repetitive, mislabeled, or
low-quality datasets can make a model less reliable regardless of training efficiency.

=== One terminology note

LoRA usually means Low-Rank Adaptation in machine learning.

LoRa, with a lowercase “a,” commonly means Long Range, a low-power wireless radio technology used in
IoT systems.

== QLoRA: Quantized LoRA

QLoRA@dettmersQLoRAEfficientFinetuning2023 combines LoRA with 4-bit quantization so that a much
larger base model can fit in GPU memory during fine-tuning. It keeps the pretrained model frozen and
quantized, while training LoRA adapters in higher precision.

QLoRA therefore changes *how the base model is stored during training*, rather than changing the
low-rank-adapter idea introduced by LoRA.

=== How it works

For each linear layer, QLoRA stores the frozen base weights in 4-bit form. During the forward and
backward passes, the weights are temporarily dequantized to a computation datatype, commonly
`bfloat16`, and the LoRA update is added:

$
  W' = "dequantize"(W_("4-bit")) + (alpha / r) B A
$

Gradients flow through the dequantized base model into $A$ and $B$, but the 4-bit base weights are
not updated. This preserves the parameter-efficient training behavior of LoRA while substantially
reducing the memory needed to hold the pretrained model.

QLoRA uses NormalFloat 4 (NF4), a 4-bit datatype designed for weights that are approximately
normally distributed. It also uses block-wise quantization: a group of weights shares a scale value
that is used when the group is dequantized.

The scale values consume memory too. Double quantization reduces that overhead by quantizing these
quantization constants themselves. The paper also introduces paged optimizers, which use unified
memory to help absorb temporary optimizer-memory spikes rather than failing immediately with an
out-of-memory error.

=== Why it is popular

QLoRA makes it practical to fine-tune models that would otherwise not fit on an available GPU. In
the original paper, it enabled fine-tuning a 65-billion-parameter model on a single 48 GB GPU while
reporting task performance comparable to 16-bit fine-tuning. This is a result for the evaluated
models and tasks, not a guarantee for every model or setup.

It is especially useful when:

- GPU memory is the primary constraint.
- The base model is too large for conventional LoRA training at 16-bit weight precision.
- Small, portable LoRA adapters are desirable after training.

The output is still a LoRA adapter. The quantized base model and the adapter must both be available
for inference unless the deployment system supports an appropriate merge or conversion workflow.

=== Important settings

/ Quantization type: NF4 is the QLoRA paper's recommended 4-bit datatype for normally distributed
  pretrained weights. Other 4-bit schemes exist, but are not automatically equivalent.
/ Compute datatype: `bfloat16` is commonly preferred when the hardware supports it; otherwise,
  `float16` may be used with more care around numerical stability.
/ Double quantization: Further lowers memory use by quantizing the quantization constants. It is
  normally enabled for QLoRA training.
/ Target modules, rank, alpha, and dropout: These are the same LoRA design choices described above;
  QLoRA does not eliminate the need to tune them for the task.
/ Batch size and sequence length: Activation memory still grows with these settings. Quantizing the
  base weights does not make all training memory costs disappear.

=== Limitations

QLoRA greatly reduces base-model memory, but it does not remove all hardware requirements.
Activations, LoRA parameters, optimizer states, and temporary dequantized values still use memory;
long contexts and large batches can therefore still cause out-of-memory errors.

Quantization can introduce numerical or quality trade-offs, and support differs across model
architectures, GPU generations, and software libraries. A configuration that works for one model may
need changes for another.

Like LoRA, QLoRA freezes the base-model knowledge. It may not match full fine-tuning when a task
needs broad changes throughout the model, and it cannot fix poor or unsuitable training data.

=== One terminology note

The “Q” in QLoRA refers to quantization. QLoRA is commonly used to mean the training recipe of a
quantized frozen base model plus trainable LoRA adapters, not a different replacement for LoRA's
low-rank update.
= Papers

= References

#bibliography("reference.bib", style: "ieee", title: none)
