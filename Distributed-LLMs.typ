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

== Federated Learning

Federated Learning (FL) trains a shared model while keeping each participant's raw data on its own
client. Clients may be phones, hospitals, companies, or edge devices. A coordinating server sends a
model to selected clients; each client trains locally and returns a model update; the server
aggregates those updates into the next global model. The classic practical algorithm is Federated
Averaging (FedAvg)@mcmahanCommunicationEfficientLearningDeep2023.

FL provides data locality, not automatic privacy. Updates and metadata can still leak information,
and malicious clients can try to poison training. Privacy and security mechanisms need to be chosen
for the actual threat model.

=== The FedAvg loop

Let client $k$ hold $n_k$ examples, and let $n = sum_k n_k$. At round $t$, the server has model
parameters $w_t$.

1. The server selects available clients and broadcasts $w_t$.
2. Each selected client runs local optimization for $E$ steps on its own data, producing
  $w_(t+1)^k$.
3. Clients send their parameters or parameter changes; raw training examples are not normally sent.
4. The server forms a data-size-weighted average:

$
  w_(t+1) = sum_k (n_k / n) w_(t+1)^k
$

A client can instead send $Delta w_k = w_(t+1)^k - w_t$, which the server averages and adds to the
current model. More local steps can reduce costly communication rounds, but they can also make
client models drift apart when their data differs.

=== The learning objective

FL usually minimizes a weighted collection of client losses:

$
  min_w F(w) = sum_k (n_k / n) F_k(w)
$

$F_k(w)$ is client $k$'s local loss. Weighting by $n_k$ means clients with more examples have more
influence. Equal-client weighting or fairness-weighted objectives are possible, but optimize a
different target.

=== Main challenges

/ Non-IID data: Client datasets are not independent and identically distributed. For example,
  hospitals can serve different populations and users can write in different languages or styles.
  Their local optima differ, slowing or destabilizing simple averaging.
/ System heterogeneity: Devices differ in compute, battery, connectivity, and dataset size. Dropouts
  and slow clients (*stragglers*) can bias which data influences a round.
/ Communication: Sending repeated large updates is expensive and is often the main bottleneck.
/ Privacy and security: Update leakage, membership inference, data reconstruction, poisoning, and
  backdoors remain concerns even when raw examples stay local.
/ Evaluation: A good global average can hide poor performance on individual clients or minority
  groups. Report global, per-client, and worst-group results when possible.

=== Privacy and security distinctions

/ Secure aggregation: A cryptographic protocol that lets the server obtain an aggregate without
  observing every individual client update. It does not by itself stop leakage from the final model.
/ Differential privacy (DP): Clipping and calibrated noise limit the influence of one record or
  client under stated assumptions, yielding a formal privacy guarantee at a measurable utility cost.
/ Robust aggregation: Clipping, anomaly detection, and robust aggregators can limit extreme or
  malicious updates. Their value depends on the assumed attacker and is not universal protection.

A complete FL design should say who is trusted, what is observable, whether clients may be
malicious, and which risk is in scope: reconstruction, membership inference, poisoning, or
backdoors.

= Papers

#include "paper-notes/FedIT.typ"

= References

#bibliography("reference.bib", style: "ieee", title: none)
