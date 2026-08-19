== QLoRA: Quantized LoRA <background-qlora>

QLoRA combines @background-lora with 4-bit quantization so that a larger frozen base model can fit
in GPU memory during fine-tuning @dettmers_qlora_2023. The base weights are stored in 4-bit form and
temporarily dequantized for computation, while LoRA adapters remain trainable:

$
  W' = "dequantize"(W_("4-bit")) + (alpha / r) B A.
$

Gradients update $A$ and $B$, not the quantized base weights. QLoRA uses NormalFloat 4 (NF4),
block-wise quantization, double quantization of scaling constants, and paged optimizers to manage
temporary memory spikes.

=== Benefits and settings

The original paper fine-tuned a 65-billion-parameter model on one 48 GB GPU and reported performance
comparable to 16-bit fine-tuning on its evaluated tasks @dettmers_qlora_2023. Relevant settings
include the quantization and compute datatypes, double quantization, LoRA rank and targets, batch
size, and sequence length. `bfloat16` is a common compute datatype when hardware supports it.

=== Limitations

Quantizing base weights does not remove memory used by activations, adapters, optimizer state, and
temporary values. Long contexts or large batches can still exceed device memory. Quantization may
introduce numerical or quality trade-offs, and software support varies by architecture and hardware.
Like LoRA, QLoRA freezes the base-model knowledge and may not match full fine-tuning when extensive
model changes are needed.
