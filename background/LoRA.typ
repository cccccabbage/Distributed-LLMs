== LoRA: Low-Rank Adaptation <background-lora>

LoRA is a parameter-efficient fine-tuning (PEFT) method that freezes a pretrained model and trains
small adapter parameters instead of updating all model weights @huLoRALowRankAdaptation2021. For a
layer with weight matrix $W$, it represents the learned update with two low-rank matrices:

$
  W' = W + (alpha / r) B A.
$

$W$ remains frozen, $A$ and $B$ are trained, $r$ is the adapter rank, and $alpha$ scales the update.
For a $4096 times 4096$ matrix, a rank-16 adapter trains $131,072$ values rather than roughly $16.8$
million values for a dense update.

=== Benefits and settings

LoRA lowers training-memory requirements and produces small, portable adapter files. A merged
adapter can add little or no inference latency. The original paper reported up to $10,000$ times
fewer trainable parameters and roughly three times lower GPU memory use than full fine-tuning for
GPT-3; these are results for the tested settings, not universal guarantees
@huLoRALowRankAdaptation2021.

Important choices include the rank, scaling factor, target modules, and dropout. Larger ranks offer
more capacity but consume more memory and storage. Attention projections are common targets, though
other linear layers may also be adapted. There is no universally best configuration.

=== Limitations

LoRA does not shrink the frozen base model. A low rank can underfit, while an unnecessarily high
rank weakens the efficiency advantage. Adapters are generally tied to a specific architecture and
base checkpoint, and PEFT cannot compensate for unsuitable training data. Tasks requiring broad,
high-capacity changes may still favor full fine-tuning.
