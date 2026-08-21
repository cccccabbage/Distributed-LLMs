== Efficient Deployment of Large Language Models on Resource-constrained Devices <paper-fedspine>

=== Summary

FedSpine@yao_efficient_2025 is a federated framework for fine-tuning and deploying LLMs on
heterogeneous, resource-constrained devices. It combines frozen-base @background-lora[LoRA: Low-Rank
  Adaptation] with client-side structured pruning: LoRA lowers the cost of adaptation, while pruning
attention heads and feed-forward-network (FFN) channels reduces the deployed model's computation and
memory footprint.

Its central contribution is to choose a pruning ratio and LoRA rank separately for each client in
each federated round. The server uses Smooth Upper Confidence Bound (S-UCB), an online multi-armed
bandit algorithm, to make those choices from observed quality and completion-time outcomes, without
requiring a prior hardware-performance model. On a physical platform of 80 NVIDIA Jetson devices,
the authors report $1.4 times$--$6.9 times$ fine-tuning speedups and $0.4$--$4.5$ percentage-point
higher final accuracy than the evaluated pruning baselines at the same sparsity.

=== Issues Addressed

FedSpine uses @background-federated-learning[Federated Learning] to adapt a model using private,
distributed client data without centralizing raw examples. It also faces the non-IID-data effects
described in @issue-data-heterogeneity[Data Heterogeneity and Client Drift]. Data locality is not a
complete privacy guarantee: model updates can still leak information, and the paper does not make
secure aggregation, differential privacy, or defenses against malicious updates a central
contribution.

The paper's additional systems concern is that LoRA reduces the trainable state but does not reduce
the frozen base model used for inference. FedSpine instantiates
@issue-resource-heterogeneity[Resource Heterogeneity and Configuration Adaptation] by selecting a
pruning ratio and adapter rank per device. It therefore targets the joint accuracy--training-cost--
inference-cost trade-off while reducing straggler effects through per-device configurations.

=== Method

At round $t$, the server selects a configuration $(p_i^t, r_i^t)$ for each client $i$, where $p_i^t$
is its pruning ratio and $r_i^t$ is its LoRA rank. It distributes the current LoRA state and that
configuration. The frozen pretrained base model remains on the client.

Each client estimates the importance of model structures using gradients already produced for its
LoRA factors, avoiding full gradients for frozen base weights. It smooths the estimates over local
iterations, then removes the least-important complete attention heads and FFN channels. Grouping the
dependent query, key, value, and output projections, and the LLaMA gate, up, and down projections,
keeps tensor dimensions valid after pruning. Rather than immediately applying final sparsity,
FedSpine iterates pruning and local LoRA recovery so the adapter can compensate after each pruning
stage.

After local fine-tuning, clients return their LoRA updates and recorded state information. Because
client ranks can differ, their LoRA matrices need not have compatible shapes. FedSpine reconstructs
the updates into compatible representations and uses a heterogeneity-aware weighted aggregation
strategy.

S-UCB treats a pruning-ratio/rank pair as a bandit action. It partitions and progressively refines
the continuous configuration space, then balances configurations with high observed reward against
uncertain alternatives. Its reward incorporates local-loss improvement, pruning progress, estimated
LoRA-update importance, and the client's completion time relative to the other clients. Thus, the
server repeats: select $(p_i^t, r_i^t)$, distribute LoRA state, prune and fine-tune locally, and
aggregate the returned updates.

=== Pros and Cons

==== Pros

- Combining LoRA with structured pruning addresses adaptation efficiency and practical inference
  efficiency together. Unlike unstructured sparsity, removing whole heads and channels can deliver
  acceleration without specialized sparse hardware.
- Per-client S-UCB configurations explicitly address unequal device and network capabilities without
  requiring an analytical model of their performance.
- Pruning is downstream-task-aware because its importance estimates use gradients from local LoRA
  training rather than a fixed magnitude-only rule.
- The physical-device evaluation is substantial systems evidence: the platform contains 30 Jetson
  TX2, 40 Jetson NX, and 10 Jetson AGX devices. At 50% RoBERTa sparsity, the paper reports inference
  time of 51.4% of the HETLoRA reference time.

==== Cons

- The method improves an accuracy--efficiency trade-off rather than matching an unpruned model by
  default. For LLaMA-7B at 30% sparsity, the reported average task score is 54.5 for FedSpine versus
  63.4 for the unpruned HETLoRA reference.
- Training-memory savings are not universal. At 50% RoBERTa sparsity, FedSpine uses 114.5% of the
  HETLoRA reference's training memory, indicating overhead from pruning and adaptive control.
- Evidence at the largest model scale is limited: the LLaMA-7B evaluation uses Dolly-15K across
  eight heterogeneous Jetson AGX devices, whereas the 80-device platform is used mainly for the
  RoBERTa experiments. The all-Jetson fleet also leaves performance on more diverse hardware
  uncertain.
- LoRA-gradient importance estimation, dependency-aware pruning, per-device bandit control, dynamic
  ranks, and heterogeneous-rank aggregation make the system more complex to implement and tune than
  standard federated LoRA.
- The pruning/fine-tuning interval is an additional sensitive choice. The paper finds 20 local
  fine-tuning iterations most effective among its tested values; shorter intervals limited recovery,
  while longer ones increased local drift.
