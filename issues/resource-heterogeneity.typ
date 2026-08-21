== Resource Heterogeneity and Configuration Adaptation <issue-resource-heterogeneity>

Federated clients can differ substantially in available compute, memory, communication bandwidth,
and energy. This systems heterogeneity is distinct from the non-IID data heterogeneity discussed in
@issue-data-heterogeneity[Data Heterogeneity and Client Drift]. A single training configuration must
be inexpensive enough for the weakest participating client, which can leave stronger clients
underused; in synchronous rounds, slow configurations can also turn their clients into stragglers.

Parameter-efficient fine-tuning exposes adjustable capacity--cost trade-offs. For example,
increasing a LoRA rank increases adapter capacity but also increases trainable state, memory use,
and the amount of adapter data communicated; see @background-lora[LoRA: Low-Rank Adaptation]. Other
methods vary the amount of the base model used locally, such as its pruning ratio. Per-client
configurations can therefore better match unequal resources, but need a policy that balances task
quality, completion time, communication, and fairness across clients.

Client-specific configurations can also create incompatible parameter shapes, so an aggregation
mechanism must define a compatible representation rather than assume directly averageable updates.
Choosing the configuration itself remains a resource-allocation problem: it must decide which
capacity each client should receive while balancing system efficiency and learning quality
@liu_hlora_2025 @yao_efficient_2025.
