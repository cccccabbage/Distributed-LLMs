== Distributed-Inference Resource Heterogeneity and Model Placement <issue-inference-resource-heterogeneity>

Distributed LLM inference can draw on devices that differ in compute throughput, available model and
KV-cache memory, and network-link capacity. A full model may not fit on a single constrained device,
but dividing it across devices introduces activation transfers between consecutive model stages.
Consequently, adding a device is not automatically beneficial: its additional compute or memory must
outweigh both the communication it introduces and the risk that its stage becomes a pipeline
bottleneck @zhangEdgeShardEfficientLLM2024 @meiHelixServingLarge2025.

Model placement, routing, and scheduling are therefore coupled decisions. A placement must satisfy
memory constraints, preserve the model's layer order, and account for the execution rate of each
stage and the bandwidth or latency of each required transfer. An equal layer split can underuse
faster devices or leave work queued behind a slow device. In throughput-oriented pipelines, the
slowest stage bounds steady-state throughput; in latency-oriented execution, the relevant cost is
the end-to-end sum of computation and activation-transfer delays.

These costs can be estimated from profiling, but the estimate may age poorly as devices experience
contention, thermal throttling, failure, or changing network conditions. A practical system must
therefore decide whether to optimize offline from measured capacities, adapt its placement or
routing online, or accept the overhead and instability of frequent reconfiguration. This issue is
about serving-system resource allocation; it is distinct from the client-training configuration
problem in @issue-resource-heterogeneity[Resource Heterogeneity and Configuration Adaptation].
