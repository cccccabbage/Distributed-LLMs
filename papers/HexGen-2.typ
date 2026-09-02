== HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environments <paper-hexgen-2>

=== Summary

HexGen-2@jiangHexGen2DisaggregatedGenerative2025 serves LLMs on heterogeneous GPU clusters by
disaggregating the prefill and decode phases described in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache]. The system assigns
those phases to separate replicas, transfers the KV cache between them, and jointly selects their
heterogeneous hardware placement, parallelism, and request routing.

Its optimizer combines graph partitioning, maximum-flow analysis, and maximum-flow-guided iterative
refinement to maximize throughput while respecting compute, memory, network, and KV-transfer
constraints. For OPT-30B and Llama-2 70B deployments, the paper reports up to $2 times$ and on
average $1.3 times$ higher serving throughput, and $1.5 times$ lower average inference latency, than
its evaluated state-of-the-art systems at the same budget.

=== Issues Addressed

The paper extends @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] and the phase-specific trade-offs in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache] to disaggregated
serving. Using one replica for both inference phases can let a long prompt-prefill workload compete
with latency-sensitive decoding, while a GPU well suited to compute-intensive prefill need not be
optimal for memory-bandwidth-intensive decode.

Disaggregation creates a distinct placement problem: each prefill result's KV cache must reach a
decode replica. Cache transfers can be substantial for long prompts, large models, or large batches,
and a slow link can erase the benefit of separately provisioning the two phases. The system must
therefore jointly choose replica membership, phase roles, tensor- and pipeline-parallel
configurations, and KV-cache routes instead of merely selecting the fastest GPUs.

=== Method

HexGen-2 models the GPU cluster as a graph whose nodes capture device memory, compute capability,
and memory bandwidth, and whose edges represent communication capacity. It first partitions GPUs
into model-replica groups using spectral partitioning followed by Kernighan--Lin refinement, seeking
groups with feasible memory capacity and favorable internal connectivity. The resulting groups act
as super-nodes and are assigned prefill or decode roles with attention to the links required for KV
cache transfer.

The system separately selects tensor parallelism (TP) and pipeline parallelism (PP) for each phase.
Prefill configurations emphasize compute performance and latency; decode configurations emphasize
memory-bandwidth-limited throughput and batching. Consequently, the two phases need not use the same
GPUs or parallelism strategy.

For a candidate assignment, HexGen-2 constructs a flow network. Source-to-prefill edges encode
prefill capacity, prefill-to-decode edges encode the rate at which their network connection can
transfer KV caches, and decode-to-sink edges encode decoding capacity. Maximum flow estimates
achievable serving throughput and supplies routing proportions between prefill and decode replicas.
The system then uses saturated links and underutilized resources identified by the flow solution to
rearrange replica groups or relationships, retaining refinements that improve the deployment.

=== Comparison with HexGen

HexGen-2 builds on @paper-hexgen[HexGen]'s asymmetric deployment of full-model replicas over mixed
GPUs, but changes the scheduling unit from an undivided inference replica to separately placed
prefill and decode replicas. HexGen uses dynamic programming to configure a fixed replica and a
genetic search to group GPUs. HexGen-2 uses graph partitioning plus maximum-flow-guided refinement,
and it treats KV-cache movement as an explicit capacity and routing constraint.

#table(
  columns: (1.45fr, 1.7fr, 1.7fr),
  table.header([Aspect], [HexGen], [HexGen-2]),
  [Inference phases],
  [Prefill and decode colocated in each replica],
  [Separate prefill and decode replicas],

  [Principal contribution],
  [Asymmetric TP and PP],
  [Phase-specific allocation and KV-aware disaggregation],

  [Main search approach],
  [Dynamic programming and genetic search],
  [Graph partitioning, maximum flow, and iterative refinement],

  [Network emphasis],
  [TP and pipeline-stage communication],
  [TP/PP communication plus inter-phase KV-cache transfer],

  [Routing],
  [Replica grouping is central],
  [Prefill-to-decode request and cache routing is explicitly optimized],
)

Disaggregation gives HexGen-2 more freedom to match compute-oriented and memory-bandwidth-oriented
hardware to their respective phases, and it reduces direct prefill--decode interference. In return,
it introduces KV-transfer overhead, phase coordination, and a stronger dependence on network
quality.

=== Pros and Cons

==== Pros

- Phase-specific GPU assignment can match compute-efficient devices to prefill and
  memory-bandwidth-efficient devices to decode instead of forcing a single compromise configuration.
- Separating phases can reduce interference from large prefills on ongoing decoding and thereby
  improve serving responsiveness.
- The flow network jointly exposes prefill, decode, and KV-transfer bottlenecks, so maximizing
  throughput does not treat communication as an afterthought.
- Maximum-flow-guided refinement uses observed bottlenecks in the candidate model to improve the
  initial graph partition, while retaining support for mixed GPU fleets.
- The paper evaluates both OPT-30B and Llama-2 70B and reports improvements in throughput, latency,
  and required budget over its evaluated baselines.

==== Cons

- KV-cache transfer is a fundamental overhead of disaggregation. For large caches or constrained
  links, network capacity can dominate execution and limit or reverse the expected benefit.
- The optimizer depends on cost and capacity estimates for compute, memory, TP/PP communication, and
  cache transfers; changing contention or network conditions can invalidate a static placement.
- The supplied notes report scheduling times from 4.03 minutes for 64 GPUs to 47.77 minutes for 320
  GPUs, making frequent global re-optimization less attractive for rapidly changing deployments.
- The optimal prefill-to-decode resource balance varies with prompt and output lengths. A placement
  tuned for one workload may underuse resources when the workload mix changes.
- The reported evaluation focuses on OPT-30B and Llama-2 70B. Generalization to mixture-of-experts,
  multimodal, extremely long-context, or otherwise different KV-cache workloads remains uncertain.
- Separate replicas, cache movement, routing, and synchronization add operational complexity and
  introduce additional failure-handling concerns relative to colocated inference.
