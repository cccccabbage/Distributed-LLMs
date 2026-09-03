== ThunderServe: High-Performance and Cost-Efficient LLM Serving in Cloud Environments <paper-thunderserve>

=== Summary

ThunderServe@jiangThunderServeHighperformanceCostefficient2025 is an LLM serving system for
heterogeneous cloud GPU environments. Available devices may differ in GPU model, compute capability,
memory capacity, memory bandwidth, network connectivity, and rental cost. The system's central
observation is that prefill is relatively compute-bound, whereas decode is relatively
memory-bandwidth-bound. It therefore disaggregates the two phases and assigns GPU groups according
to their hardware strengths.

ThunderServe jointly decides how GPUs are grouped, which groups perform prefill or decode, which
parallelism strategy each group uses, and how requests are routed between phases. It also reduces
the cost of transferring the resulting KV cache through compression and supports lightweight
rescheduling when resources or workloads change. Under approximately equivalent GPU rental budgets,
the authors report up to $2.1 times$ higher throughput and $2.5 times$ tighter latency deadlines
than the evaluated baselines.

=== Issues Addressed

ThunderServe addresses @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] in cloud environments where GPU capabilities and network links
are uneven. A strategy designed for a homogeneous cluster can waste fragmented resources or make a
weaker device the bottleneck. The paper also treats network quality as part of resource allocation:
PCIe or Ethernet links can be substantially slower than NVLink or InfiniBand, and different pairs of
GPU groups can have different bandwidths and communication latencies.

The system uses the phase distinction described in @background-prefill-decode-kv-cache[Prefill,
  Decode, and the Key-Value Cache]. Prefill processes the prompt and is relatively compute-oriented,
while decode generates tokens one at a time and is more strongly affected by memory bandwidth. Using
one GPU type and one deployment for both phases is therefore not necessarily cost-efficient.
Disaggregation introduces a second problem, since the KV cache produced by prefill must be
transferred to a decode group. ThunderServe must optimize computation and communication together,
including the choice of prefill-to-decode routes.

The paper further addresses changing cloud resources and workloads, including GPU availability,
request rate, prompt length, output length, and the prefill-to-decode workload ratio. Rebuilding a
deployment can require moving and reloading model weights, so adapting the service without a
complete reconstruction is an explicit concern.

=== Method

ThunderServe formulates serving configuration as a two-level optimization process. The upper level
constructs GPU groups and assigns their prefill or decode roles. The lower level selects tensor and
pipeline parallelism for each group and determines request routing. A group corresponds roughly to
one model replica, but its GPUs need not be identical. For example, a group may combine two A40 GPUs
with two A5000 GPUs. Initial grouping is influenced by network connectivity, since GPUs with faster
internal connections are generally better suited to the same group.

The scheduler searches the configuration space with four group operations:

- *Flip* changes a group's role from prefill to decode or from decode to prefill.
- *Split* divides one group into multiple groups.
- *Merge* combines groups into one group.
- *Move* transfers a GPU from one group to another.

GPU grouping and phase assignment are computationally difficult and are related in the paper to the
NP-hard Job Shop Scheduling Problem. ThunderServe consequently uses tabu search. It evaluates nearby
configurations produced by the group operations, records recently explored changes in a tabu list,
and retains promising configurations. This search seeks a good configuration within practical
scheduling time, rather than guaranteeing a global optimum.

After grouping, phase assignment maps each group to prefill or decode according to its hardware
characteristics. Groups with stronger compute capability are suitable candidates for prefill, while
groups with stronger memory bandwidth are suitable candidates for decode. The phases can therefore
use different GPU groups and different parallelism configurations.

For tensor parallelism (TP), ThunderServe divides operations within model layers across GPUs. TP
requires frequent synchronization, so the scheduler restricts it to GPUs for which the required
communication is practical and generally avoids slow inter-node links. It also generally uses GPUs
of the same type within a TP group instead of mixing substantially different types.

Pipeline parallelism (PP) places different model layers on different GPUs. ThunderServe supports
non-uniform pipeline layer allocation, giving more layers to stronger GPUs and fewer layers to
weaker ones. This can balance execution time across a heterogeneous group instead of forcing an
equal layer split. Communication bandwidth is also considered when choosing pipeline paths.

The optimization objective is phase-specific. For prefill, ThunderServe approximately minimizes
processing latency because the phase is compute-oriented and gains relatively little from increasing
the batch after compute saturation. For decode, it approximately maximizes throughput because
batching can improve efficiency while the workload remains strongly influenced by memory bandwidth.

Once phase roles are selected, ThunderServe estimates each possible prefill-to-decode pairing using
prefill execution time, decode execution time, KV-cache size, network bandwidth, and communication
latency. It then uses linear programming to determine request-routing ratios across the pairings.
The objective is overall service-level objective attainment, rather than independently choosing a
route for each pair.

Disaggregation requires KV-cache transfer. ThunderServe compresses the cache during transmission,
using quantization to reduce the communication volume and dequantization before decode. The
transmitted representation is used primarily for transfer and need not be the representation used by
normal decode computation. In the reported experiments, compression substantially reduces the share
of inference time spent transferring the cache, with only small accuracy differences on the
evaluated tasks.

ThunderServe also provides lightweight rescheduling for changes in resources or workload. This mode
primarily changes prefill/decode role assignment and request routing while preserving more of the
existing grouping and parallelism configuration. It can adapt faster because it avoids expensive
model movement and reloading, but it may produce a less optimal deployment than complete
rescheduling.

ThunderServe differs from @paper-hexgen[HexGen], which keeps prefill and decode colocated in each
replica and focuses on asymmetric TP and PP placement. It is closer to @paper-hexgen-2[HexGen-2] in
disaggregating the phases and making KV-cache movement explicit, but uses tabu search with flip,
split, merge, and move operations plus linear-programming routing. In contrast, @paper-helix[Helix]
formulates heterogeneous serving as a max-flow problem over layer placement and network paths.
ThunderServe combines network-aware placement with phase-specific objectives and lightweight role
reassignment.

=== Pros and Cons

==== Pros

- It makes better use of fragmented cloud fleets by allowing heterogeneous GPU groups and matching
  compute-oriented hardware to prefill and memory-bandwidth-oriented hardware to decode.
- It jointly considers GPU grouping, phase assignment, TP and PP configuration, request routing, and
  KV-cache communication. This can avoid configurations that look efficient locally but create a
  system-level bottleneck.
- Non-uniform pipeline layer allocation lets stronger GPUs host more layers, which is more suitable
  for mixed devices than an equal split.
- Network-aware routing and KV-cache compression make prefill/decode disaggregation more practical
  when links between groups are uneven or constrained.
- Lightweight rescheduling can react to resource and workload changes without necessarily rebuilding
  the entire deployment.
- The paper reports up to $2.1 times$ higher throughput and $2.5 times$ tighter latency deadlines at
  approximately equivalent GPU rental budgets than the evaluated baselines.

==== Cons

- Tabu search is heuristic and does not guarantee a globally optimal grouping, phase assignment, or
  deployment. Its result depends on the search procedure and the accuracy of the performance model.
- Scheduling overhead grows with the heterogeneous configuration space. The supplied notes discuss
  evaluation on clusters of up to a few dozen GPUs, so behavior at substantially larger production
  scales needs further validation.
- Lightweight rescheduling is faster partly because it avoids regrouping GPUs, changing parallelism,
  and reloading models. Its resulting configuration can therefore be worse than one found by a full
  rescheduling process.
- Compression reduces, but does not remove, KV-cache communication. Extremely slow links can still
  dominate latency and erase the benefit of disaggregation.
- KV-cache quantization is lossy. The paper reports small quality degradation on its evaluated
  tasks, but the effect may vary with model architecture, dataset, context length, generation
  length, and application requirements.
- The reported evaluation covers selected models, workloads, GPU configurations, and cluster sizes.
  The results do not establish the same gains for larger models, mixture-of-experts models, much
  longer contexts, different request distributions, or more variable cloud networks.
- The system depends on accurate estimates of GPU execution time, communication bandwidth, KV-cache
  transfer time, and workload characteristics. It also introduces more implementation and
  operational complexity than a simple homogeneous deployment.
