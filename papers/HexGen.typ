== HexGen: Generative Inference of Large Language Models over Heterogeneous Environments <paper-hexgen>

=== Summary

HexGen@jiang_hexgen_2024 is a serving system for generative LLM inference across GPUs and network
links with uneven compute performance, memory capacity, bandwidth, latency, and physical location.
Its central contribution is asymmetric tensor and pipeline parallelism: rather than assigning an
equal number of layers or an equal tensor-parallel (TP) degree to every pipeline stage, it assigns
work in proportion to the capabilities of the selected hardware. HexGen searches for a
memory-feasible deployment that maximizes the proportion of requests meeting a latency service-level
objective (SLO).

The paper evaluates service of Llama-2 70B and reports that, for the same budget, HexGen can meet
latency deadlines up to $2.3 times$ lower or tolerate request rates up to $4 times$ higher than its
evaluated homogeneous baseline.

=== Issues Addressed

HexGen instantiates @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] in a cross-datacenter serving setting. An equal partition is
poorly suited to mixed GPUs: a weaker or memory-constrained device can become the bottleneck, while
a stronger device is underused. Moreover, TP requires frequent synchronization, so putting a TP
group across a slow or high-latency link can make an otherwise feasible placement inefficient.

The paper's particular problem is to jointly choose model replicas, their GPU membership, pipeline
stages, layer counts, and per-stage TP degrees while accounting for compute, communication, and
memory limits. These coupled choices create a combinatorial optimization problem, rather than a
simple equal layer split.

=== Method

HexGen represents each model replica as an ordered sequence of pipeline stages. A stage can be a TP
group, and stages may differ both in their TP degree and in the number of consecutive Transformer
layers they host. Thus, faster groups can receive more layers while smaller or slower groups receive
less work. The deployment must fit model parameters, KV cache, intermediate activations, and runtime
buffers on every participating GPU.

Candidate configurations are assessed with computation and communication cost models. Computation
cost depends on the GPU characteristics, assigned layers, prompt length, and generated length.
Communication cost combines frequent intra-layer TP transfers with inter-stage pipeline activation
transfers, using the relevant message sizes, bandwidths, and latencies. The objective estimates SLO
attainment under the inference workload. HexGen uses an AlpaServe-based simulator for this
evaluation.

The search has two levels. For a fixed GPU group that will host one replica, dynamic programming
selects its pipeline stages, consecutive layer allocation, and TP degree under the constrained
search space. TP groups are limited to same-type GPUs on one machine, avoiding TP traffic over
unfavorable links. A genetic search then partitions the overall GPU pool into replicas. It begins
from network-aware groupings derived with k-means clustering and explores alternatives through
merge, split, and swap mutations; each resulting group is optimized by the dynamic-programming
stage.

For communication between adjacent TP-backed pipeline stages, HexGen designates a leader in each TP
group. Leaders exchange inter-stage data, after which the receiving leader distributes it within its
local group. This avoids every member of a TP group directly communicating across a potentially slow
inter-stage link.

=== Pros and Cons

==== Pros

- Asymmetric layer allocation and TP degree directly address unequal GPU compute and memory
  capacity, enabling mixed hardware to participate without an equal-split assumption.
- The optimization jointly considers model replication, pipeline partitioning, TP configuration,
  communication, and memory feasibility rather than optimizing one of these choices in isolation.
- Restricting TP to same-machine, same-type GPUs and using leader-based inter-stage transfers are
  practical design choices for avoiding high-frequency traffic over slow links.
- The two-stage search makes a globally intractable deployment problem manageable while optimizing
  for the workload's SLO, and the paper reports sizeable improvements over its homogeneous baseline.

==== Cons

- The constrained dynamic-programming and genetic searches are heuristics; they do not guarantee a
  globally optimal allocation, and outcomes can depend on initialization, mutations, and search
  budget.
- Deployment quality depends on compute and network cost models. Contention, thermal effects, or
  changing wide-area links can make offline estimates inaccurate, while the described scheduler is
  not primarily a continuously adaptive routing mechanism.
- Restricting TP groups to identical GPUs on one machine improves practicality but excludes some
  feasible cross-machine or mixed-type configurations that could be beneficial in a particular
  environment.
- The evaluated system does not integrate advanced continuous or dynamic batching, a central source
  of efficiency in modern LLM serving; asymmetrical stages may also complicate batching.
- Evaluation centers on Llama-2 70B and selected heterogeneous GPU and network settings, so the
  reported gains may not transfer directly to other models, workloads, GPU mixes, or topologies.
