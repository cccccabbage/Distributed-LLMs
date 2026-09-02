== Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow <paper-helix>

=== Summary

Helix@meiHelixServingLarge2025 is an LLM-serving system for GPU clusters whose compute capacity,
memory, and network links are heterogeneous. It treats inference throughput as flow through a graph:
GPU processing and inter-GPU links impose capacities, while a request's model-layer order restricts
which transitions are valid. Helix jointly selects a model-layer placement and traffic distribution
with a mixed-integer linear program (MILP), then converts the resulting continuous flow into
discrete, per-request pipelines using Interleaved Weighted Round-Robin (IWRR) scheduling.

Across heterogeneous clusters with 24--42 GPU nodes, the authors report up to $3.3 times$ higher
throughput than their evaluated baselines, as well as up to 66% lower prefilling latency and 24%
lower decoding latency.

=== Issues Addressed

The paper instantiates @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] with a throughput-oriented cluster design. Its particular
problem is to model the interaction between layer placement, network congestion, and traffic
routing, rather than choosing each independently and creating bottlenecks elsewhere in the pipeline.

Helix also addresses the limited flexibility of fixed, disjoint pipelines. Different requests may
need to take different, potentially overlapping GPU paths to use the cluster's mixed resources
effectively. The paper is concerned with serving-system efficiency rather than federated training or
protections for private training data.

=== Method

Helix profiles each GPU's inference capacity under candidate layer assignments and measures
bandwidth between machines. For a proposed placement, it constructs a directed graph with source and
sink nodes. A split GPU node has an internal edge representing its compute capacity, and an edge
between GPUs represents network capacity. Layer assignments determine which GPU-to-GPU transitions
preserve the Transformer's execution order. The maximum source-to-sink flow approximates the
placement's achievable throughput.

The offline MILP chooses layer assignments and edge flows subject to GPU-memory, compute-capacity,
network-capacity, valid-layer-ordering, and flow-conservation constraints. It maximizes total source
flow. Thus, placement determines both the resources used by a stage and the graph on which traffic
can travel.

The max-flow solution gives target weights for each edge, but requests are indivisible. At runtime,
the coordinator uses IWRR to choose outgoing edges so that requests approximate these weights over
time, producing a pipeline for each request rather than selecting from a small fixed set. A request
keeps its selected pipeline through decoding so its KV cache remains on the corresponding GPUs.
Helix masks GPUs near a KV-cache high-water mark from new assignments and dynamically batches work
that reaches each GPU.

=== Pros and Cons

==== Pros

- The formulation jointly accounts for compute, memory, network capacity, model placement, and
  routing, enabling choices that avoid downstream congestion rather than optimizing only a local
  pipeline stage.
- Per-request, overlapping pipelines provide more flexibility than a few fixed disjoint pipelines
  and can better use mixed GPU types and network links.
- The separation between offline optimization and online IWRR scheduling gives a principled way to
  turn a global continuous-flow solution into executable request paths.
- The reported evaluation covers physical heterogeneous clusters of meaningful size and shows up to
  $3.3 times$ throughput improvement over the evaluated baselines.

==== Cons

- Solving the MILP is an expensive offline step. The supplied notes report that high-quality
  solutions for clusters of several dozen nodes can require hours, making direct global optimization
  at hundreds or thousands of nodes uncertain.
- Placement quality depends on GPU and network profiling. Runtime changes, especially on wide-area
  links, can make the planned flow distribution inaccurate.
- The throughput objective does not guarantee the lowest latency for every request; deliberately
  using slower GPUs can trade some individual decoding latency for higher cluster utilization.
- Evaluation is concentrated on 24--42-node clusters, and some geo-distributed experiments rely on
  simulation. The full system's performance under much larger deployments and real wide-area
  variability remains unproven.
- Few prior serving systems target precisely the same heterogeneous setting, so some baseline
  implementations or adaptations are not perfectly apples-to-apples.
