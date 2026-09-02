== Model-Distributed Inference for Large Language Models at the Edge (MDI-LLM) <paper-mdi-llm>

=== Summary

MDI-LLM@macarioModelDistributedInferenceLarge2025 is an edge-inference framework that distributes a
Transformer's consecutive layers across several resource-constrained devices. A starter node
receives a prompt, performs embedding and output processing, and coordinates a recurrent path
through the layer partitions; secondary nodes process their local layers and forward intermediate
activations. The system's recurrent pipeline parallelism circulates several independent generation
requests through this path, aiming to keep stages occupied during autoregressive decoding. It also
keeps a separate, layer-local KV cache for each active request. On three NVIDIA Jetson TX2 devices,
the paper evaluates NanoLlama (304M parameters) and TinyLlama (1.1B parameters); the latter did not
fit on one tested device but could execute when its layers were distributed across two or three.

=== Issues Addressed

The paper instantiates @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] for low-power edge devices. Its immediate constraint is that a
single device may not have enough memory to hold an LLM, whereas assigning consecutive layer groups
to several devices pools their available memory. The paper contrasts this with data parallelism,
which still requires every participant to store a full model, and tensor parallelism, which needs
frequent synchronization that may be impractical on edge links.

Layer partitioning alone does not solve utilization during the decode phase described in
@background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache]. A single request must
complete a full causal model pass before its next token can begin, leaving other pipeline stages
idle. MDI-LLM specifically seeks higher aggregate throughput for several independent generation
requests; it does not remove the per-request sequential dependency between output tokens.

=== Method

MDI-LLM partitions Transformer layers among the participating devices. The starter node handles the
prompt, embeddings, its assigned layers, final output layers, next-token sampling, and coordination.
Each secondary node receives an intermediate activation, applies its consecutive layer partition,
and transfers the result to the next node. After the last partition, the activation returns to the
starter node so that its sampled token can begin the next autoregressive pass.

Recurrent pipeline parallelism injects several independent generation requests into this ring-like
path. Once the pipeline fills, different devices can execute different requests at the same time.
Its steady-state throughput is bounded by the slowest partition, so high utilization requires enough
active requests and approximately balanced partition processing times.

Each device maintains the attention keys and values for its own layers separately for every active
request. When a request arrives, the device selects that request's local KV cache, allowing later
token steps to process only the new token rather than recompute the full preceding context. These
per-request caches rotate operationally with requests as they circulate through the pipeline.

The implementation uses HTTP for initialization and coordination, persistent TCP/IP socket
connections for activation transfers, and separate processing, receiving, and sending threads with
FIFO queues. The reported prototype uses three Jetson TX2 devices with 8 GB shared memory each and
Gigabit Ethernet. For TinyLlama, the paper reports approximately 4.57 GB per device with two devices
and 3.26 GB per device with three; a single-device run was not possible in the tested setup.

=== Pros and Cons

==== Pros

- Partitioning model layers pools device memory, enabling a model that exceeds any individual
  device's capacity to execute across the group.
- Recurrent pipeline parallelism targets aggregate throughput: several requests can occupy different
  partitions concurrently instead of leaving downstream devices idle.
- Per-request, layer-local KV caches avoid recomputing prior context for every generated token and
  make concurrent requests compatible with the recurrent pipeline.
- The evaluation uses physical low-cost edge hardware rather than only simulated devices, and
  demonstrates the memory-per-device benefit for a 1.1B-parameter model.

==== Cons

- A single generation request still exposes pipeline bubbles and cannot bypass autoregressive token
  dependencies; the system is principally a multi-request-throughput design rather than a
  single-request-latency optimization.
- Every model pass transfers activations between devices. Bandwidth, latency, overhead, and jitter
  can therefore become bottlenecks, while the reported Gigabit Ethernet setup may not represent
  unstable wireless or mobile edge links.
- Adding devices lowers memory per device but does not compress the model or necessarily lower total
  system memory: runtime components, communication infrastructure, and per-request KV caches add
  overhead.
- Throughput depends on balanced layer partitions. The evaluation uses identical Jetson devices, so
  partitioning and scheduling for highly heterogeneous hardware remain less explored.
- The paper evaluates only three devices and 304M- and 1.1B-parameter models. Energy use, failure
  recovery, churn, activation confidentiality, dynamic repartitioning, and larger-scale deployment
  receive limited analysis in the supplied notes.
