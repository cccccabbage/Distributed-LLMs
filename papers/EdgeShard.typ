== EdgeShard: Efficient LLM Inference via Collaborative Edge Computing <paper-edgeshard>

=== Summary

EdgeShard@zhang_edgeshard_2024 is a distributed-inference framework that runs a large language model
(LLM) across heterogeneous edge devices and, when beneficial, cloud resources. Rather than placing a
full model on one device, it assigns consecutive groups of Transformer layers to selected devices.
Its central contribution is jointly choosing the participating devices and layer partitions with
objectives for either single-request latency or pipeline throughput. In a physical prototype with
NVIDIA Jetson devices and an RTX 3090 server, the authors report up to 50% lower inference latency
and up to a $2 times$ throughput improvement over their evaluated baselines, including for models
too large for a single tested edge device.

=== Issues Addressed

The paper instantiates @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] for resource-constrained edge devices and optional cloud
resources. Its particular framing contrasts collaborative edge execution with cloud-only inference,
which can add network delay, consume wide-area bandwidth, and transmit user data off-device, and
with quantization, which can reduce memory needs at the cost of accuracy and may still be
insufficient for very large models.

EdgeShard's specific systems problem is to choose an ordered subset of devices and assign each a
memory-feasible consecutive layer shard, with separate objectives for individual-request latency and
pipeline throughput.

=== Method

EdgeShard profiles each layer's execution time, memory use, and activation size; it also records
each candidate device's available memory and the available network bandwidth between devices. It
uses these measurements to estimate a partition's computation and communication costs. A shard's
output activation is transferred to the device holding the next consecutive shard, and faster
devices can receive more layers than slower ones.

For latency, the framework uses dynamic programming to jointly select devices and layer boundaries
while respecting memory constraints. The objective estimates the total sequential execution time:
the sum of shard-computation costs and inter-shard activation-transfer costs. The supplied notes
report a complexity of $O(N M^2)$ for $N$ model layers and $M$ candidate devices.

For throughput, EdgeShard partitions the model into pipeline stages and aims to minimize the time of
the slowest stage, so multiple requests can occupy different shards concurrently. Its
EdgeShard-No-Bubbles scheduling variant starts later micro-batch work earlier to reduce idle time
caused by autoregressive generation dependencies. The throughput optimization jointly considers
device selection and partitioning, but the reported $O(N^2 2^M M^2)$ complexity can grow quickly
with the number of candidate devices.

To reduce direct exposure of the prompt, EdgeShard requires the source device to execute the first
layer. Subsequent devices receive its intermediate activation rather than the raw input; this is a
data-locality design choice, not a formal privacy guarantee, because activations may still reveal
input information.

=== Pros and Cons

==== Pros

- Joint device selection and unequal layer partitioning explicitly account for heterogeneous
  compute, memory, and network resources instead of using a fixed or equal split.
- The two objectives support distinct deployment needs: sequential low-latency inference and
  pipelined high-throughput serving.
- Sharing model shards lets several devices pool memory, enabling models that cannot fit on one
  tested edge device without depending solely on aggressive quantization.
- The reported results use physical heterogeneous hardware rather than simulation alone and show
  improvements over the paper's evaluated baselines.
- Retaining the first layer locally avoids directly sending the raw prompt to another participant.

==== Cons

- The throughput optimization is exponential in the number of candidate devices, which limits its
  direct scalability to larger device pools.
- Partitioning depends on offline profiling; wireless congestion, device workloads, thermal
  throttling, and disconnections can invalidate the estimated costs, while the supplied notes do not
  describe continuous re-optimization.
- Intermediate activations leave the source device, so the first-layer constraint does not protect
  against inference or disclosure by a malicious or compromised participant.
- The approach depends on every selected device remaining available. Failure recovery, replication,
  device churn, and dynamic rerouting receive limited discussion in the supplied notes.
- Evaluation is concentrated on Llama2 models and a controlled Jetson-plus-server testbed, leaving
  performance under long-context, multimodal, mobile, or highly variable network workloads open.
- Batch size and KV-cache demand influence memory and throughput, but the described optimization
  focuses on device selection and model partitioning rather than jointly optimizing those factors.
