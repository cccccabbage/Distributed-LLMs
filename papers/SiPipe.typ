== SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference <paper-sipipe>

=== Summary

SiPipe@heSiPipeBridgingCPUGPU2025 is an LLM inference system that improves the efficiency of
pipeline parallelism (PP) by treating the CPU and GPU as a heterogeneous execution system. Its
central idea is to use underutilized CPU resources for auxiliary work so that GPUs spend more time
executing the model. The system targets three sources of pipeline bubbles: extra sampling work on
the final stage, CPU-side preparation that serializes with GPU execution, and synchronization and
metadata overhead during inter-stage communication.

SiPipe addresses these bubbles with CPU Sampling, a Token-Safe Execution Model (TSEM), and
Structure-Aware Transmission (SAT). The design improves utilization by hiding work around
transformer execution rather than by changing individual transformer kernels.

=== Issues Addressed

@issue-pipeline-bubbles-utilization[Pipeline Bubbles and Stage Utilization in Distributed Inference]
This is an inference-side instance of the stage bottleneck and activation-transfer concerns
described in @issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity
  and Model Placement]. SiPipe focuses on bubbles that remain even when the model layers are
otherwise evenly distributed.

The first bubble is load imbalance at the final stage. Other stages primarily run a transformer
forward pass, while the final stage also processes logits and samples the next token. Sampling may
include temperature scaling, repetition, frequency, and presence penalties, softmax, top-k or top-p
filtering, and token selection. This additional work can make the last stage a bottleneck and cause
other stages to wait. SiPipe’s CPU Sampling addresses this paper-specific final-stage imbalance.

The second bubble is an intra-stage CPU preparation gap. Before a GPU forward pass, the CPU prepares
attention metadata, input buffers, tensor information, and other execution metadata. CUDA Graphs
reduce repeated launch overhead but require stable tensor addresses. Reusing a buffer while the GPU
is reading it creates a memory hazard, so a straightforward implementation serializes preparation
and execution. This serialization creates execution gaps before the GPU can begin its next forward
pass.

The third bubble is inter-stage communication overhead. A receiver may otherwise wait for metadata,
deserialize it, allocate buffers, receive tensor data, and only then begin computation. Across
decoding iterations, tensor values change but their structure, including names, shapes, dtypes, and
devices, is usually stable. Repeating this structural exchange and allocation therefore adds a
synchronization cost around the useful transfer. TSEM and SAT provide the paper-specific remedies.

=== Method

==== CPU Sampling

SiPipe removes sampling from the critical path of the final GPU stage. The GPU can start work on the
next microbatch after producing the logits, while the CPU samples the previous result
asynchronously. This reduces the load imbalance between stages without changing the transformer
architecture or model weights.

Moving an unmodified GPU sampler to the CPU would still be expensive for vocabularies with more than
100,000 tokens. SiPipe therefore changes the sampling data layout and update process. If standard
logits have shape $Z in RR^(B times V)$ for batch size $B$ and vocabulary size $V$, the relevant
sampling information is organized in a column-oriented form with shape $Z^T in RR^(V times B)$. This
layout makes token-associated updates more efficient.

The sampler maintains state incrementally. When a sequence produces a new token, it updates the
state associated with that token, such as its frequency, instead of reconstructing statistics from
the entire generated sequence. This is useful for frequency, presence, and repetition penalties.
SiPipe also preallocates sampling buffers and reuses memory to avoid repeated allocation and data
reconstruction. These changes make CPU sampling efficient enough to overlap with GPU execution.

==== Token-Safe Execution Model

TSEM overlaps CPU preparation for iteration $i+1$ with GPU computation for iteration $i$:

$
  "CPU"_(i+1) parallel "GPU"_i
$

The model uses multiple versions of execution buffers. While the GPU reads one buffer for the
current iteration, the CPU prepares another buffer for the next iteration; the roles are then
swapped. This preserves the stable addresses expected by CUDA Graphs without allowing CPU writes to
overwrite data still in use by the GPU.

TSEM separates the CPU executor, GPU executor, communicator, queues, and progress or state
indicators. Together, these components track when a buffer is safe to read, write, or reuse,
allowing the CPU to work ahead without corrupting execution inputs.

==== Structure-Aware Transmission

SAT reduces synchronization before inter-stage activation transfer by learning tensor structure on
the first transmission. The receiver records the number of tensors and their names, shapes, dtypes,
and devices. On later iterations, it uses the known structure and current batch information to
predict the required buffers, preallocate them, and post an asynchronous receive before the sender's
output is ready.

The sender can then transmit tensor data asynchronously when computation finishes, without another
metadata exchange or allocation step. This allows communication to overlap with computation and
reduces the effective pipeline bubble. SiPipe is described as a plugin for vLLM, making the
techniques applicable to an existing serving stack.

=== Pros and Cons

==== Pros

- The three techniques map directly to three distinct bottlenecks: CPU Sampling addresses
  final-stage load imbalance, TSEM addresses CPU--GPU serialization, and SAT addresses inter-stage
  transfer synchronization.
- SiPipe uses otherwise underutilized host CPU resources for sampling and preparation, allowing GPU
  execution and auxiliary work to proceed concurrently.
- It is primarily a systems optimization and does not require changes to the transformer
  architecture, model weights, training procedure, or attention algorithm.
- Buffer versioning preserves CUDA Graph compatibility while enabling CPU preparation to overlap
  with GPU execution, and SAT avoids repeatedly communicating stable tensor structure.
- Implementing the system as a vLLM plugin improves its practical relevance for existing serving
  deployments.

==== Cons

- CPU Sampling depends on sufficient spare CPU capacity. Tokenization, networking, request
  processing, scheduling, and KV-cache management may already contend for those resources.
- The approach may be less effective on machines with few or slow CPU cores, limited memory
  bandwidth, or high CPU contention.
- Incremental sampling benefits from stable decoding batches. Highly dynamic scheduling or frequent
  batch changes can reduce the benefit of retaining and updating sampling state.
- SAT relies on predictable intermediate tensor structures. Highly dynamic model or runtime
  structures make structure caching and preallocation harder to apply.
- SiPipe primarily targets system-induced bubbles and cannot completely remove imbalance caused by
  model computation itself, such as dynamic expert workloads in mixture-of-experts models.
- The approach is most useful for large models that require multiple GPUs or nodes and benefit from
  PP; smaller models may be adequately served with simpler parallelism strategies.
- The main focus is decoding efficiency and time per output token, not time to first token. Queuing,
  scheduling, admission control, and prefill remain important influences on TTFT.
- TSEM and SAT add buffers, state tracking, queues, asynchronous communication, and CUDA Graph
  coordination. For very fast forward passes, this coordination overhead may become significant.
