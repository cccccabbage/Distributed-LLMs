== Pipeline Bubbles and Stage Utilization in Distributed Inference <issue-pipeline-bubbles-utilization>

Pipeline parallelism divides consecutive model layers into stages and moves work between them. A
pipeline bubble is an interval in which a stage has no useful work because the next required
activation or synchronization event is not ready. Autoregressive decoding makes this especially
visible: causal dependencies require the model to finish the current token before the next token can
start, so a single request provides little independent work. This utilization problem recurs across
distributed inference systems with different scheduling mechanisms
@liuFlowSpecContinuousPipelined2025 @heSiPipeBridgingCPUGPU2025
@macarioModelDistributedInferenceLarge2025 @yeJupiterFastResourceEfficient2025.

Pipeline execution has three broad phases. During fill, work moves from the first stage toward the
last and downstream stages are initially idle. In steady state, independent microbatches or requests
can occupy different stages concurrently. During drain, work leaves the pipeline and stages again
become idle. With one request, these phases recur around successive token steps and throughput is
limited by the end-to-end pass. With multiple independent requests, the pipeline can amortize fill
and drain and improve aggregate throughput, although each request retains its causal token
dependency.

Stage utilization also depends on placement. The slowest stage bounds steady-state throughput, and
uneven layer placement can leave faster stages waiting behind a slow one. Device and link variation
therefore couples utilization to the resource heterogeneity and placement concerns in
@issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity and Model
  Placement]. Activation transfer and synchronization add further costs described in
@issue-communication-cost[Communication Cost and Synchronization].

Bubbles may come from more than transformer computation. CPU preparation, sampling at the final
stage, metadata exchange, buffer allocation, activation transfer, and synchronization can all delay
the next stage. Prefill and decode have different available work and dependency patterns; the
relationship between prompt processing, autoregressive decoding, and layer-local KV state is
described in @background-prefill-decode-kv-cache[Prefill, Decode, and the Key-Value Cache]. These
inference bubbles should be distinguished from training pipeline bubbles: training can often expose
independent microbatches and backward work, while decode is constrained by the causal sequence of
generated tokens.

Broad solution families include creating more independent work through multiple requests or
finer-grained decomposition, using speculative work that can be verified or discarded, overlapping
CPU and GPU activity, transferring activations asynchronously, and scheduling work with awareness of
communication and stage capacity. Each family trades extra memory, computation, coordination, or
possible quality changes against higher utilization; it does not remove the underlying causal
dependency for an individual ordinary decode.
