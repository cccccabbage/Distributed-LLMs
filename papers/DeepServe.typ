== DeepServe: Serverless Large Language Model Serving at Scale <paper-deepserve>

=== Summary

DeepServe@huDeepServeServerlessLarge2025 is a serverless system for large-scale LLM inference
serving. Its goal is to hide infrastructure complexity while managing accelerator resources,
distributed inference, KV-cache state, request scheduling, prefill and decode execution,
autoscaling, and cold starts. The central idea is that LLM serving should be treated as a stateful
and heterogeneous distributed workload rather than as ordinary stateless serverless computation.

The design combines a Request–Job–Task abstraction, the FlowServe inference engine, a Relational
Tensor Cache (RTC) with DistFlow for state movement, a scheduler that jointly considers execution
mode, cache locality, and load, and startup optimizations including prewarming and NPU-fork. The
supplied notes do not include experimental results, so the assessment below is design rationale
rather than measured outcomes.

=== Issues Addressed

DeepServe targets cloud AI platforms that must support workloads with different duration, latency,
accelerator demand, and utilization patterns, including online inference, batch inference,
fine-tuning, and agent workloads. Static allocation can therefore leave capacity unused or
insufficient. The system aims to allocate resources dynamically behind a serverless interface. This
is related to the coupling of resource allocation, placement, and communication in
@issue-inference-resource-heterogeneity[Distributed-Inference Resource Heterogeneity and Model
  Placement], but DeepServe's emphasis is heterogeneous *workloads* rather than mixed-device model
placement.

LLM inference is also stateful. As described in @background-prefill-decode-kv-cache[Prefill, Decode,
  and the Key-Value Cache], workers hold KV-cache prefixes that later requests may reuse. Routing
therefore faces the locality-versus-load trade-off in @issue-cache-locality-load[Cache Locality
  versus Load Balancing in Distributed Serving].

Prefill and decode use hardware differently, so colocating them is not always the best resource
configuration. If the phases run on different workers, or if cache reuse requires moving tensors,
the system must choose between transferring cached state and recomputing it locally. The better
choice depends on communication and computation cost.

Finally, LLM cold starts are much more expensive than starting a typical serverless function. A new
instance may need to create a runtime, initialize accelerator and communication libraries, load
model weights from storage into accelerator memory, allocate KV-cache memory, and initialize the
inference engine. Ordinary reactive autoscaling can therefore respond too slowly to sudden traffic
changes.

=== Method

==== Request–Job–Task Abstraction and Executors

DeepServe separates user-facing requests from physical execution. A *request* is the external
operation submitted by a user. A *job* represents the logical work needed to complete that request.
A *task* is an execution unit assigned to a Task Executor. This lets the system change the
underlying strategy without changing the user-facing API. Colocated inference can be one serving job
with a single inference task; prefill/decode-disaggregated inference can be one serving job with
separate prefill and decode tasks.

The Job Executor manages the logical request: it receives jobs, creates tasks, coordinates
execution, and interacts with the scheduler. The Task Executor performs the inference work and
contains the serving engine and accelerator resources.

==== FlowServe

FlowServe is DeepServe's LLM inference engine. It separates scheduling, model execution,
tokenization, memory management, KV-cache management, and communication rather than coupling all
serving functionality into one monolithic component.

Its execution is accelerator-centric: CPU-side scheduling, communication, and data preparation are
overlapped with accelerator computation so that preparing the next batch can proceed while the
current batch is running. FlowServe uses a Single Program Multiple Data (SPMD) style in which a
master coordinates execution across accelerator workers, supporting distributed and tensor-parallel
inference.

==== Relational Tensor Cache and DistFlow

The Relational Tensor Cache (RTC) manages KV-cache data, including allocation, prefix matching,
locating cached tensors, placement, movement between memory locations, and asynchronous transfer.
For an incoming prompt, RTC can determine whether part of its prefix has already been computed and
whether reuse is worthwhile. Cache reuse is not automatically beneficial: DeepServe may compare the
cost of transferring a cache against the cost of recomputing it and choose the cheaper option.

RTC determines what state should move. DistFlow determines how that state is transferred, including
worker-to-worker KV-cache and tensor movement and movement between memory tiers. This supports cache
reuse, prefill/decode disaggregation, distributed inference, and model and state movement.

==== Prefill–Decode Execution and Combined Scheduling

DeepServe supports both colocated and disaggregated prefill/decode execution. Colocation avoids
KV-cache transfer between workers and is simpler. Disaggregation lets the phases use separately
managed, independently scaled resources at the cost of KV-cache transfer and more complicated
orchestration. The system does not assume that one mode is universally better.

The scheduler considers whether a request should use PD-colocated or PD-disaggregated execution,
using characteristics such as input length, estimated output length, and current system conditions.
Output length is unknown at arrival, so DeepServe estimates it when selecting a mode. It then
instantiates @issue-cache-locality-load[Cache Locality versus Load Balancing in Distributed Serving]
by jointly considering execution mode, KV-cache locality, and worker load. A simplified conceptual
process is to choose a PD execution mode, find candidate workers, and then prefer cache locality
when load is reasonably balanced or prefer load balancing when load is highly imbalanced. The
supplied notes present this as a design principle rather than as an exact algorithm.

==== Fast Autoscaling

DeepServe optimizes several stages of instance startup. Pre-warming initializes runtime
environments, accelerators, and serving infrastructure before demand arrives, and tries to make
pre-warmed resources reusable across serving configurations rather than permanently binding every
warm instance to one model. Model weights can be prepared in host DRAM so that scale-up need not
repeatedly fetch large files from slower storage.

NPU-fork uses an already-running model instance as the source of weights for a new accelerator,
replicating model state over high-speed accelerator interconnects instead of reloading from storage
through host memory. This is useful when increasing the number of existing replicas. It does not
remove the need to create the first instance by another loading path.

=== Pros and Cons

==== Pros

- The design addresses execution, scheduling, caching, networking, resource management, and
  autoscaling together rather than optimizing only one part of LLM serving.
- Its scheduler instantiates @issue-cache-locality-load[Cache Locality versus Load Balancing in
    Distributed Serving] together with PD execution-mode selection, rather than treating locality
  and load as independent decisions.
- Supporting both colocated and disaggregated prefill/decode execution lets the system adapt the
  physical strategy to the request and cluster conditions.
- The Request–Job–Task model hides accelerator placement, distributed workers, phase assignment, and
  cache transfers from the user-facing API.
- FlowServe's modular split of scheduling, execution, memory management, and communication can make
  the serving stack easier to extend.
- Cold-start optimizations target several bottlenecks separately—runtime initialization, model
  loading, accelerator setup, memory preparation, and model replication—rather than treating startup
  as a single problem.

==== Cons

- The implementation is closely tied to Huawei accelerator and networking infrastructure, including
  Ascend NPUs, accelerator interconnects, Huawei communication libraries, and SuperPod-style
  architectures. The general concepts can apply elsewhere, but the implementation cannot necessarily
  be transferred directly to other hardware.
- Several techniques rely on efficient movement of large tensors: KV-cache transfer, prefill/decode
  disaggregation, NPU-fork, and distributed inference. On clusters with slower interconnects,
  communication overhead may reduce their benefits.
- Disaggregation adds communication, coordination, state ownership, scheduling, and failure-handling
  complexity compared with colocated serving.
- PD-aware scheduling depends on predicted output length. Incorrect estimates can lead to suboptimal
  routing.
- Prewarming improves responsiveness but consumes host memory, CPU, accelerator-related resources,
  and cluster capacity.
- NPU-fork cannot scale from zero on its own: it needs an existing accelerator that already holds
  the model weights.
- Jointly considering load, KV locality, execution mode, request characteristics, and available
  resources increases scheduler complexity. The supplied notes treat the centralized-coordination
  risk in @issue-cache-locality-load[Cache Locality versus Load Balancing in Distributed Serving] as
  a design concern rather than a measured failure.
