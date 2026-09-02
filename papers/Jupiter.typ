== Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices <paper-jupiter>

=== Summary

Jupiter@yeJupiterFastResourceEfficient2025 is a collaborative inference system that runs a
generative large language model (LLM) across resource-constrained, nearby edge devices. It uses
unequal, consecutive Transformer-layer partitions to form a pipeline, then adds phase-specific
scheduling to accelerate both prompt processing and autoregressive generation for a single request.
For prefill, it turns one prompt into dependent sub-sequences that can occupy several pipeline
stages concurrently. For decoding, it combines Medusa-style speculative decoding with an optional
outline-based scheme that generates relatively independent answer sections in parallel. The paper
reports up to a $26.1 times$ end-to-end latency reduction over its evaluated baselines.

=== Issues Addressed

Jupiter instantiates @issue-inference-resource-heterogeneity[Distributed-Inference Resource
  Heterogeneity and Model Placement] on edge hardware. A single device may lack enough compute or
memory for model parameters, activations, and the KV cache, while an equal layer split can leave a
slower participant as the bottleneck. Its dynamic-programming planner therefore assigns a
memory-feasible, unequal number of consecutive layers to each profiled device.

The paper also targets the communication and utilization trade-off of distributed inference. Tensor
parallelism requires frequent intra-layer synchronization, which is costly over edge links; Jupiter
instead transfers hidden states only between adjacent pipeline stages. Ordinary pipeline
parallelism, however, leaves later stages idle for a single request. Jupiter creates finer-grained
work during both phases described in @background-prefill-decode-kv-cache[Prefill, Decode, and the
  Key-Value Cache], seeking to fill this pipeline without changing the causal dependencies of prompt
processing.

=== Method

Jupiter profiles device execution and memory characteristics, then uses dynamic programming to
choose the layer placement subject to each device's memory limit. The objective balances stage
latencies, so a faster device can host more layers than a slower one. A second planner chooses
input-subsequence boundaries using measured costs that depend on the current chunk length and its
preceding context.

During prefill, the system splits a prompt into ordered sub-sequences. A stage processes the next
sub-sequence once it has finished the preceding sub-sequence at that same stage and stored its local
KV states; it need not wait for that earlier sub-sequence to traverse the entire model. This lets an
earlier stage process a later chunk while a downstream stage continues the earlier one, preserving
causal attention while reducing pipeline bubbles.

During decoding, Medusa-style self-drafting proposes multiple tokens for joint verification by the
main model. Accepted proposals advance decoding by several tokens, while rejected candidates require
their distributed KV-cache entries to be removed. For outputs that can be separated into mostly
independent sections, Jupiter can first produce an outline and then submit section-generation
requests to the pipeline concurrently. These requests reuse the original prompt's common KV cache;
the optimization is optional because independent sections no longer retain all dependencies of a
single autoregressive continuation.

=== Pros and Cons

==== Pros

- Pipeline stages exchange only adjacent hidden states, avoiding the frequent synchronization of
  tensor parallelism over bandwidth-constrained edge links.
- Intra-sequence prefill creates concurrent pipeline work from one prompt while retaining
  layer-local causal-attention dependencies through the KV cache.
- Unequal layer and input partitions account for heterogeneous compute, memory, and
  context-dependent attention costs rather than assuming uniform devices or chunks.
- The system addresses both inference phases; the paper reports substantial end-to-end latency
  improvements over its evaluated baselines.

==== Cons

- Outline-based decoding changes the response-generation process. The supplied notes report quality
  degradation, especially for mathematics and coding tasks that depend on tightly coupled sequential
  reasoning; speculative decoding alone has a different, verification-based trade-off.
- The planned partition depends on offline profiles, which can become inaccurate with wireless
  variation, contention, thermal throttling, changing memory availability, or device failure.
- Evaluation is concentrated on small edge-device clusters and Llama2 7B and 13B models, leaving
  scaling to larger models, many devices, long contexts, and highly variable consumer hardware open.
- The supplied notes report limited energy measurements. Reduced latency, memory per device, and
  communication do not by themselves demonstrate lower total energy consumption.
