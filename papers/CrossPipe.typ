== CrossPipe: Towards Optimal Pipeline Schedules for Cross-Datacenter Training <paper-crosspipe>

=== Summary

CrossPipe@chenCrossPipeOptimalPipeline2025 is a pipeline-parallel training system for
cross-datacenter training, where communication between pipeline stages may traverse slow wide-area
links. Conventional pipeline schedules assume relatively fast communication. Across datacenters,
long transfer delays can leave GPUs idle in pipeline bubbles.

CrossPipe makes scheduling communication-aware. Instead of a fixed forward/backward order, it
rearranges forward and backward work using computation time, communication latency and bandwidth,
link availability, data dependencies, and GPU memory limits. It provides a constraint-based
optimizer that searches for an optimal schedule and a cheaper greedy scheduler. It also separates
the high-level computation schedule from low-level communication execution so that unnecessary
synchronization can be avoided.

The central idea is to treat cross-datacenter delay as a scheduling problem: fill communication-
induced idle periods with other legal computation whenever dependencies and memory constraints
allow. The supplied notes do not include experimental results, so the assessment below is design
rationale rather than measured outcomes.

=== Issues Addressed

CrossPipe targets pipeline-parallel training when adjacent stages sit in different datacenters.
Intra-datacenter networks are typically high-bandwidth and low-latency. Inter-datacenter links have
higher latency, lower bandwidth, larger variance, and may be shared or contended. In pipeline
parallelism, activations move forward from stage to stage, while gradients travel backward. If two
neighboring stages are in different datacenters, that hop can dominate the iteration. These idle
periods and the general techniques for hiding them are described in
@issue-communication-computation-overlap[Communication-Induced Idle Time and Computation Overlap in
  Distributed Training].

Existing schedules such as 1F1B, interleaved 1F1B, and Zero-Bubble pipelines define a largely
predetermined execution order. They work well when communication is cheap and predictable, but they
do not explicitly order operations around slow cross-datacenter links.

CrossPipe focuses on pipeline parallelism (PP) as the dimension that should cross datacenter
boundaries. Tensor, sequence, and expert parallelism tend to require frequent communication that is
hard to place on slow links. Data parallelism can require large parameter or gradient
synchronization. PP primarily exchanges activations and activation gradients between neighboring
stages, so pipeline boundaries are a natural place to cross datacenters. This communication pattern
is distinct from the periodic model-update exchanges discussed in
@issue-communication-cost[Communication Cost and Synchronization].

=== Method

==== Architecture

CrossPipe proceeds from system and model profiling, through a communication-aware performance model,
to pipeline-schedule generation, communication-plan generation, and runtime execution. The scheduler
treats computation and communication jointly rather than as a small fixed overhead.

==== Computation decomposition

A training step is represented more finely than a single forward and a single backward. Let $F$ be
forward computation, $D$ input- or data-gradient computation, and $W$ weight-gradient computation,
so that backward work is $B = D + W$. $D$ and $W$ need not run consecutively. A legal schedule may
therefore interleave later forwards with earlier weight-gradient work, provided dependencies remain
valid.

$D$ is often higher priority for pipeline progress because it produces the gradient the previous
stage needs. $W$ mainly updates local parameters, so delaying it may not immediately block another
stage. Weight-gradient work is therefore useful filler for otherwise idle periods.

==== Communication model

Communication time is modeled as

$
  T_"comm" = alpha + beta M,
$

where $alpha$ is fixed latency, $beta$ is the per-byte cost, and $M$ is message size. Small messages
are latency-dominated; large messages are increasingly bandwidth-limited. The model also accounts
for whether the link is already occupied, treating the network as a scheduling resource rather than
an invisible overhead.

==== Constraint-based scheduler

The optimization-based scheduler assigns each operation a start time $t_o$ and searches for an
ordering that minimizes iteration completion time, or makespan: the time from the start of the
iteration to the last operation. Conceptually the objective is $min T_"finish"$, subject to
dependency, resource, and memory constraints.

Dependencies include the forward path (compute, send activation, receive activation, next-stage
forward) and the backward path (compute, send gradient, previous-stage backward). A next-stage
forward cannot start before its activation arrives. Reordering is allowed only when these
dependencies remain valid.

If the execution model assumes exclusive GPU use, two computations cannot occupy the same GPU at
once. Cross-datacenter transfers may also contend for the same link. The scheduler therefore reasons
about GPU availability, link availability, and operation dependencies together.

Pipeline order also affects memory. Forward computations produce activations that generally remain
until the matching backward work. Many unfinished forwards therefore mean more stored activations.
Hiding communication by running extra in-flight microbatches is not always feasible. Memory limits
are part of schedule generation, together with dependency validity and resource conflicts.

This formulation gives a principled target for a good schedule. The number of candidate orderings
grows rapidly with the number of stages and microbatches.

==== Greedy scheduler

CrossPipe also uses a practical greedy algorithm: collect currently schedulable operations, choose
one by priority, place it, update dependencies and resource availability, and repeat. Priorities
change by pipeline phase.

- Warm-up generally ranks $F > D > W$ in order to fill the pipeline.
- Steady state interleaves forward and data-gradient operations so both pipeline directions keep
  moving.
- Tear-down generally prefers data-gradient work over weight-gradient work, because completing $D$
  can unblock upstream stages.

==== Sub-block scheduling

Computation blocks can be split into smaller pieces. A full weight-gradient operation may not fit a
short communication bubble, while smaller sub-blocks can. Finer granularity therefore gives the
scheduler more ways to hide delay.

==== Communication orchestration

A good computation order can still stall if the communication runtime synchronizes sends and
receives unnecessarily. CrossPipe separates computation scheduling from communication execution. The
runtime can prepare transfers independently and use asynchronous communication where possible, so
sends and receives overlap with GPU work on a separate path rather than forcing one to wait for the
other.

=== Pros and Cons

==== Pros

- Communication is a first-class scheduling constraint. Latency, bandwidth, communication
  dependencies, and link occupancy are modeled explicitly, which is more appropriate for
  cross-datacenter settings than treating communication as negligible.
- Splitting backward work into $D + W$ creates additional legal orders. Work that does not
  immediately unblock another stage can be postponed and used to fill idle time.
- Memory is an explicit constraint. The scheduler does not hide communication by blindly increasing
  the number of in-flight microbatches.
- The scheduler jointly considers compute dependencies, network dependencies, GPU resources, network
  resources, and memory.
- The constraint solver and the greedy scheduler are complementary: the former is a principled
  search, the latter is cheaper to generate. The optimization formulation also clarifies what the
  greedy method is approximating.
- Sub-block scheduling can fit computation into communication-induced gaps more precisely.
- Separating computation scheduling from communication orchestration recognizes that a theoretically
  good schedule can still perform poorly if the runtime adds unnecessary synchronization.

==== Cons

- Full constraint optimization is expensive. The number of orderings grows rapidly with pipeline
  stages, microbatches, computation blocks, and communication operations, so the optimal solver does
  not scale cheaply to arbitrary configurations.
- The greedy scheduler has no global optimality guarantee. A locally attractive choice can block a
  better later arrangement.
- Scheduling can hide waiting time but cannot create extra network bandwidth. If communication
  volume is too large relative to link bandwidth, there may not be enough computation to cover the
  transfer.
- Schedule quality depends on estimates of computation duration, communication latency and
  bandwidth, and memory use. If those quantities change after the schedule is generated, the chosen
  order may no longer be suitable. This is especially relevant on variable wide-area links.
- The runtime is more complex than a fixed pipeline rule: profiling, schedule generation, dependency
  tracking, sub-block execution, and communication orchestration all add implementation burden.
- Heterogeneous datacenters, with different GPU speeds, memory capacities, and network
  characteristics, add load-balancing problems beyond the basic cross-datacenter communication
  issue.
- Optimizing the pipeline schedule does not provide fault tolerance. GPU or node failure, datacenter
  outage, and network disconnection still require mechanisms such as checkpointing and distributed
  recovery.
