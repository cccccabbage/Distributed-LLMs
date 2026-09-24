== Communication-Induced Idle Time and Computation Overlap in Distributed Training <issue-communication-computation-overlap>

Distributed training can leave an accelerator idle even though more work remains available: a
participant is ready to continue, but the data, gradient, or model state it depends on is still in
transit, or a synchronization barrier has not yet been satisfied. This topic complements
@issue-communication-cost[Communication Cost and Synchronization], which concerns how often and how
much is communicated. Here the concern is the idle time that appears when a required communication
cannot be hidden behind useful computation. The loss of utilization grows with latency, limited
bandwidth, and the number of synchronization points.

Idle time arises from two structurally different situations.

The first is an unmet dependency between stages. In pipeline-parallel training, activations move
forward and activation gradients move backward between adjacent stages. When adjacent stages are
separated by a slow link, a stage can finish its current work while its next input is still in
flight, so it waits. A delay is not local to one transfer: because pipeline work has dependencies
across stages and microbatches, a late activation or gradient can stall later operations and other
stages. Reducing the duration of a single transfer is therefore not sufficient, and the relevant
objective becomes the completion time of the whole schedule @chenCrossPipeOptimalPipeline2025
@aljahdaliIdleNoMore2025.

The second is a synchronization barrier. Methods that lower communication frequency by taking many
local steps before exchanging updates replace per-step synchronization with a periodic outer step.
If that outer step is blocking, every participant must finish its local work, send its update, and
wait for the aggregate before starting the next block, so all participants idle until the slowest
link and the slowest worker are ready @kaleEagerUpdatesOverlapped2025
@kimHALoSHierarchicalAsynchronous2025 @douillardDiLoCoDistributedLowCommunication2024. In
synchronous rounds, slow devices or links similarly make their peers wait and bound the round
duration @mcmahanCommunicationEfficientLearningDeep2023 @zhaoEfficientSplitFederated2025.

Lowering the cost of the underlying communication reduces these idle periods, but it cannot remove
them as long as some work must wait for a remote result. A broad family of techniques instead
reorganizes execution around the delay:

- Reordering and scheduling place independent ready operations into otherwise idle intervals when
  dependencies and memory limits still allow, and can split coarse operations into smaller pieces
  that fit into shorter gaps @chenCrossPipeOptimalPipeline2025.
- Overlapping communication with computation lets transfers proceed concurrently with local
  computation rather than forcing one to wait for the other, as with asynchronous sends and receives
  or an outer update that is applied while the next local block runs @kaleEagerUpdatesOverlapped2025
  @kimHALoSHierarchicalAsynchronous2025.
- Using idle time for additional work keeps a participant busy when it would otherwise wait, for
  example by performing extra local optimization on data it already holds @aljahdaliIdleNoMore2025.
- Reducing the number of synchronization points means participants wait at fewer boundaries,
  although each remaining boundary still blocks @douillardDiLoCoDistributedLowCommunication2024.

These families trade extra computation, memory, or staleness for higher utilization. Hiding a
dependency requires independent work to exist and to remain legal, and no schedule can create
bandwidth that the link does not have @chenCrossPipeOptimalPipeline2025. Proceeding without waiting
introduces staleness, analyzed in @issue-asynchronous-update-staleness[Asynchronous Update
  Staleness], so reducing idle time and keeping contributions fresh are in tension. Extra
computation performed during idle periods must also be controlled to avoid wasted work or
overfitting, and the benefit of overlap depends on the resource asymmetry across participants
discussed in @issue-resource-heterogeneity[Resource Heterogeneity and Configuration Adaptation].

In reinforcement-learning post-training, the same structure appears across stages rather than across
pipeline layers: rollout generation, policy training, and weight transfer can be overlapped instead
of run in lockstep, at the cost of training on rollouts produced by an earlier policy
@teamINTELLECT2ReasoningModel2025.

This training-side problem should be distinguished from inference-side pipeline bubbles, where a
single autoregressive request exposes little independent work and the causal token dependency limits
how much can be overlapped; see @issue-pipeline-bubbles-utilization[Pipeline Bubbles and Stage
  Utilization in Distributed Inference]. Training can often expose independent microbatches and
backward work by comparison, although not always to a degree that fully hides a slow link.
