== Eager Updates for Overlapped Communication and Computation in DiLoCo <paper-eager-updates>

=== Summary

Eager Updates extends DiLoCo @douillard_diloco_2024 for training across poorly connected workers,
such as separate data centers. Although DiLoCo communicates only after a block of local
optimization, its usual outer update remains a blocking synchronization point. Eager Updates
overlaps that communication with the next local-training block. When the delayed aggregate becomes
available, each worker replaces its stale local contribution with its newly computed one before
applying the outer update. The authors report performance competitive with ordinary DiLoCo in
low-bandwidth settings, while avoiding most of the communication wait @kale_eager_2025.

=== Issues Addressed

The paper addresses the residual synchronization latency in low-communication distributed training,
an instance of @issue-communication-cost[Communication Cost and Synchronization], and the resulting
trade-off in @issue-asynchronous-update-staleness[Asynchronous Update Staleness]. In standard
DiLoCo, a worker must finish a local block, exchange outer gradients, wait for the aggregate, and
only then begin the following block. This leaves accelerators idle when cross-datacenter
communication is slow. Eager Updates specifically seeks to hide that outer-update communication
while limiting the quality loss from delayed remote contributions.

=== Method

Let $M$ workers each perform $H$ inner optimizer steps. Worker $m$ produces an outer gradient
$Delta_m^(t)$ at outer round $t$; DiLoCo normally averages the current outer gradients and applies
the resulting update synchronously. In the delayed setting, workers instead transmit an outer
gradient and immediately begin their next inner block while the exchange proceeds. The available
aggregate at the next outer update is consequently $Delta^(t-H)$, an average of the workers'
previous outer gradients.

For worker $m$, Eager Updates corrects that delayed aggregate using the worker's current local outer
gradient:

$
  tilde(Delta)_m^(t) = Delta^(t-H) + 1 / M (Delta_m^(t) - Delta_m^(t-H)).
$

Equivalently, it averages the worker's fresh local contribution with the other workers' delayed
contributions:

$
  tilde(Delta)_m^(t) = 1 / M (Delta_m^(t) + sum_(m' != m) Delta_(m')^(t-H)).
$

Thus, a worker does not wait for a current contribution that it already possesses; only remote
contributions are stale. Since every worker has a different current local contribution, their eager
outer updates may temporarily differ. The paper also evaluates eager updates alongside Streaming
DiLoCo and lower-precision communication as complementary ways to reduce network requirements.

=== Pros and Cons

==== Pros

- The outer-update communication can be hidden behind the following local-training block, improving
  accelerator utilization when links between workers are slow.
- Replacing the local stale contribution makes the update more current than a naive delayed
  aggregate; the paper reports substantially better training behavior for eager updates than for
  fully stale delayed updates.
- It is a focused modification of DiLoCo's outer update and complements techniques that reduce
  communication frequency or payload size.

==== Cons

- Remote contributions remain stale. Longer delays therefore still reduce model quality, so the
  method tolerates rather than removes synchronization lag; see
  @issue-asynchronous-update-staleness[Asynchronous Update Staleness].
- Workers apply different eager aggregates during the overlap period, weakening the exact consensus
  of synchronized DiLoCo.
- The main empirical evaluation trains models up to roughly one billion parameters; larger-scale
  bandwidth and speed results are system simulations, so full-training evidence at those scales is
  limited.
- Most experiments use few replicas. As the worker count rises, a single fresh local contribution is
  a smaller share of the aggregate, and the behavior at much larger replica counts remains
  uncertain.
- The evaluation is empirical; the paper does not provide a complete convergence analysis for
  delayed outer gradients.
