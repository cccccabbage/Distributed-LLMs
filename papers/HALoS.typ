== HALoS: Hierarchical Asynchronous Local SGD over Slow Networks for Geo-Distributed LLM Training <paper-halos>

=== Summary

HALoS@kimHALoSHierarchicalAsynchronous2025 is a hierarchical asynchronous local-SGD method for
training language models across geographically distributed datacenters, where inter-region
communication is much slower than communication within a region. It places a Local Parameter Server
(LPS) in each region and a Global Parameter Server (GPS) above them, so workers exchange updates
frequently inside a region while only accumulated regional updates cross the WAN. Local SGD,
hierarchical aggregation, asynchronous updates, update accumulation, local/global model merging, and
separate local and global momentum are combined so that WAN traffic and barrier waiting fall without
forcing every local interval to hide a full inter-region round trip.

=== Issues Addressed

HALoS targets geo-distributed LLM training in which intra-region networks are fast while WAN links
between datacenters are slower, higher-latency, and coupled with unequal worker speeds. Frequent
all-worker barriers then spend a large share of time on inter-region communication, and a slow
worker or slow link stalls the rest. That setting is a particularly severe instance of
@issue-communication-cost[Communication Cost and Synchronization] and of
@issue-communication-computation-overlap[Communication-Induced Idle Time and Computation Overlap in
  Distributed Training].

@paper-diloco[DiLoCo]-style local SGD reduces how often workers communicate by taking $H$ local
steps, but synchronous variants still wait for every worker at each outer boundary, so stragglers
remain. Async-Local-SGD removes that barrier, yet workers still send updates directly to a global
server over the WAN. Increasing $H$ reduces that traffic, but it also increases model drift and
staleness.

The method tries to cut WAN communication, synchronization waiting, straggler effects, and the
optimization damage from extremely large local-step intervals at once. Its paper-specific
observation is that the physical network is already hierarchical, so the training algorithm should
be too: frequent intra-region communication, infrequent inter-region communication, and no global
barrier.

Asynchronous aggregation at the LPS and GPS removes waiting at the cost of using work computed from
older parameters; see @issue-asynchronous-update-staleness[Asynchronous Update Staleness]. Long
local intervals and non-IID regional data also produce the worker and regional drift discussed in
@issue-data-heterogeneity[Data Heterogeneity and Client Drift]. Raising $H$ or $K$ cuts
communication but lets those models move further apart.

=== Method

HALoS uses a three-level hierarchy of Worker $->$ Local Parameter Server (LPS) $->$ Global Parameter
Server (GPS). Workers in a region communicate with that region's LPS over a fast network. Only
aggregated LPS updates travel to the GPS over the slow WAN.

==== Worker training and local aggregation

A worker receives parameters $theta_(t,0)$ from its LPS and performs $H$ local SGD steps. For
$j = 1, ..., H$,

$
  theta_(t,j) = theta_(t, j-1) - eta_w nabla F_i(theta_(t, j-1)).
$

It then sends the accumulated displacement $delta = theta_(t,H) - theta_(t,0)$ to the LPS. $H$ is
the number of worker steps between worker-to-LPS communication. Larger $H$ means less frequent local
communication and, typically, more worker-level drift.

Workers need not finish together. The LPS applies each $delta$ as it arrives, with local momentum
$beta_l$, rather than waiting for a regional barrier. A relatively large local momentum, for example
$beta_l = 0.9$, suits this setting, because updates inside a regional group tend to be consistent.

Sending every LPS update to the GPS would still produce too much WAN traffic. After $K$ worker
updates move the LPS from $theta_"old"$ to $theta_"new"$, the LPS sends

$
  Delta = theta_"new" - theta_"old"
$

to the GPS. $K$ is the number of LPS updates accumulated before LPS-to-GPS communication. Larger $K$
reduces WAN rounds but can let regional models drift further apart.

==== Global aggregation and model merging

The GPS likewise applies regional updates as they arrive, without waiting for every LPS. It uses a
separate global momentum $beta_g$. HALoS generally keeps the global momentum more conservative than
the local momentum, because global updates can be staler, computed from different model states, and
produced by regions whose data and progress differ; one example is $beta_l = 0.9$ and
$beta_g = 0.5$.

An LPS does not pause workers while a WAN round trip is in flight. Local training continues from the
pre-send state through later local states. Replacing the current local model with the returned
global model $Theta$ would discard that in-flight work. HALoS instead interpolates:

$
  theta <- (1 - alpha) theta + alpha Theta,
$

where $theta$ is the current local model, $Theta$ is the received global model, and $alpha$ is the
merging weight. An intermediate value such as $alpha approx 0.25$ keeps most recent local progress
while still incorporating global information. $alpha = 0$ ignores the global model; $alpha = 1$
replaces the local model entirely.

The main hyperparameters are:

- $H$: local SGD steps a worker takes before sending an update to its LPS
- $K$: LPS updates accumulated before communicating with the GPS
- $alpha$: weight on the received global model in the merge
- $beta_l$: local momentum at the LPS
- $beta_g$: global momentum at the GPS
- $eta_w$: worker learning rate

In outline, a worker pulls a model from its LPS, runs $H$ local steps, and returns $delta$. That LPS
applies the update immediately with $beta_l$, and after $K$ updates it sends $Delta$ to the GPS
while continuing to serve workers. Regional updates are applied at the GPS asynchronously with
$beta_g$, which then returns $Theta$. The LPS merges $Theta$ into its current $theta$ using $alpha$,
and training continues. Frequent local communication, infrequent global communication, and
asynchronous execution are the organizing pattern.

=== Pros and Cons

==== Pros

- Workers mostly communicate with a nearby LPS; only accumulated regional updates cross the WAN,
  which reduces expensive inter-region traffic relative to sending every worker update to a global
  server.
- Asynchronous worker-to-LPS aggregation lets faster workers continue without waiting, which the
  paper argues reduces straggler delay under heterogeneous compute and networks.
- An LPS keeps processing worker updates while a WAN round trip is in flight, so communication can
  overlap with computation.
- Merging $theta <- (1 - alpha) theta + alpha Theta$ keeps local progress made during that wait
  rather than overwriting it.
- Analysis: the Worker / LPS / GPS hierarchy matches the fast-intra-region, slow-inter-region
  topology instead of treating all links as equal.
- Separate $beta_l$ and $beta_g$ let the optimizer treat fresher intra-region updates differently
  from staler, more heterogeneous global updates.

==== Cons

- Several additional hyperparameters affect convergence: $H$, $K$, $alpha$, $beta_l$, $beta_g$, and
  $eta_w$. Too large $H$ or $K$ increases drift; too small $alpha$ underuses global information; too
  large $alpha$ discards recent local progress; too large $beta_g$ can reinforce stale global
  directions.
- Worker grouping and LPS load balance affect stability. Analysis: uneven grouping lets regional
  models progress at very different rates, so deployment topology and load balancing are part of the
  method, not incidental.
- Global momentum can hurt when regions train on significantly different data. Conflicting regional
  directions $Delta_A$ and $Delta_B$ may then be mixed with a stale global velocity, so a strong
  global momentum can reinforce an outdated direction.
- Asynchronous execution still introduces staleness; see
  @issue-asynchronous-update-staleness[Asynchronous Update Staleness]. Analysis: HALoS reduces the
  cost of waiting but does not eliminate the wait-versus-staleness trade-off.
