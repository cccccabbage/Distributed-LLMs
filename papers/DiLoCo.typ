== DiLoCo: Distributed Low-Communication Training of Language Models <paper-diloco>

=== Summary

DiLoCo is a distributed optimization method for training language models across several poorly
connected clusters. It periodically combines models that have each trained independently for many
steps, rather than synchronizing gradients after every step. The paper frames the method as a
variant of @background-federated-learning[Federated Learning] with a large local-update interval: on
C4, its eight-worker configuration matched fully synchronous optimization while communicating 500
times less often.

=== Issues Addressed

The communication and synchronization constraints described in
@issue-communication-cost[Communication Cost and Synchronization] are especially acute when compute
is split across separate data centers or geographically dispersed clusters. DiLoCo targets that
setting by allowing a cluster to compute locally for a long interval before synchronization.

Long local intervals also make the workers' parameters diverge. This is related to the trade-off in
@issue-data-heterogeneity[Data Heterogeneity and Client Drift], although here a worker represents a
cluster rather than an individual data holder. The paper evaluates non-IID data partitions and
changing worker availability, reporting robustness in its tested settings.

=== Method

At outer round $t$, $K$ workers begin from shared global parameters $theta^(t-1)$. Each worker $i$
uses AdamW to train independently on its local shard for $H$ inner steps (the main setting uses
$H = 500$), producing $theta_i^(t)$. No communication occurs during those inner steps.

The worker sends its accumulated parameter displacement

$
  Delta_i^(t) = theta_i^(t) - theta^(t-1)
$

to the coordinator. For equally weighted workers, the coordinator averages the displacements:

$
  Delta^(t) = 1 / K sum_(i=1)^K Delta_i^(t).
$

It then applies an outer Nesterov-momentum optimizer using $Delta^(t)$ to produce the next global
parameters $theta^(t)$, broadcasts them to the workers, and repeats. Thus, the transmitted value
captures the effect of a full local training interval rather than only a final-step gradient. The
combination of AdamW for inner optimization and Nesterov momentum for outer optimization is central
to the successful configuration reported by the authors.

=== Pros and Cons

==== Pros

- With 500 local steps per round, synchronization occurs roughly 500 times less frequently than in
  fully synchronous training; the paper reports comparable C4 language-model results with eight
  workers in that setting.
- The approach can use compute spread across weakly connected clusters, and the final result is an
  ordinary language model with no distributed inference dependency.
- The authors report robustness to their tested non-IID data partitions and to workers becoming
  unavailable or joining during training.

==== Cons

- The communication trade-off described in @issue-communication-cost[Communication Cost and
    Synchronization] remains: synchronization is less frequent but still model-sized, so fewer
  rounds do not automatically imply the same reduction in total bytes transferred.
- More local steps increase worker drift, making the local-update interval and outer optimizer
  important choices. This communication--consensus trade-off is discussed in
  @issue-data-heterogeneity[Data Heterogeneity and Client Drift].
- The original experiments cover approximately 60M-, 150M-, and 400M-parameter models, so they do
  not alone establish behavior at contemporary frontier-model scales.
- The method saves communication rounds rather than total compute: workers each perform substantial
  independent optimization. Adding workers also need not yield linear speedup.
