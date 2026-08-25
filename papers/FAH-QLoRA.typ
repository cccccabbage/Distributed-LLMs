== FAH-QLoRA: Federated Adaptive Fine-Tuning with Heterogeneous Quantization and LoRA <paper-fah-qlora>

=== Summary

FAH-QLoRA@gao_federated_nodate is a federated framework for fine-tuning large language models on
devices with unequal compute, memory, and communication resources. It combines a quantized frozen
base model with LoRA adapters, then adapts the federation's average LoRA rank across rounds and
assigns clients different ranks within each round. This seeks to reduce memory use and the
slowest-client delay without giving every client the same adapter workload. The paper reports up to
45.86 percent lower training time and 44.15 percent lower memory use than its baselines.

=== Issues Addressed

The framework uses @background-federated-learning[Federated Learning] to fine-tune with data that
remains on participating clients. It reduces the local memory burden left by adapter-only training
by using the quantized-base-model approach described in @background-qlora[QLoRA: Quantized LoRA].

Its main focus is @issue-resource-heterogeneity[Resource Heterogeneity and Configuration
  Adaptation]. A common LoRA rank makes synchronous rounds depend on the slowest client, although
larger ranks are affordable for better provisioned clients. Varying rank also changes the size of
adapter updates, connecting the method to @issue-communication-cost[Communication Cost and
  Synchronization].

Client data remains local, but individual adapter updates retain the risks described in
@issue-privacy-leakage[Privacy Leakage Beyond Data Locality]; FAH-QLoRA does not add formal privacy
mechanisms such as secure aggregation or differential privacy.

=== Method

FAH-QLoRA freezes the pretrained base model and stores it at reduced precision, while each client
trains only a LoRA update $Delta W = B A$. The server may use heterogeneous quantization
configurations and assigns client $n$ a rank $r_n^i$ in round $i$. A lower rank reduces the client's
trainable state, computation, memory demand, and adapter communication, but also limits adapter
capacity.

The server first adapts the desired average rank $r^i$ between rounds according to loss reduction
per wall-clock time:

$
  R^i = (F^(i - 1) - F^i) / T^i,
$

where $F^i$ is the round loss and $T^i$ is the round duration. It then assigns individual,
integer-valued ranks so that their average approximately equals the selected target:

$
  (1 / N) sum_(n = 1)^N r_n^i = r^i.
$

The assignment considers each client's computation and communication time. Because the resulting
rank-allocation problem is discrete, the paper relaxes it to a continuous approximation, solves it,
then rounds and adjusts the allocation to meet the constraints.

Clients truncate the global adapter to their assigned rank, train locally, and return their
different-sized LoRA factors. Before aggregation, the server zero-pads lower-rank factors to a
common shape, aggregates the compatible updates into a global adapter, and distributes rank-
appropriate truncations for the next round. The paper also provides convergence analysis for
non-convex, non-IID federated optimization with heterogeneous LoRA modules and quantized models.

=== Pros and Cons

==== Pros

- Combining quantization with adapter-only training directly targets the frozen-base-model memory
  burden that LoRA alone does not remove.
- Per-client rank assignment explicitly accounts for heterogeneous compute, memory, and network
  resources, helping reduce straggler-dominated round time.
- Adapting the average rank by loss reduction per wall-clock time aligns the policy with elapsed
  training time rather than merely minimizing the number of rounds.
- The paper supplies both convergence analysis and reported end-to-end savings of up to 45.86
  percent in training time and 44.15 percent in memory use.

==== Cons

- The system must estimate client compute and communication time, solve and enforce a rank
  allocation, and reconcile differently shaped adapters, adding coordination complexity.
- Resource estimates can become inaccurate as network conditions and device load change, producing
  inefficient assignments.
- Zero-padding makes aggregation possible, but missing dimensions contribute zeros; it is a simple
  compatibility mechanism rather than evidence that heterogeneous adapters are combined optimally.
- Clients with lower ranks cannot update all dimensions trained by higher-rank clients, which may
  give better-resourced clients disproportionate influence over parts of the global adapter.
- Quantization trades memory savings for approximation error, and the method does not itself protect
  exposed client updates from information leakage.
