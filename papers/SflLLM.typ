== Efficient Split Federated Learning for Large Language Models over Communication Networks <paper-sflllm>

=== Summary

SflLLM is a framework for fine-tuning language models from data held by resource-constrained edge
devices. It combines split federated learning with @background-lora[LoRA]: clients run an early
portion of a frozen base model and train lightweight adapters, while a main server executes the
remaining, more expensive portion. A separate federated server aggregates the client-side adapters
across clients @zhaoEfficientSplitFederated2025.

The paper's central systems contribution is to jointly choose wireless subchannels, transmit power,
the model split point, and LoRA rank to minimize estimated *total* training latency to a target
performance level. This accounts for the fact that LoRA rank changes both the cost of each training
round and the estimated number of rounds required for convergence. On E2E text generation with GPT-2
variants, the authors report perplexity close to centralized LoRA training and up to roughly 60%
lower latency than a fully random resource-and-configuration baseline.

=== Issues Addressed

SflLLM addresses the resource limits that make conventional federated fine-tuning impractical when
every client must store and execute a full language model. Splitting the model shifts later-layer
computation to the main server, while LoRA restricts training to small adapter parameters.

It also addresses the communication and synchronization costs described in
@issue-communication-cost[Communication Cost and Synchronization]. In particular, synchronous
training is limited by stragglers: clients with weak processors or poor wireless links determine the
round duration. The paper treats the resulting maximum per-client delay as a quantity to optimize
rather than optimizing an average client.

Finally, the work specializes the configuration trade-offs in @issue-resource-heterogeneity[Resource
  Heterogeneity and Configuration Adaptation] to a wireless split-learning setting. A higher LoRA
rank can improve capacity and reduce the rounds needed to reach a target, but increases adapter
computation and communication. Similarly, a deeper split increases client computation while changing
server workload and activation-transfer cost.

SflLLM's cut-layer activations and corresponding labels are visible to the main server. This is a
paper-specific instance of the risks summarized in @issue-privacy-leakage[Privacy Leakage Beyond
  Data Locality]; the paper does not add a protection mechanism for those transmissions.

=== Method

The system has clients, a main server, and a federated server. The pretrained model is cut at a
chosen layer, with LoRA adapters on both its client-side and server-side portions. For each local
iteration, a client runs a minibatch through its local layers, sends the cut-layer activation and
labels to the main server, and receives the activation gradient after the server completes its
forward and backward passes. Both sides update their respective LoRA parameters. The process repeats
for $I$ local iterations.

After these iterations, clients send their client-side adapters to the federated server. It computes
a data-size-weighted FedAvg-style aggregate, as described in @background-fedavg[Federated
  Averaging]:

$
  Delta W_c = sum_(k=1)^K (D_k / D) Delta W_k,
$

where $D_k$ is client $k$'s data size and $D = sum_(k=1)^K D_k$. The federated server broadcasts the
aggregated client-side adapter for the next round; the main server updates its server-side adapter
during split training.

The optimization estimates total time as the rank-dependent number of global rounds $E(r)$ times the
cost of a round, including $I$ local iterations, activation communication, computation, and adapter
aggregation. The authors estimate $E(r)$ offline on representative training data, then use block
coordinate descent to alternate among four decisions:

- A greedy subchannel-allocation heuristic gives priority to clients likely to be stragglers and
  continues assigning channels to the client with the largest latency.
- Auxiliary rate variables reformulate the power-control subproblem as a convex optimization
  problem.
- Exhaustive search selects among candidate cut layers.
- Exhaustive search selects among candidate LoRA ranks using the offline estimate $E(r)$.

The full formulation is mixed-integer and non-convex. The paper therefore presents the procedure as
a practical iterative solution, not a globally optimal one.

=== Pros and Cons

==== Pros

- The total-latency objective connects model convergence to networking and computation costs,
  avoiding the assumption that the fastest individual round yields the fastest training process.
- Model splitting, adapter-only training, and straggler-aware allocation address complementary
  limitations of edge-device fine-tuning.
- The optimization variables have clear operational interpretations: channel allocation, transmit
  power, computation placement, and adapter capacity.
- In the reported GPT2-S E2E experiment, SflLLM's final perplexity was close to centralized LoRA; at
  rank four, the paper reports approximately $1.0399$ versus $1.0393$.

==== Cons

- The evaluation is small relative to contemporary LLM deployments: it primarily uses 124-million-
  parameter GPT2-S, some GPT2-M convergence experiments, and typically five simulated clients.
- Wireless links, device capabilities, and much of the latency evaluation are simulated. The results
  support the proposed model and optimizer, but do not demonstrate an end-to-end deployment on
  heterogeneous physical devices.
- Learning results use the E2E restaurant-domain generation dataset. Performance under broader
  tasks, strongly non-IID data, larger client populations, and larger models remains untested.
- The rank choice depends on an offline estimate of $E(r)$. If representative data do not reflect
  client data, the predicted latency-optimal rank may not be optimal in practice.
- The block-coordinate procedure has no formal global-optimality guarantee, and exhaustive search
  over cut locations and ranks becomes less appealing with more architectural choices.
- The resource-management baselines mostly randomize one or more proposed decisions. They serve as
  useful ablations, but stronger comparisons with alternative joint allocation methods would better
  establish the optimizer's advantage.
