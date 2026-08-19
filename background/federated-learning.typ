== Federated Learning <background-federated-learning>

Federated learning (FL) trains a shared model while keeping each participant's raw data on its own
client. A coordinating server sends a model to selected clients, each client trains locally, and the
server aggregates their updates. This gives data locality, not an automatic privacy guarantee:
updates and metadata may still leak information, and malicious clients may attempt to poison
training.

=== Federated Averaging <background-fedavg>

Federated Averaging (FedAvg) is the classic practical FL algorithm
@mcmahan_communication-efficient_2023. Let client $k$ hold $n_k$ examples, let $n = sum_k n_k$, and
let the server model at round $t$ be $w_t$. A round proceeds as follows:

1. The server selects available clients and broadcasts $w_t$.
2. Each client runs local optimization for $E$ steps, producing $w_(t+1)^k$.
3. Clients return parameters or parameter changes rather than raw examples.
4. The server computes the data-size-weighted average

$
  w_(t+1) = sum_k (n_k / n) w_(t+1)^k.
$

Equivalently, clients may send $Delta w_k = w_(t+1)^k - w_t$. More local work can reduce the number
of communication rounds, but it can also amplify client drift under heterogeneous data.

FL commonly minimizes the weighted objective

$
  min_w F(w) = sum_k (n_k / n) F_k(w),
$

where $F_k$ is client $k$'s local loss. Equal-client or fairness-aware weighting is possible, but it
optimizes a different target.

=== Operational considerations

Clients differ in compute, connectivity, availability, and dataset size. Dropouts and stragglers can
therefore affect which data influences training. Evaluation should look beyond a global average to
per-client and worst-group performance. Client datasets are also commonly non-IID, so their local
objectives may pull the shared model in different directions. Repeated model exchange can be a major
cost, especially for large models.
