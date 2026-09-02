== Photon: Federated LLM Pre-Training <paper-photon>

=== Summary

Photon@saniPhotonFederatedLLM2025 studies cross-silo pre-training of decoder-only language models
when GPU clusters and training data are geographically distributed. Instead of gradient
synchronization at every optimization step, it applies the periodically averaged local-training
pattern of @background-federated-learning[Federated Learning] to full-model pre-training. Its main
finding is that small local batches, high local learning rates, and many local steps between
aggregations can maintain strong language-modeling quality while greatly reducing cross-site
communication. The authors demonstrate models from 125M to 7B parameters, reporting 64--512 times
less communication than conventional distributed training and lower C4 validation perplexity than
their corresponding centralized baselines in the reported 1.3B-, 3B-, and 7B-parameter experiments.

=== Issues Addressed

The paper targets the high bandwidth and low latency normally required for data-parallel LLM
pre-training. This is the communication-frequency aspect of @issue-communication-cost[Communication
  Cost and Synchronization]: a parameter-sized exchange is still needed, but only after a sequence
of local steps rather than after every step. Photon is intended for organizations whose compute and
data are separated by wide-area links, where pooling data or building a data-center-scale network is
impractical.

Long local intervals cause the client parameters to diverge, particularly when participants train on
different corpora. This is the optimization trade-off described in @issue-data-heterogeneity[Data
  Heterogeneity and Client Drift]. Photon evaluates both a homogeneous C4 split and heterogeneous
Pile sources, including ArXiv, C4, Wikipedia, and Project Gutenberg. It also keeps raw data at
clients, but does not present data locality as a privacy guarantee; the remaining exposure from
model updates is described in @issue-privacy-leakage[Privacy Leakage Beyond Data Locality].

=== Method

At federated round $t$, an aggregator broadcasts global parameters $theta_t$ to $K$ clients. Each
client then runs AdamW on its local data for $H$ steps, yielding $theta_k$. The main experiments
consider $H$ values of 62, 128, and 512. For equally weighted clients, the returned displacement and
its average are

$
  Delta_k = theta_t - theta_k
$

and

$
  Delta = 1 / K sum_(k=1)^K Delta_k.
$

The aggregator applies this FedAvg update with a server learning rate of about 1.0 and no server
momentum, producing $theta_(t+1)$ for the next round. Increasing $H$ changes the number of
model-sized exchanges from the per-step scaling $O(|theta| T)$ to approximately $O(|theta| T / H)$,
while leaving the local optimization work intact.

Photon's reported training recipe combines small local batches with comparatively high learning
rates and periodic averaging. The authors argue that the noise of independent local optimization,
followed by averaging, can make this combination effective despite the long local intervals. Each
client may use DDP or FSDP within its high-bandwidth local cluster, while the federation manages
communication between sites. The implementation separates local data storage from compute through
streaming, caching, pre-tokenization, and compression; the aggregator handles client selection,
model exchange, update aggregation, and checkpoints.

The evaluation uses H100-based clusters in locations including England, Utah, Texas, Quebec, and
Maharashtra. On C4 validation, the reported Photon perplexities for the 1.3B, 3B, and 7B models are
20.1, 15.7, and 13.8, compared with 23.2, 18.2, and 16.6 for the corresponding centralized
baselines. In reported 125M-model comparisons, Photon reaches target perplexities in about half the
wall-clock time of the tested DiLoCo configurations. Adding clients supplies more compute but also
increases the effective global batch size, so the authors identify a compute-optimal federation size
rather than claiming unbounded scaling.

=== Pros and Cons

==== Pros

- The experiments extend federated pre-training to decoder-only models up to 7B parameters and
  geographically separated clusters, beyond small-scale federated-learning demonstrations.
- Infrequent model averaging reduces the reported communication volume by 64--512 times, which
  directly addresses wide-area network constraints without requiring data-center connectivity.
- The paper reports strong C4 perplexity results, including better values than its corresponding
  centralized baselines, and faster convergence than the tested DiLoCo configurations at 125M
  parameters.
- The two-level design can retain efficient DDP or FSDP within a site while using federated
  synchronization across sites, and allows participating organizations to retain raw datasets.

==== Cons

- The study is cross-silo federation: clients have powerful GPUs, often multiple H100s, and
  relatively capable networks. It does not establish practical training across a large population of
  consumer devices.
- Infrequent synchronization reduces communication rounds but still transfers full models. It also
  does not remove the substantial memory and compute requirements of LLM pre-training.
- Performance depends on the local-step count, batch size, learning-rate schedule, client count, and
  resulting global batch size. More participants can eventually yield diminishing optimization
  returns, while heterogeneous data can increase drift.
- The strongest comparisons use validation perplexity; the supplied results describe downstream
  evaluations, but provide less comprehensive centralized comparisons for capabilities such as ARC,
  HellaSwag, PIQA, and MMLU.
- TLS, secure aggregation, and differential privacy may protect parts of a deployment, but privacy
  is not the paper's primary evaluated contribution. Individual model updates can still leak
  information without an appropriate protection mechanism.
