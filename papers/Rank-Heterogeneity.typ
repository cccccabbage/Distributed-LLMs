== Towards Federated Low-Rank Adaptation of Language Models with Rank Heterogeneity <paper-rank-heterogeneity>

=== Summary

Byun and Lee study rank-heterogeneous federated LoRA, where selected clients train adapters with a
higher rank than the remaining clients @byunFederatedLowRankAdaptation2025. They identify
conventional zero-padding as the cause of a harmful aggregation effect: dimensions that occur only
in high-rank adapters are averaged with zeros from lower-rank clients and thus diluted. Their
server-side replication-based padding copies the corresponding high-rank components into the padded
portions of low-rank updates, preserving the high-rank signal during aggregation. In the reported
DistilBERT and ALBERT text classification experiments, it converges faster than homogeneous and
zero-padding baselines while retaining a low average communication cost.

=== Issues Addressed

The work combines the parameter efficiency of @background-lora[LoRA: Low-Rank Adaptation] with
@background-federated-learning[Federated Learning]. It addresses
@issue-resource-heterogeneity[Resource Heterogeneity and Configuration Adaptation] by assigning
larger ranks to clients judged to have higher-quality data, while retaining lower ranks for the
others. Differing ranks make LoRA factor shapes incompatible for direct aggregation.

The paper's particular aggregation problem is not simply non-IID data, discussed in
@issue-data-heterogeneity[Data Heterogeneity and Client Drift], but the dilution introduced when
missing rank dimensions are zero-padded. If a component $x$ occurs only in one high-rank client,
averaging it with $K - 1$ zeros changes it to $x / K$. This counteracts the intended priority given
to the higher-rank client. The approach also seeks the communication savings described in
@issue-communication-cost[Communication Cost and Synchronization] without assigning the largest rank
to every client.

=== Method

For a frozen layer, a client trains a LoRA update $Delta W = B A$, where $A in RR^(r times n)$ and
$B in RR^(m times r)$. All clients initially use rank $5$. After the first local-training round, the
server ranks clients by validation performance, assigns rank $20$ to the highest-performing $10$
percent, and leaves the other clients at rank $5$. In the experiments, validation performance is
used as a proxy for data quality.

Standard heterogeneous aggregation pads a lower-rank adapter to the maximum rank with zeros. With
replication, the server instead fills each missing factor component from the high-rank aggregate. In
a two-client example, a rank-$2$ factor $(a_(2,1), a_(2,2))$ padded to rank $4$ becomes
$(a_(2,1), a_(2,2), a_(1,3), a_(1,4)),$ rather than $(a_(2,1), a_(2,2), 0, 0)$, where the last two
entries come from the rank-$4$ client. Their average therefore retains $a_(1,3)$ and $a_(1,4)$
rather than halving them. With several high-rank clients, the server first aggregates those clients,
uses their extra components for replication, and applies aggregation weights that account for their
number. Replication is wholly server-side, so it does not make clients upload additional adapter
parameters.

The reported evaluation uses 100 simulated clients, 10 percent high-quality clients, one local epoch
per round, and 10 percent client participation per round. High-quality clients have more balanced
class distributions, whereas lower-quality clients have more imbalanced distributions generated with
different Dirichlet parameters. On AG's News and DBpedia with DistilBERT and ALBERT, the method is
compared with homogeneous ranks $5$, $7$, and $20$, naive zero-padding, and Frobenius-norm-weighted
zero-padding. For DistilBERT, the proposed configuration averages 179,715 LoRA parameters per client
(0.69 MB), versus 552,960 (2.11 MB) for homogeneous rank $20$, while reaching high accuracy in
substantially fewer rounds than the zero-padding baselines.

=== Pros and Cons

==== Pros

- Replication directly targets the documented source of the high-rank signal's dilution and is
  straightforward to implement at the server.
- It preserves the communication benefit of rank heterogeneity because client-side parameter counts
  and transmitted updates do not increase.
- The paper provides both an aggregation-level diagnostic---a reported first-round change from 84.34
  percent to 38.95 percent for a high-quality client under zero-padding, versus 82.11 percent with
  replication---and end-to-end convergence results.
- Assigning high ranks selectively gives a favorable resource--performance trade-off in the reported
  setting.

==== Cons

- The method relies on the assumption that selected high-rank clients are more informative. A poor
  rank-assignment decision can instead preserve or amplify an unrepresentative client's update.
- Validation performance after one round is an imperfect proxy for data quality; it can undervalue
  rare, small, or otherwise useful client datasets.
- The evaluation uses a binary rank allocation and simulated class-imbalance heterogeneity. It does
  not establish how the approach behaves with a continuum of ranks or other forms of data and device
  variation.
- Evidence is limited to two BERT-style models, two classification datasets, and 100 simulated
  clients. Its effectiveness for large generative-model instruction tuning and real client fleets
  remains untested.
