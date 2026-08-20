== Data Heterogeneity and Client Drift <issue-data-heterogeneity>

Federated and decentralized learning commonly operate on non-independent and identically distributed
(non-IID) client data. Clients may differ in domain, task, language, label distribution, data
volume, or style. These differences can be useful: together they can broaden the knowledge and
capabilities represented in the shared model. They also mean that clients optimize different local
objectives @zhang_towards_2024 @ghiasvand_decentralized_2025.

When clients perform several local updates before communicating, their parameters can move in
different directions, a phenomenon usually called client drift. Aggregating these divergent updates
can slow or destabilize convergence and can reduce performance for some clients. Increasing the
number of local steps lowers communication frequency but generally increases the risk of drift.

In decentralized systems, the communication graph adds a second source of disagreement. Sparse or
poorly connected graphs limit per-client communication, but information and model changes propagate
slowly; more connected graphs improve consensus while increasing communication cost. The practical
choice of local-update count, aggregation rule, and network topology therefore trades communication
against consensus and accuracy @ghiasvand_decentralized_2025.
