== Incentivizing Permissionless Distributed Learning of LLMs <paper-gauntlet>

=== Summary

Gauntlet@lidinIncentivizingPermissionlessDistributed2025 is an incentive and validation mechanism
for synchronous, permissionless distributed training of foundation models. Rather than requiring
anonymous peers to prove that they followed an identical prescribed training procedure, validators
assess whether each submitted compressed update improves model loss and reward peers accordingly.
The mechanism combines fast reliability and synchronization checks, a longer-term OpenSkill quality
rating, and a statistical check intended to distinguish unique computation from copied updates. The
authors deployed it on Bittensor and report a live training run of a 1.2B-parameter LLM whose
per-iteration quality was competitive with their centralized baselines.

=== Issues Addressed

Gauntlet addresses the threat model summarized in @issue-untrusted-permissionless-compute[Untrusted
  Permissionless Compute] for compressed model updates. Its specific objective is to assign both
aggregation influence and economic rewards to useful peer contributions without enforcing an
identical amount of local computation.

The paper also addresses the communication burden summarized in
@issue-communication-cost[Communication Cost and Synchronization]. Its peers transmit compressed
pseudo-gradients rather than full uncompressed updates, but training remains synchronous, so slow or
poorly connected peers can still be disadvantaged. Its permissionless setting additionally makes the
compute differences discussed in @issue-resource-heterogeneity[Resource Heterogeneity and
  Configuration Adaptation] an incentive-design concern: a peer able to process more data can submit
a stronger update and receive a larger reward.

=== Method

At round $t$, peers begin from an approximately shared model and locally train on assigned data.
Peer $p$ submits a pseudo-gradient $Delta_p$, a compressed model update produced with the DeMo
optimizer's error feedback, discrete cosine transform, and top-$k$ compression. Validators first
apply inexpensive checks for timely submission, valid tensor shape and data, and model
synchronization. The synchronization check compares a small sample of peer parameters with the
validator's state; the reported implementation tolerates roughly three update differences.

For a more costly primary evaluation, a validator measures the change in held-out loss when applying
an individual peer's update. In simplified form, the quality signal is the loss difference

$
  L(theta_t, D) - L(theta_t - beta Delta_p, D),
$

where a positive value indicates that the update reduced loss on evaluation data $D$. Because an
individual measurement is noisy, Gauntlet uses relative rankings from these loss scores to maintain
an OpenSkill `LossRating` over time.

To discourage copying, every peer receives a distinct data subset. Validators compare the update's
loss improvement on that assigned data with its improvement on random data and accumulate the
resulting statistical evidence as $mu_p$. The final score is proportional to

$
  "PeerScore"_p = mu_p times "LossRating"_p.
$

Scores determine nonlinear rewards (with exponent $2$ in the reported implementation) and select the
updates used for training. In the live experiment, only the top $G = 15$ peers were aggregated, with
equal weight $1 / G$; the rest received zero aggregation weight. Gradient normalization and
sign-based aggregation reduce the influence of simple magnitude-scaling attacks. Updates are shared
through S3-compatible object storage rather than a bespoke peer-to-peer transport.

=== Pros and Cons

==== Pros

- Gauntlet evaluates useful outcomes instead of mandating uniform hardware, local-data volume, or
  optimizer behavior, which suits a permissionless and heterogeneous participant pool.
- The two-stage validator design applies cheap checks broadly and reserves more expensive loss
  evaluations for a smaller subset, limiting validation overhead.
- Loss ratings smooth noisy per-round observations, while assigned-data evaluation gives a concrete,
  though statistical, deterrent to submitting another peer's update unchanged.
- The work includes a live, token-rewarded 1.2B-model training deployment rather than only a
  simulation or theoretical mechanism.

==== Cons

- Assigned-data evidence is not a cryptographic proof of computation; a sophisticated participant
  may still game the evaluation signal or avoid immediate detection when primary evaluation is
  sampled. The broader limits of these defenses are discussed in
  @issue-untrusted-permissionless-compute[Untrusted Permissionless Compute].
- The defenses do not provide complete Byzantine robustness. A harmful update can influence the
  model before the peer's rating falls, and normalization mainly limits simple magnitude attacks.
- Historical OpenSkill ratings can lag a peer's sudden change in behavior, while the synchronous
  design can penalize slow hardware or high-latency participants.
- The reported full training run is limited to a 1.2B-parameter model, leaving efficiency and
  security at much larger scales unestablished. Its nonlinearly concentrated rewards may also favor
  a small number of powerful operators.
- S3-compatible storage and coordination information from the highest-staked validator simplify the
  deployment but leave centralized dependencies; permissionless worker enrollment is therefore not
  equivalent to complete system decentralization.
