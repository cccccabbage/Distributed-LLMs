== Untrusted Permissionless Compute <issue-untrusted-permissionless-compute>

Permissionless distributed learning allows participants to join without a prior trust relationship,
but it also removes the assumption that every submitted result was produced by the requested model,
data, hardware configuration, and procedure. A participant may submit malformed, stale, copied,
low-quality, or deliberately harmful work while seeking aggregation influence or rewards. This
threat model applies whether the contribution is a local model update, gradient-like signal, or
inference rollout @lidin_incentivizing_2025 @team_intellect-2_2025.

Defenses usually combine inexpensive admissibility checks with a more costly assessment of value or
provenance. Admissibility checks can verify timely arrival, schema, model version, or basic output
properties. Value-based checks evaluate the effect of an update on held-out data, whereas
computation-oriented checks can compare execution evidence, such as model activations, against the
expected computation. Assigned or distinct tasks can also make copied contributions easier to
detect. The appropriate design depends on the submission type: an update may be judged by its effect
on a validation objective, while an inference result may need evidence that the required generation
was actually performed.

These mechanisms have limits. Validation can be statistical, sampled, expensive, or vulnerable to an
adaptive participant that targets the evaluator rather than the intended training objective.
Incentives may reward observed usefulness but do not themselves prove honest computation or provide
complete Byzantine robustness. Moreover, permissionless enrollment and verified contributions do not
make the entire system trustless: orchestration, discovery, validation, storage, reward
distribution, and model dissemination can remain centralized dependencies. A system should state
which workers and infrastructure components remain trusted, and which attacks its checks are meant
to deter rather than claim that decentralization automatically supplies security.
