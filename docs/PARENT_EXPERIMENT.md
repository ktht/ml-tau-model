# Parent-conditioned ParTauDETR experiment

## Existing path and targets

`ParTauDETR_dataloader.py` reads detector candidates and reconstructed-jet
references. Its daughter targets are jet-relative
`[log(pt/jet_pt), eta-jet_eta, sin(dphi), cos(dphi), log(m/jet_m)]`,
with log ratios clipped to [-5, 5]. The ParticleTransformer encoder supplies
particle memory and a pooled token. The DETR decoder has per-query objectness,
kinematics, charge and meson-class heads. `HungarianMatcher` uses weighted L1
kinematics and classification costs. `TauLoss` retains the configured scaled
Huber residuals, including the periodic phi chord. No Cartesian L2 daughter
loss is introduced.

`SetCriterion` reconstructs parent momentum either from Hungarian-assigned
queries or from all queries weighted by the calibrated logit-space sigmoid
gate. Charge and decay-mode constraints have their own probability sums.
Background jets supervise only tau identification, not reconstruction.

The parent is **generator visible tau**, `gen_jet_tau_p4`, not full tau.
This is documented in TECHNICAL.md and the evaluation code. No latent neutrino
is justified here. However, species filtering, optional daughter pT cuts and
slot truncation can remove visible momentum without changing that parent.
The opt-in raw target sum audit measures this discrepancy before interpreting
closure as a daughter-level constraint. This audit uses actual retained p4,
not a decode of the clipped kinematic targets. The data-production definition
has not been verified against a parquet sample in this change.

The existing jet validation helper treats daughters as massless; the parent
training loss does not. Baseline behavior is unchanged. Experimental physical
daughter vectors bypass that approximation in jet validation and external
prediction decoding.

## Why not a fixed tetrahedron?

No tetrahedral coordinate implementation was found in the searchable repository
Python, YAML or Markdown sources. There is therefore no existing axis convention
to reuse. A mathematically valid construction would take unit tetrahedral
directions n_a with sum_a n_a=0 and rest-frame null vectors
v_a=(M n_a/4, M/4), using this repository's (px,py,pz,E) ordering.
They sum to the parent rest vector. Nonnegative coefficients preserve the future
cone by the triangle inequality:

    |sum_a F_ia v_a.spatial| <= sum_a F_ia |v_a.spatial|
                              <= sum_a F_ia v_a.energy.

Positive energy follows for nonzero coefficients. To also preserve the parent,
one needs **sum_i F_ia=1 for each vertex a**, i.e. softmax over queries, not over
vertices. Row normalization alone supplies neither this closure nor a physical
basis. A negative remainder can be spacelike or past-directed even when every
preceding daughter is physical.

The fixed tetrahedron has a serious expressivity limitation: daughter velocities
lie inside its inscribed tetrahedron, not the full unit ball. In particular,
arbitrary opposite relativistic daughter directions are inaccessible. Rather
than force that basis onto tau decays, the implemented alternative uses learned
physical seeds with scalar energy fractions and three independent velocity
coordinates. It is not claimed to be a tetrahedral implementation.

## Physical fractions and closure

The pooled token predicts five parent coordinates relative to the reconstructed
jet. The parent is decoded directly and supervised by the existing parent Huber
helper. Each query independently predicts four unconstrained coordinates:

    f_i = softmax_queries(z_i0)
    u_i = (z_i1, z_i2, z_i3)
    v_i = u_i / sqrt(1 + |u_i|^2)
    q_i = (f_i v_i, f_i).

The velocity coordinates are bounded to [-100,100] for numerical protection.
For finite logits in exact arithmetic, f_i>0, sum_i f_i=1 and |v_i|<1. Hence
q_i.energy>0 and q_i^2=f_i^2/(1+|u_i|^2)>0. This is a physicality proof, not an
assumption about softmax. Extremely separated logits can underflow to zero in
floating point; a zero component remains in the closed future cone.

In `fractions` mode, multiply seeds by the predicted parent mass and boost them
from the parent rest frame. They are physical, but their sum need not equal the
parent because the seed system can have net spatial momentum. This is the
nonstructural-closure control. An optional Huber closure penalty compares the
sum against the direct parent in the same relative kinematic representation.

In `closure` mode, let Q=sum_i q_i and mu=sqrt(Q^2)>0. First apply the inverse
boost taking Q to (0,mu); multiply every seed by M_parent/mu; then apply the boost
taking (0,M_parent) to the predicted parent. Linearity gives:

    sum_i p_i = B_parent [(M_parent/mu) B_Q_inverse Q] = P_parent.

Every transformation preserves future-directed physicality. This imposes exact
all-component closure in real arithmetic, up to rounding in implementation.
It avoids an asymmetric remainder and a competing closure penalty. Gradients
couple daughter and parent kinematics without going through objectness gates.
The boost algebra is in float64 to protect invariant masses at high boost.
The experimental parent has |eta|<=10 and mass >= max(0.001, |p|*1e-4), in the
dataset momentum units. These numerical bounds are explicit model restrictions;
they do not fix the parent to the full tau mass. Daughter masses are not forced
onto individual species mass shells.

**Closure uses exactly the target multiplicity during supervision.** For each
signal event with N retained targets, the matcher tries every N-element subset
of the Q candidate queries. Each subset is independently normalized and closed
before computing the existing objectness, kinematic, charge and meson costs.
Hungarian solves the assignment within each subset; the smallest total cost
wins (lexicographically first subset on an exact tie). The winning construction
is recomputed with gradients. Unselected queries have exactly zero four-momentum
and receive no-object supervision. Selection remains discrete, as in ordinary
Hungarian matching. There is no heuristic pruning or ambiguous-case fallback:
all binomial(Q,N) subsets are evaluated, grouped by multiplicity across events.
This becomes expensive for large Q; the baseline matcher is unchanged.

Inference has no target count. It uses the calibrated objectness-selected subset,
renormalizes only its fractions and closes that subset. Threshold changes in
external evaluation reconstruct the subset again, rather than cutting already
constructed vectors. Supervision uses a local output copy so truth-selected
momenta never leak into inference validation. Empty selections return zero:
an empty set cannot sum to a nonzero parent. For one selected component, closure
fixes its momentum to the parent and its fraction coordinates are unidentifiable.
Multiplying the closed set by additional soft gates still breaks closure; the
soft-gate loss remains an explicitly separate ablation.

Retained target momentum must still be compatible with the visible-parent
target. With filtering or truncation, unmatched queries can no longer absorb
omitted visible energy. Inspect the raw target closure audit and redefine the
parent target or retain the missing daughters if necessary. No neutrino or
unconstrained remainder is silently introduced.

Exact vectors remain in `pred_daughter_p4`. `encode_p4` converts them back to
the existing clipped coordinates for Hungarian matching and daughter losses.
Parent sums and experimental inference consume the exact vectors instead of
round-tripping through those lossy coordinates. Closure mode extends assignment
to a subset-dependent construction; the matching cost definition is unchanged.

## Ablations

Use Hydra overrides with the existing training command. All settings live under
`model.detr`. Defaults preserve the old architecture and objective.

| Run | Overrides |
| --- | --- |
| A | `parent_experiment.mode=baseline` |
| B | `parent_experiment.mode=direct` |
| C | `parent_experiment.mode=fractions` |
| D | `parent_experiment.mode=closure` |
| E | Repeat B/C/D with `parent_experiment.hungarian_losses=false` |
| F | Repeat with `loss.weight_soft_parent_kinematics=1` and `loss.detach_momentum_gate=false/true` |

Prefix each override above with `model.detr.`. `weight_direct` defaults to 1.
For penalty versus structural closure, compare C with
`parent_experiment.weight_closure=1` to D with `weight_closure=0`.
C is already physical: an unphysical control is not needed to isolate closure.
For a pure direct-parent-only objective in B, also set all pre-existing
`weight_parent_*` and `weight_soft_parent_*` values to zero; otherwise they remain
the explicitly configured auxiliary constraints for controlled comparisons.

E removes Hungarian-dependent objectness, daughter regression, charge/meson
classification and matched-parent penalties from the total. Matching is still
computed for diagnostics and individual losses are still logged. Existing
nonmatching auxiliary losses, direct-parent loss and tau identification remain.
Without objectness supervision, selection calibration may degrade; report this
rather than interpreting arbitrary selected sets as trained constituents.

F detaches gates **only for the soft parent momentum term**. Forward loss values
are identical for fixed predictions. Charge/decay-mode losses still train gates,
and shared encoder gradients can still affect objectness indirectly. In C/D,
fraction energy allocation offers another way to redistribute momentum: this
ablation isolates the objectness-gate route, not every allocation mechanism.

## Diagnostics and interpretation

Enable `model.detr.parent_experiment.diagnostics=true` for A-F. The added work
is disabled by default. `matching/hard_iou` measures the Hungarian set versus
the calibrated hard cut (equivalently gate>=0.5); `matching/soft_iou` provides
the continuous analogue. Added `train/experiment/*` and `val/experiment/*`
metrics include log-pT/log-mass biases and MAEs, eta/wrapped-phi residuals,
configured Huber loss and residuals split at IoU=0.5, for:

- Hungarian-selected, hard-selected and soft-gated sums;
- all physical components and the direct parent where enabled;
- the retained raw target-daughter sum when supplied by the loader;
- the best target-count-matched subset under the configured parent Huber loss.

In closure mode, `hungarian` and `all_components` refer to the winning
target-sized closed system; `hard_gate` is independently reconstructed using
inference selection. The fixed-candidate parent oracle below is disabled in
closure mode: it would be invalid to cut an existing decomposition, and every
reclosed nonempty subset has identical parent momentum. Daughter matching cost,
not parent residual, distinguishes these subsets. Thus parent closure alone
cannot diagnose daughter association quality in this mode.

The oracle enumerates 2^Q subsets, retains subsets of the target cardinality,
and uses truth only as a diagnostic. It is exact for this specified objective
and cardinality, not an inference algorithm or a universal lower bound on other
residual metrics. It runs only when Q<=oracle_max_queries (default 8, hard cap
12). Set that option to 0 to disable. Diagnostic costs can be substantial.

Charge agreement and a charge-count-derived decay-mode agreement are provided
for matched/hard-selected sets. The existing `val_jet` metrics remain the
authoritative upstream species-based decay classification. Direct-parent,
daughter and closure losses are separate; the existing individual daughter and
matched/soft-parent loss logs are retained.

Per-event signal flags, IoU and residual/loss tensors are available in
`criterion.last_parent_diagnostics` for the most recent batch. Set
`model.detr.parent_experiment.diagnostic_dir=/path/to/run/diagnostics` to persist
every nonsanity validation batch as epoch/rank/batch `.pt` files on CPU. Use a
distinct directory per run; repeated epochs overwrite the same batch filenames.
Only load trusted torch artifacts. Plot residuals versus IoU using the signal
mask, and inspect distributions rather than only means. Empty IoU strata log
zero; `low_iou_fraction` identifies their occupancy.

If the oracle is good but Hungarian momentum is poor, selection is limiting.
If both are poor, available daughter candidates or target completeness are
limiting. If direct parent is good but selected sums are poor, inspect unmatched
energy. Compare normal/detached F runs using matched daughter and parent errors,
not just the optimized soft-parent loss. No such empirical conclusion can be
drawn without checkpoint/data evaluation or training these ablations.

## Validation

Focused tests: `python -m unittest discover -s tests -p test_partau_parent.py`.
They exercise physicality, closure, nonclosure, permutation equivariance,
boost inversion, kinematic round trips, finite gradients, gate detachment and
the actual criterion diagnostics/oracle path. Subset tests check exact target
cardinality, zero inactive momentum, empty-set finite gradients, and the optimal
subset/assignment against exhaustive query permutations for mixed target counts.
The environment setup was skipped
in this editing session, so these tests and training comparisons have **not**
been executed. Editor diagnostics reported no errors in the changed files.