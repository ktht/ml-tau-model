import math
import warnings
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from mltau.models.ParTauDETR import ParTauDETR
from mltau.tools.io.general import BatchInputs
from mltau.tools.logging import set_to_set as s2s
from mltau.tools.losses import TauLoss
from mltau.tools.meson_classes import MesonClass, get_meson_classes
from mltau.tools.partau_detr import decode_kinematics

try:  # scipy's LAPJVsp solver is ~10x faster than the pure-python fallback below
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    _scipy_lsa = None


def _hungarian_rect_min_cost(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    """Hungarian algorithm for rectangular cost matrices (n_rows <= n_cols)."""
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows > 0 else 0

    if n_rows == 0 or n_cols == 0:
        return [], []
    if n_rows > n_cols:
        raise ValueError("_hungarian_rect_min_cost expects n_rows <= n_cols.")

    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1

    rows = list(range(n_rows))
    cols = assignment
    return rows, cols


def _solve_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost bipartite matching for one [n_rows, n_cols] cost matrix."""
    if _scipy_lsa is not None:
        return _scipy_lsa(cost)

    n_rows, n_cols = cost.shape
    if n_rows <= n_cols:
        rows, cols = _hungarian_rect_min_cost(cost.tolist())
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    rows_t, cols_t = _hungarian_rect_min_cost(cost.T.tolist())
    pred_idx = np.asarray(cols_t, dtype=np.int64)
    tgt_idx = np.asarray(rows_t, dtype=np.int64)
    order = np.argsort(pred_idx)  # scipy returns row indices in ascending order
    return pred_idx[order], tgt_idx[order]


CLASSIFICATION_COSTS = ("prob", "nll", "clipped_nll")


class HomoscedasticLossWeighting(nn.Module):
    """Learn task weights through homoscedastic log variances."""

    def __init__(
        self,
        initial_weights: dict[str, float],
        priorities: dict[str, float] | None = None,
    ):
        super().__init__()
        # Inspired by Kendall et al., arXiv:1705.07115, with s = log(sigma^2).
        # Their regression factor 0.5 only shifts s by a constant and does not
        # change optimization of the effective weight, so all terms use exp(-s).
        # Each task contributes exp(-s) * L + 0.5 * c * s, with effective
        # weight w = exp(-s) and configured priority c. Its gradient is
        # d/ds = 0.5*c - w*L, so each task independently moves towards
        # w*L = 0.5*c. The default c=1 recovers the paper-inspired balance.
        #
        # This balances loss magnitudes, not necessarily gradient magnitudes or
        # downstream performance: tasks still interact through the shared model
        # parameters. The +0.5*c*s term prevents every w collapsing to zero.
        self.log_variances = nn.ParameterDict()
        self._active_terms: dict[str, torch.Tensor] = {}
        priorities = priorities or {}
        unknown_priorities = priorities.keys() - initial_weights.keys()
        if unknown_priorities:
            raise ValueError(
                "Automatic loss-weight priorities contain unknown terms: "
                f"{sorted(unknown_priorities)}"
            )
        self.priorities: dict[str, float] = {}
        for name, weight in initial_weights.items():
            if weight <= 0:
                continue
            priority = float(priorities.get(name, 1.0))
            if priority <= 0:
                raise ValueError(
                    f"Automatic loss-weight priority for {name!r} must be positive."
                )
            self.priorities[name] = priority
            # Choose s so exp(-s) equals the configured initial weight.
            initial_log_variance = -math.log(weight)
            self.log_variances[name] = nn.Parameter(
                torch.tensor(initial_log_variance, dtype=torch.float32)
            )

    def effective_weights(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.exp(-log_variance)
            for name, log_variance in self.log_variances.items()
        }

    def forward(
        self,
        losses: dict[str, torch.Tensor],
        supervision_counts: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        if not self.log_variances:
            return next(iter(losses.values())).new_zeros(())

        total = next(iter(losses.values())).new_zeros(())
        for name, log_variance in self.log_variances.items():
            if name not in losses:
                raise KeyError(f"Missing loss for automatic weight {name!r}.")
            count = torch.as_tensor(
                supervision_counts[name], device=losses[name].device
            )
            is_supervised = (count > 0).to(dtype=losses[name].dtype)
            if torch.is_grad_enabled():
                active = (count > 0) & (losses[name].detach() > 0)
                previous = self._active_terms.get(name)
                self._active_terms[name] = (
                    active if previous is None else previous | active
                )
            # At exact equilibrium the log-variance gradient is already zero.
            # A raw loss of zero is different: only the +0.5*s term remains and
            # would increase the effective weight without bound. Freeze s for
            # that batch while preserving the loss's model-gradient path.
            active_log_variance = torch.where(
                losses[name].detach() > 0,
                log_variance,
                log_variance.detach(),
            )
            # The 0.5*c*s term prevents the learned weight collapsing to zero.
            total = total + is_supervised * (
                torch.exp(-active_log_variance) * losses[name]
                + 0.5 * self.priorities[name] * active_log_variance
            )
        return total

    def freeze_inactive_log_variances(self) -> None:
        """Prevent AdamW momentum from moving unsupervised or zero-loss terms."""
        for name, log_variance in self.log_variances.items():
            active = self._active_terms.get(name)
            if active is not None and not bool(active):
                log_variance.grad = None
        self._active_terms.clear()


def _classification_cost_matrix(
    pred_logits: torch.Tensor,
    target_classes: torch.Tensor,
    ignore_index: int = -100,
    kind: str = "clipped_nll",
    clip: float = 5.0,
) -> torch.Tensor:
    """
    Per-query/per-target classification matching cost.

    Three forms, all functions of p = p(true class of the target):

      "nll"          -log p          unbounded (the first model's choice)
      "prob"         1 - p           bounded in [0, 1] (DETR's own choice)
      "clipped_nll"  min(-log p, clip)   bounded, but sharp where it matters

    The cost decides WHICH daughter a query is trained towards, so the question
    is how much a confident identity should count against a position error.
    Tau daughters are collimated: while the model's angular error is about one
    target sigma (~0.07), position cannot tell two daughters of a jet apart and
    identity and pt -- which are learned first -- have to carry the decision.
    On synthetic daughters with realistic spreads, 1-sigma kinematic noise and a
    3% class error rate, the intended assignment was recovered:

        unweighted L1 + nll (first model)     93%, 11% churn between noise draws
        1/sigma-weighted L1 + prob            71%, 46% churn
        [1,5,5,5,0.2] L1 + clipped nll (5)    94%, 10% churn   <- default

    "prob" is too weak here: a wrong class costs at most 1 while one sigma of
    angular noise costs ~1.5 through the kinematics term, so identity loses.
    "nll" is unbounded: a confidently wrong query (p -> 0) pays an arbitrarily
    large cost, and late in training, when predictions are sharp, the class
    share of the total cost was measured to grow from 26% to 62%, i.e. the
    assignment starts being decided by predicted identity rather than by
    position, and a single confident mistake can flip a whole jet's assignment
    (the late-training loss spikes). The clip keeps the useful range -- a wrong
    class at p = 0.01 still costs 4.6, a soft p = 0.6 costs 0.5 -- and removes
    the tail.

    Args:
        pred_logits: [B, Q, C]
        target_classes: [B, T]
        kind: one of CLASSIFICATION_COSTS.
        clip: upper bound of the per-pair cost for "clipped_nll".
    Returns:
        cost: [B, Q, T] in float32, zero where the target is `ignore_index`.
    """
    # Built in fp32 for AMP stability.
    log_p = F.log_softmax(pred_logits.float(), dim=-1)  # [B, Q, C]
    if kind == "prob":
        per_class = 1.0 - log_p.exp()
    elif kind == "nll":
        per_class = -log_p
    elif kind == "clipped_nll":
        per_class = (-log_p).clamp(max=float(clip))
    else:
        raise ValueError(
            f"classification cost must be one of {CLASSIFICATION_COSTS}, got {kind!r}"
        )
    valid = target_classes != ignore_index  # [B, T]
    # `gather` needs in-range indices even for the entries we discard afterwards.
    idx = target_classes.clamp_min(0).unsqueeze(1).expand(-1, per_class.size(1), -1)
    cost = torch.gather(per_class, 2, idx)  # [B, Q, T]
    return cost * valid.unsqueeze(1)


class HungarianMatcher(nn.Module):
    """DETR-style matcher with mixed regression/classification costs."""

    # Cost charged to padded target slots. Constant across queries, so it only
    # shifts the objective by a constant and leaves the optimum over the real
    # targets untouched -- this lets every jet be solved on one fixed [Q, T]
    # matrix instead of a per-jet compacted one.
    _PAD_COST = 1.0e6

    def __init__(
        self,
        cost_objectness: float = 1.0,
        cost_kinematics_l1: float = 2.0,
        cost_charge_ce: float = 1.0,
        cost_meson_class_ce: float = 1.0,
        object_class_index: int = 0,
        ignore_index: int = -100,
        kinematics_component_weights: list[float] | None = None,
        classification_cost: str = "clipped_nll",
        classification_cost_clip: float = 5.0,
    ):
        super().__init__()
        if classification_cost not in CLASSIFICATION_COSTS:
            raise ValueError(
                f"matcher.classification_cost must be one of {CLASSIFICATION_COSTS}, "
                f"got {classification_cost!r}"
            )
        self.classification_cost = classification_cost
        self.classification_cost_clip = float(classification_cost_clip)
        # Mean of each cost term over the pairs actually chosen, refreshed every
        # forward. The criterion reads it so the split can be logged: if the
        # classification share climbs while the losses blow up, the matcher has
        # started matching by predicted identity instead of by position.
        self.last_cost_terms: dict[str, float] = {}
        self.cost_objectness = cost_objectness
        self.cost_kinematics_l1 = cost_kinematics_l1
        # Per-component weights for the kinematics matching cost, over
        # [log_pt_ratio, delta_eta, sin_dphi, cos_dphi, log_mass_ratio].
        #
        # Unweighted, the log ratios (sigma ~1) dominate and the angular offsets
        # of collimated daughters (sigma ~0.07) barely count. Weighting the
        # angles at 1/sigma (20) over-corrects: while the model's angular error
        # is still of order sigma, position is noise and an angle-dominated
        # cost reassigns queries between steps (71% intended-assignment recovery
        # and 46% churn under 1-sigma noise, against 93% / 11% unweighted).
        # A moderate weight (5, set in the config) keeps direction in the cost
        # without letting it override pt and identity: 94% / 10%. See
        # _classification_cost_matrix for the measurement.
        if kinematics_component_weights is None:
            kinematics_component_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.register_buffer(
            "kinematics_component_weights",
            torch.tensor([float(w) for w in kinematics_component_weights]),
        )
        self.cost_charge_ce = cost_charge_ce
        self.cost_meson_class_ce = cost_meson_class_ce
        self.object_class_index = object_class_index
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_kinematics: torch.Tensor,
        pred_charge_logits: torch.Tensor,
        pred_meson_class_logits: torch.Tensor,
        target_kinematics: torch.Tensor,
        target_charge_cls: torch.Tensor,
        target_meson_class: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_logits: [B, Q, 2]
            pred_kinematics: [B, Q, K]
            pred_charge_logits: [B, Q, C_charge]
            pred_meson_class_logits: [B, Q, C_meson]
            target_kinematics: [B, T, K]
            target_charge_cls: [B, T]
            target_meson_class: [B, T]
            target_mask: [B, T]

        Returns:
            (batch_idx, query_idx, target_idx), three flat int64 tensors of equal
            length listing every matched (jet, query, target-slot) triplet.
        """
        device = pred_logits.device
        batch_size, num_queries, _ = pred_logits.shape
        num_targets = target_mask.size(1)

        # Cleared every call: a batch without a single match must not re-log
        # the previous batch's cost split as if it were current.
        self.last_cost_terms = {}

        empty = torch.empty(0, dtype=torch.long, device=device)
        if num_targets == 0 or batch_size == 0:
            return empty, empty, empty

        # ------------------------------------------------------------------
        # All costs are built for the whole batch at once, in fp32 to avoid AMP
        # dtype mismatches. Roughly a dozen kernels per step instead of ~10 per
        # jet, and no host synchronisation until the single transfer below.
        # ------------------------------------------------------------------
        obj_cost = -F.log_softmax(pred_logits.float(), dim=-1)[
            ..., self.object_class_index
        ]  # [B, Q]
        # Weighted L1 by explicit broadcast: torch.cdist(p=1) has no fast kernel
        # and could not apply per-component weights anyway.
        component_diff = (
            pred_kinematics.float().unsqueeze(2)
            - target_kinematics.float().unsqueeze(1)
        ).abs()  # [B, Q, T, K]
        weights = self.kinematics_component_weights.to(component_diff.dtype)
        if weights.numel() != component_diff.size(-1):
            raise ValueError(
                f"matcher.kinematics_component_weights has {weights.numel()} "
                f"entries but the kinematics target has "
                f"{component_diff.size(-1)} components."
            )
        kin_cost = (component_diff * weights).sum(-1)  # [B, Q, T]
        charge_cost = _classification_cost_matrix(
            pred_charge_logits,
            target_charge_cls,
            self.ignore_index,
            self.classification_cost,
            self.classification_cost_clip,
        )
        meson_class_cost = _classification_cost_matrix(
            pred_meson_class_logits,
            target_meson_class,
            self.ignore_index,
            self.classification_cost,
            self.classification_cost_clip,
        )

        total_cost = (
            self.cost_objectness * obj_cost.unsqueeze(-1)
            + self.cost_kinematics_l1 * kin_cost
            + self.cost_charge_ce * charge_cost
            + self.cost_meson_class_ce * meson_class_cost
        )
        total_cost = torch.nan_to_num(total_cost, nan=0.0, posinf=1e4, neginf=-1e4)
        total_cost = total_cost.masked_fill(
            ~target_mask.unsqueeze(1), self._PAD_COST
        )

        # One device -> host transfer per step for the whole batch.
        cost_np = total_cost.cpu().numpy()
        valid_np = target_mask.cpu().numpy()
        to_solve = np.flatnonzero(valid_np.any(axis=1))
        if to_solve.size == 0:
            return empty, empty, empty

        # Every solve returns exactly min(Q, T) pairs, so the results pack into
        # a dense array and the padded slots are dropped in one vectorised pass.
        num_pairs = min(num_queries, num_targets)
        query_idx = np.empty((to_solve.size, num_pairs), dtype=np.int64)
        target_idx = np.empty((to_solve.size, num_pairs), dtype=np.int64)
        for i, b in enumerate(to_solve):
            rows, cols = _solve_assignment(cost_np[b])
            query_idx[i] = rows
            target_idx[i] = cols

        batch_idx = np.repeat(to_solve, num_pairs)
        query_idx = query_idx.reshape(-1)
        target_idx = target_idx.reshape(-1)
        keep = valid_np[batch_idx, target_idx]

        # Cost split over the pairs actually chosen. Detached scalars only, so
        # this costs four reductions and holds no graph.
        b_sel = torch.from_numpy(batch_idx[keep]).to(device)
        q_sel = torch.from_numpy(query_idx[keep]).to(device)
        t_sel = torch.from_numpy(target_idx[keep]).to(device)
        if b_sel.numel() > 0:
            with torch.no_grad():
                terms = {
                    "objectness": self.cost_objectness * obj_cost[b_sel, q_sel],
                    "kinematics": self.cost_kinematics_l1 * kin_cost[b_sel, q_sel, t_sel],
                    "charge": self.cost_charge_ce * charge_cost[b_sel, q_sel, t_sel],
                    "meson_class": self.cost_meson_class_ce
                    * meson_class_cost[b_sel, q_sel, t_sel],
                }
                means = {k: float(v.mean()) for k, v in terms.items()}
                total = sum(means.values()) or 1.0
                self.last_cost_terms = {
                    **{f"cost/{k}": v for k, v in means.items()},
                    # The share is the number that matters: it is what shifts as
                    # the model sharpens.
                    "cost/class_share": (means["charge"] + means["meson_class"]) / total,
                }

        return (
            torch.from_numpy(batch_idx[keep]).to(device, non_blocking=True),
            torch.from_numpy(query_idx[keep]).to(device, non_blocking=True),
            torch.from_numpy(target_idx[keep]).to(device, non_blocking=True),
        )


class SetCriterion(nn.Module):
    """
    DETR-style criterion: objectness + kinematics + charge + meson-class losses
    on the matched daughters, the jet-level tauID loss, and auxiliary penalties
    (consistency, charge count, and the parent constraints).

    Division of labour, fixed by design:

    - Whether a jet is a tau at all is decided by the tauID head ONLY. It is the
      one term background jets contribute to, and the one term `jet_weights`
      (cls_weight) applies to: that weight matches the (theta, p) spectra of
      signal and background so the tagger cannot key on pt and theta, and it
      has no other purpose.
    - Objectness answers "which queries are real daughters of this tau", and
      together with kinematics, charge, meson class and the parent constraints
      it is trained on signal jets only and UNWEIGHTED. Reweighting the
      reconstruction would reshape the pt spectrum the regression and
      classification heads are optimised on without decorrelating anything;
      the first model trained them unweighted (its production had no
      cls_weight column) and that is the behaviour kept.
    - Background jets are deliberately NOT pushed towards "no object": on a
      background jet objectness is untrained and its scores are meaningless, so
      inference and evaluation gate the daughter set with the tauID score.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        tau_loss: TauLoss,
        meson_classes: tuple[MesonClass, ...],
        loss_objectness_weight: float = 1.0,
        loss_tau_id_weight: float = 1.0,
        loss_kinematics_weight: float = 5.0,
        loss_charge_weight: float = 1.0,
        loss_meson_class_weight: float = 1.0,
        loss_consistency_weight: float = 0.0,
        loss_charge_count_weight: float = 0.0,
        loss_parent_kinematics_weight: float = 0.0,
        loss_parent_charge_weight: float = 0.0,
        loss_parent_decay_mode_weight: float = 0.0,
        loss_soft_parent_kinematics_weight: float = 0.0,
        loss_soft_parent_charge_weight: float = 0.0,
        loss_soft_parent_decay_mode_weight: float = 0.0,
        parent_objectness_temperature: float = 0.1,
        automatic_weight_optimization: bool = False,
        automatic_weight_priorities: dict[str, float] | None = None,
        no_object_class_index: int = 1,
        object_class_index: int = 0,
        eos_coef: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.matcher = matcher
        self.tau_loss = tau_loss
        self.loss_objectness_weight = loss_objectness_weight
        self.loss_tau_id_weight = loss_tau_id_weight
        self.loss_kinematics_weight = loss_kinematics_weight
        self.loss_charge_weight = loss_charge_weight
        self.loss_meson_class_weight = loss_meson_class_weight
        self.loss_consistency_weight = loss_consistency_weight
        self.loss_charge_count_weight = loss_charge_count_weight
        self.loss_parent_kinematics_weight = loss_parent_kinematics_weight
        self.loss_parent_charge_weight = loss_parent_charge_weight
        self.loss_parent_decay_mode_weight = loss_parent_decay_mode_weight
        self.loss_soft_parent_kinematics_weight = loss_soft_parent_kinematics_weight
        self.loss_soft_parent_charge_weight = loss_soft_parent_charge_weight
        self.loss_soft_parent_decay_mode_weight = loss_soft_parent_decay_mode_weight
        configured_loss_weights = {
            "objectness": loss_objectness_weight,
            "tau_id": loss_tau_id_weight,
            "kinematics": loss_kinematics_weight,
            "charge": loss_charge_weight,
            "meson_class": loss_meson_class_weight,
            "consistency": loss_consistency_weight,
            "charge_count": loss_charge_count_weight,
            "parent_kinematics": loss_parent_kinematics_weight,
            "parent_charge": loss_parent_charge_weight,
            "parent_decay_mode": loss_parent_decay_mode_weight,
            "soft_parent_kinematics": loss_soft_parent_kinematics_weight,
            "soft_parent_charge": loss_soft_parent_charge_weight,
            "soft_parent_decay_mode": loss_soft_parent_decay_mode_weight,
        }
        self.configured_loss_weights = configured_loss_weights
        matched_parent_enabled = any(
            configured_loss_weights[name] > 0
            for name in (
                "parent_kinematics",
                "parent_charge",
                "parent_decay_mode",
            )
        )
        soft_parent_enabled = any(
            configured_loss_weights[name] > 0
            for name in (
                "soft_parent_kinematics",
                "soft_parent_charge",
                "soft_parent_decay_mode",
            )
        )
        if automatic_weight_optimization and not any(
            weight > 0 for weight in configured_loss_weights.values()
        ):
            raise ValueError(
                "Automatic loss-weight optimization requires at least one positive "
                "configured loss weight."
            )
        if (
            automatic_weight_optimization
            and matched_parent_enabled
            and soft_parent_enabled
        ):
            raise ValueError(
                "Automatic loss-weight optimization supports only one parent "
                "constraint family at a time. Set either all weight_parent_* or "
                "all weight_soft_parent_* values to zero."
            )
        self.loss_weighting = (
            HomoscedasticLossWeighting(
                configured_loss_weights,
                priorities=automatic_weight_priorities,
            )
            if automatic_weight_optimization
            else None
        )
        if not 0.0 < parent_objectness_temperature < 1.0:
            raise ValueError("parent_objectness_temperature must be between 0 and 1.")
        self.parent_objectness_temperature = parent_objectness_temperature
        self.no_object_class_index = no_object_class_index
        self.object_class_index = object_class_index
        self.eos_coef = eos_coef
        self.ignore_index = ignore_index

        # Charge classes are ordered [-1, 0, +1]. Meson-class columns follow
        # configuration order, with allowed combinations declared per class.
        n_charge = 3
        valid = torch.zeros(n_charge, len(meson_classes), dtype=torch.float32)
        for meson_class_index, meson_class in enumerate(meson_classes):
            for charge in meson_class.charges:
                valid[charge + 1, meson_class_index] = 1.0
        self.charge_meson_class_valid: torch.Tensor
        self.register_buffer("charge_meson_class_valid", valid)

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros(())
        if weights is None:
            return values.mean()
        return (values * weights).sum() / (weights.sum() + 1e-8)

    def _soft_objectness_gate(
        self, pred_logits: torch.Tensor, threshold: torch.Tensor | float
    ) -> torch.Tensor:
        """
        Turn each objectness score into a smooth approximation of a threshold cut.

        A query at the threshold receives weight 0.5. Scores above and below it
        are pushed towards one and zero, respectively. The temperature controls
        how closely this follows a hard cut while keeping the result differentiable.
        """
        probabilities = F.softmax(pred_logits.float(), dim=-1)[
            ..., self.object_class_index
        ]
        eps = torch.finfo(probabilities.dtype).eps
        threshold_tensor = torch.as_tensor(
            threshold, dtype=probabilities.dtype, device=probabilities.device
        ).clamp(eps, 1.0 - eps)
        return torch.sigmoid(
            (
                torch.logit(probabilities.clamp(eps, 1.0 - eps))
                - torch.logit(threshold_tensor)
            )
            / self.parent_objectness_temperature
        )

    def _compute_parent_kinematics_loss(
        self,
        pred_parent_p4: torch.Tensor,
        target_parent_p4: dict[str, torch.Tensor],
        parent_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare a reconstructed parent four-momentum with the true tau.

        The daughter momenta have already been combined before entering this
        helper. Their total is expressed in the same relative kinematic form as
        the daughter loss, then compared with the parent tau target.
        """
        device = pred_parent_p4.device
        pred_px, pred_py, pred_pz, pred_energy = pred_parent_p4.unbind(dim=-1)
        pred_pt = torch.sqrt((pred_px**2 + pred_py**2).clamp_min(1e-12))
        pred_eta = torch.asinh(pred_pz / pred_pt.clamp_min(1e-6))
        pred_phi = torch.atan2(pred_py, pred_px)
        pred_mass = torch.sqrt(
            torch.clamp(
                pred_energy**2 - pred_px**2 - pred_py**2 - pred_pz**2,
                min=1e-12,
            )
        )

        true_pt = target_parent_p4["pt"].to(dtype=pred_pt.dtype, device=device)
        true_eta = target_parent_p4["eta"].to(dtype=pred_eta.dtype, device=device)
        true_phi = target_parent_p4["phi"].to(dtype=pred_phi.dtype, device=device)
        true_energy = target_parent_p4["energy"].to(
            dtype=pred_energy.dtype, device=device
        )
        true_mass = torch.sqrt(
            torch.clamp(
                true_energy**2 - (true_pt * torch.cosh(true_eta)) ** 2,
                min=1e-12,
            )
        )
        delta_phi = pred_phi - true_phi
        pred_parent_kinematics = torch.stack(
            [
                torch.log(
                    pred_pt.clamp_min(1e-6) / true_pt.clamp_min(1e-6)
                ).clamp(-5.0, 5.0),
                pred_eta - true_eta,
                torch.sin(delta_phi),
                torch.cos(delta_phi),
                torch.log(
                    pred_mass.clamp_min(1e-6) / true_mass.clamp_min(1e-6)
                ).clamp(-5.0, 5.0),
            ],
            dim=-1,
        )
        target_parent_kinematics = torch.zeros_like(pred_parent_kinematics)
        target_parent_kinematics[:, 3] = 1.0
        loss, _ = self.tau_loss.compute_kinematics_loss(
            pred_parent_kinematics,
            target_parent_kinematics,
            parent_weights,
        )
        return loss

    @staticmethod
    def _decode_predicted_p4(
        pred_kinematics: torch.Tensor,
        kinematics_reference_p4: dict[str, torch.Tensor],
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert predicted daughter kinematics into physical four-momenta.

        Predictions are stored relative to each reconstructed jet. This helper
        restores their absolute momenta so selected or softly weighted daughters
        can be summed to reconstruct the parent tau.
        """
        references = {}
        for name in ("pt", "eta", "phi", "energy"):
            reference = kinematics_reference_p4[name].to(
                dtype=pred_kinematics.dtype, device=pred_kinematics.device
            )
            if batch_indices is not None:
                reference = reference[batch_indices]
            references[name] = reference
        return decode_kinematics(
            pred_kinematics,
            references["pt"],
            references["eta"],
            references["phi"],
            references["energy"],
            clamp_log_ratios=True,
        )

    def _compute_parent_charge_loss(
        self,
        charge_probabilities: torch.Tensor,
        target_parent_charge: torch.Tensor,
        signal_mask: torch.Tensor,
        parent_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare the possible total daughter charges with the true tau charge.

        Each query supplies probabilities for negative, neutral, and positive
        charge. These are combined into a probability distribution for the sum
        over all queries, which is then supervised by the parent tau charge.
        """
        batch_size, num_queries, _ = charge_probabilities.shape
        parent_charge_probabilities = charge_probabilities.new_zeros(
            (batch_size, 2 * num_queries + 1)
        )
        parent_charge_probabilities[:, num_queries] = 1.0
        for query_index in range(num_queries):
            probability = charge_probabilities[:, query_index]
            parent_charge_probabilities = (
                F.pad(parent_charge_probabilities[:, 1:], (0, 1))
                * probability[:, 0, None]
                + parent_charge_probabilities * probability[:, 1, None]
                + F.pad(parent_charge_probabilities[:, :-1], (1, 0))
                * probability[:, 2, None]
            )

        charge_loss = F.cross_entropy(
            parent_charge_probabilities.clamp_min(1e-8).log(),
            (
                target_parent_charge.to(
                    device=charge_probabilities.device, dtype=torch.long
                )
                + num_queries
            ).masked_fill(~signal_mask, self.ignore_index),
            reduction="none",
            ignore_index=self.ignore_index,
        )
        return self._weighted_mean(charge_loss, parent_weights)

    def forward(
        self,
        outputs: dict,
        target_kinematics: torch.Tensor,
        target_charge_cls: torch.Tensor,
        target_meson_class: torch.Tensor,
        target_mask: torch.Tensor,
        target_parent_charge: torch.Tensor,
        target_parent_decay_mode: torch.Tensor,
        target_parent_p4: dict[str, torch.Tensor],
        kinematics_reference_p4: dict[str, torch.Tensor],
        target_is_tau: torch.Tensor | None = None,
        jet_weights: torch.Tensor | None = None,
        objectness_threshold: torch.Tensor | float = 0.5,
    ) -> dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_kinematics = outputs["pred_kinematics"]
        pred_charge_logits = outputs["pred_charge_logits"]
        pred_meson_class_logits = outputs["pred_meson_class_logits"]

        batch_size, num_queries, _ = pred_logits.shape
        device = pred_logits.device

        # Daughter-level losses are signal-only: background jets only supervise
        # the tau-tagging head (loss_tau_id). If no tau label is provided we treat
        # every jet as signal to preserve the previous behaviour.
        if target_is_tau is not None:
            signal_mask = target_is_tau.bool()
        else:
            signal_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Restricting the matcher to signal jets means background jets are never
        # even handed to the assignment solver, which removes most of the
        # matching work whenever background dominates the jet count. (The
        # actual mix is set by dataset.max_jets_per_sample and, for the tauID
        # loss, by cls_weight; do not assume a ratio here.)
        match_mask = target_mask & signal_mask.unsqueeze(1)
        pair_b, pair_q, pair_t = self.matcher(
            pred_logits=pred_logits,
            pred_kinematics=pred_kinematics,
            pred_charge_logits=pred_charge_logits,
            pred_meson_class_logits=pred_meson_class_logits,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_meson_class=target_meson_class,
            target_mask=match_mask,
        )
        num_matched = pair_b.numel()

        tgt_classes = torch.full(
            (batch_size, num_queries),
            self.no_object_class_index,
            dtype=torch.long,
            device=device,
        )
        tgt_classes[pair_b, pair_q] = self.object_class_index

        if num_matched > 0:
            # Unweighted on purpose (class docstring): cls_weight is the
            # tagger's decorrelation weight, not a reconstruction weight.
            pair_w = pred_logits.new_ones(num_matched)

            kin_pred_cat = pred_kinematics[pair_b, pair_q]
            kin_tgt_cat = target_kinematics[pair_b, pair_t]

            tgt_charge_sel = target_charge_cls[pair_b, pair_t]
            valid_charge = tgt_charge_sel != self.ignore_index
            # cross_entropy zeroes the ignored entries, so folding the validity
            # flag into the weights reproduces the mean over valid pairs only.
            ce_charge = F.cross_entropy(
                pred_charge_logits[pair_b, pair_q],
                tgt_charge_sel,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            charge_w = pair_w * valid_charge.to(pair_w.dtype)

            target_meson_class_sel = target_meson_class[pair_b, pair_t]
            valid_meson_class = target_meson_class_sel != self.ignore_index
            ce_meson_class = F.cross_entropy(
                pred_meson_class_logits[pair_b, pair_q],
                target_meson_class_sel,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            meson_class_w = pair_w * valid_meson_class.to(pair_w.dtype)

        # objectness over all queries
        class_weight = pred_logits.new_tensor([1.0, self.eos_coef])
        ce_per_query = F.cross_entropy(
            pred_logits.transpose(1, 2),
            tgt_classes,
            weight=class_weight,
            reduction="none",
        )
        # Signal jets only, unweighted (class docstring).
        sig_w = signal_mask.to(dtype=ce_per_query.dtype)
        loss_objectness = (ce_per_query * sig_w[:, None]).sum() / (
            sig_w.sum() * num_queries + 1e-8
        )

        # jet-level tau-tagging loss: the ONLY term that sees cls_weight, and the
        # only one background jets contribute to.
        if "is_tau" in outputs and target_is_tau is not None:
            if jet_weights is not None:
                tau_w = jet_weights.to(dtype=pred_logits.dtype, device=device)
            else:
                tau_w = pred_logits.new_ones(batch_size)
            loss_tau_id = self.tau_loss.compute_tagging_loss(
                outputs["is_tau"], target_is_tau, tau_w
            )
        else:
            loss_tau_id = pred_logits.new_zeros(())

        # matched losses
        if num_matched > 0:
            loss_kinematics, kin_components = self.tau_loss.compute_kinematics_loss(
                kin_pred_cat,
                kin_tgt_cat,
                pair_w,
            )
            loss_charge = self._weighted_mean(ce_charge, charge_w)
            loss_meson_class = self._weighted_mean(
                ce_meson_class, meson_class_w
            )
            num_charge_supervised = valid_charge.sum()
            num_meson_class_supervised = valid_meson_class.sum()
        else:
            loss_kinematics = pred_logits.new_zeros(())
            kin_components = {
                "log_pt": pred_logits.new_zeros(()),
                "delta_eta": pred_logits.new_zeros(()),
                "phi_chord": pred_logits.new_zeros(()),
                "log_mass": pred_logits.new_zeros(()),
            }
            loss_charge = pred_logits.new_zeros(())
            loss_meson_class = pred_logits.new_zeros(())
            num_charge_supervised = pred_logits.new_zeros(())
            num_meson_class_supervised = pred_logits.new_zeros(())

        total_loss = (
            self.loss_objectness_weight * loss_objectness
            + self.loss_tau_id_weight * loss_tau_id
            + self.loss_kinematics_weight * loss_kinematics
            + self.loss_charge_weight * loss_charge
            + self.loss_meson_class_weight * loss_meson_class
        )

        # ---- Auxiliary penalties ----
        loss_consistency = pred_logits.new_zeros(())
        loss_charge_count = pred_logits.new_zeros(())
        loss_parent_kinematics = pred_logits.new_zeros(())
        loss_parent_charge = pred_logits.new_zeros(())
        loss_parent_decay_mode = pred_logits.new_zeros(())
        loss_soft_parent_kinematics = pred_logits.new_zeros(())
        loss_soft_parent_charge = pred_logits.new_zeros(())
        loss_soft_parent_decay_mode = pred_logits.new_zeros(())

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_consistency_weight > 0
        ):
            p_charge = F.softmax(pred_charge_logits, dim=-1)  # [B, Q, 3]
            p_meson_class = F.softmax(pred_meson_class_logits, dim=-1)
            p_joint = p_charge[..., :, None] * p_meson_class[..., None, :]
            invalid_prob = (
                p_joint * (1 - self.charge_meson_class_valid)
            ).sum(dim=(-2, -1))
            sig_w = signal_mask.to(dtype=invalid_prob.dtype)
            loss_consistency = (invalid_prob * sig_w[:, None]).sum() / (
                sig_w.sum() * num_queries + 1e-8
            )
        if self.loss_consistency_weight > 0:
            total_loss = total_loss + self.loss_consistency_weight * loss_consistency

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_charge_count_weight > 0
        ):
            p_object = F.softmax(pred_logits, dim=-1)[..., 0]  # [B, Q]
            # Probability of being charged, NOT argmax(charge) != neutral.
            #
            # The argmax made this term a step function of the charge logits:
            # every query crossing its decision boundary moved expected_charged
            # by a whole unit, so the penalty jumped discontinuously, and the
            # jumps get more frequent late in training when many queries sit
            # near a boundary. It also meant the charge head received NO
            # gradient from this term at all -- only p_object did -- so the term
            # could push objectness around without ever correcting the charge
            # prediction that caused it. Summing the charged probability instead
            # is continuous in both heads and differentiable through both.
            charge_probs = F.softmax(pred_charge_logits, dim=-1)  # [B, Q, 3]
            is_charged_pred = charge_probs[..., 0] + charge_probs[..., 2]
            expected_charged = (p_object * is_charged_pred).sum(dim=-1)  # [B]
            # Charged daughters are class 0 (-1) or class 2 (+1); ignore padded slots.
            is_charged_true = (
                (target_charge_cls == 0) | (target_charge_cls == 2)
            ).float()
            n_charged_true = is_charged_true.sum(dim=-1)  # [B]
            excess = F.relu(expected_charged - n_charged_true)
            sig_w = signal_mask.to(dtype=excess.dtype)
            loss_charge_count = (excess * sig_w).sum() / (sig_w.sum() + 1e-8)
        if self.loss_charge_count_weight > 0:
            total_loss = total_loss + self.loss_charge_count_weight * loss_charge_count

        # Parent constraints ask the predicted daughters to reconstruct the
        # known parent tau. Kinematics compares the summed daughter momentum to
        # the tau momentum, charge compares the summed daughter charge to the
        # tau charge, and decay mode compares the charged/neutral daughter
        # counts to the tau decay mode. They are signal-only and unweighted,
        # like the daughter losses (class docstring).
        parent_weights = signal_mask.to(dtype=pred_logits.dtype)

        # Matched constraints use only queries assigned to true daughters by
        # the Hungarian matcher. Other queries do not contribute to the parent.
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_parent_kinematics_weight > 0
        ):
            pred_p4 = self._decode_predicted_p4(
                pred_kinematics[pair_b, pair_q],
                kinematics_reference_p4,
                pair_b,
            )
            pred_parent_p4 = pred_p4.new_zeros((batch_size, 4)).index_add(0, pair_b, pred_p4)
            loss_parent_kinematics = self._compute_parent_kinematics_loss(
                pred_parent_p4,
                target_parent_p4,
                parent_weights,
            )
        if self.loss_parent_kinematics_weight > 0:
            total_loss = total_loss + self.loss_parent_kinematics_weight * loss_parent_kinematics

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_parent_charge_weight > 0
        ):
            matched_query_mask = torch.zeros(
                (batch_size, num_queries), dtype=torch.bool, device=device
            )
            matched_query_mask[pair_b, pair_q] = True
            charge_probabilities = F.softmax(pred_charge_logits.float(), dim=-1)
            unmatched_charge = charge_probabilities.new_tensor([0.0, 1.0, 0.0])
            charge_probabilities = torch.where(
                matched_query_mask.unsqueeze(-1),
                charge_probabilities,
                unmatched_charge,
            )
            loss_parent_charge = self._compute_parent_charge_loss(
                charge_probabilities,
                target_parent_charge,
                signal_mask,
                parent_weights,
            )
        if self.loss_parent_charge_weight > 0:
            total_loss = total_loss + self.loss_parent_charge_weight * loss_parent_charge

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_parent_decay_mode_weight > 0
        ):
            matched_query_mask = torch.zeros(
                (batch_size, num_queries), dtype=torch.bool, device=device
            )
            matched_query_mask[pair_b, pair_q] = True
            neutral_probability = F.softmax(pred_charge_logits.float(), dim=-1)[..., 1]
            neutral_probability = neutral_probability * matched_query_mask

            neutral_count_probabilities = neutral_probability.new_zeros(
                (batch_size, num_queries + 1)
            )
            neutral_count_probabilities[:, 0] = 1.0
            for query_index in range(num_queries):
                probability = neutral_probability[:, query_index, None]
                neutral_count_probabilities = (
                    neutral_count_probabilities * (1 - probability)
                    + F.pad(neutral_count_probabilities[:, :-1], (1, 0)) * probability
                )

            num_constituents = matched_query_mask.sum(dim=-1, keepdim=True)
            num_neutral = torch.arange(num_queries + 1, device=device).unsqueeze(0)
            num_charged = num_constituents - num_neutral
            decay_mode_stride = 5
            decay_modes = decay_mode_stride * (num_charged - 1) + num_neutral
            invalid_decay_mode_index = decay_mode_stride * num_queries + 1
            decay_mode_indices = torch.where(
                num_charged > 0,
                decay_modes + decay_mode_stride,
                invalid_decay_mode_index,
            )
            decay_mode_probabilities = neutral_count_probabilities.new_zeros(
                (batch_size, 5 * num_queries + 2)
            ).scatter_add(
                1,
                decay_mode_indices,
                neutral_count_probabilities,
            )
            decay_mode_loss = F.cross_entropy(
                decay_mode_probabilities.clamp_min(1e-8).log(),
                (target_parent_decay_mode.to(device=device, dtype=torch.long) + decay_mode_stride).masked_fill(
                    ~signal_mask, self.ignore_index
                ),
                reduction="none",
                ignore_index=self.ignore_index,
            )
            loss_parent_decay_mode = self._weighted_mean(decay_mode_loss, parent_weights)
        if self.loss_parent_decay_mode_weight > 0:
            total_loss = total_loss + self.loss_parent_decay_mode_weight * loss_parent_decay_mode

        # Soft constraints use every query, weighted by a smooth version of the
        # objectness threshold. Queries well above the threshold contribute
        # almost fully, queries well below it contribute almost nothing, and
        # queries near it change smoothly so gradients can pass through.
        use_soft_parent_constraints = any(
            weight > 0
            for weight in (
                self.loss_soft_parent_kinematics_weight,
                self.loss_soft_parent_charge_weight,
                self.loss_soft_parent_decay_mode_weight,
            )
        )
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and use_soft_parent_constraints
        ):
            soft_query_weights = self._soft_objectness_gate(
                pred_logits, objectness_threshold
            )

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_soft_parent_kinematics_weight > 0
        ):
            pred_p4 = self._decode_predicted_p4(
                pred_kinematics,
                kinematics_reference_p4,
            )
            pred_parent_p4 = (pred_p4 * soft_query_weights.unsqueeze(-1)).sum(dim=1)
            loss_soft_parent_kinematics = self._compute_parent_kinematics_loss(
                pred_parent_p4,
                target_parent_p4,
                parent_weights,
            )
        if self.loss_soft_parent_kinematics_weight > 0:
            total_loss = (
                total_loss
                + self.loss_soft_parent_kinematics_weight
                * loss_soft_parent_kinematics
            )

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_soft_parent_charge_weight > 0
        ):
            charge_probabilities = F.softmax(pred_charge_logits.float(), dim=-1)
            absent_charge = charge_probabilities.new_tensor([0.0, 1.0, 0.0])
            charge_probabilities = (
                charge_probabilities * soft_query_weights.unsqueeze(-1)
                + absent_charge * (torch.ones_like(soft_query_weights) - soft_query_weights).unsqueeze(-1)
            )
            loss_soft_parent_charge = self._compute_parent_charge_loss(
                charge_probabilities,
                target_parent_charge,
                signal_mask,
                parent_weights,
            )
        if self.loss_soft_parent_charge_weight > 0:
            total_loss = (
                total_loss
                + self.loss_soft_parent_charge_weight * loss_soft_parent_charge
            )

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.loss_soft_parent_decay_mode_weight > 0
        ):
            charge_probabilities = F.softmax(pred_charge_logits.float(), dim=-1)
            neutral_probability = (
                soft_query_weights * charge_probabilities[..., 1]
            )
            charged_probability = soft_query_weights * (
                charge_probabilities[..., 0] + charge_probabilities[..., 2]
            )
            absent_probability = torch.ones_like(soft_query_weights) - soft_query_weights

            count_probabilities = charge_probabilities.new_zeros(
                (batch_size, num_queries + 1, num_queries + 1)
            )
            count_probabilities[:, 0, 0] = 1.0
            for query_index in range(num_queries):
                count_probabilities = (
                    count_probabilities
                    * absent_probability[:, query_index, None, None]
                    + F.pad(count_probabilities[:, :-1, :], (0, 0, 1, 0))
                    * charged_probability[:, query_index, None, None]
                    + F.pad(count_probabilities[:, :, :-1], (1, 0, 0, 0))
                    * neutral_probability[:, query_index, None, None]
                )

            num_charged = torch.arange(num_queries + 1, device=device).view(-1, 1)
            num_neutral = torch.arange(num_queries + 1, device=device).view(1, -1)
            decay_mode_stride = 5
            decay_modes = decay_mode_stride * (num_charged - 1) + num_neutral
            invalid_decay_mode_index = decay_mode_stride * num_queries + 1
            decay_mode_indices = torch.where(
                (num_charged > 0)
                & (num_charged + num_neutral <= num_queries),
                decay_modes + decay_mode_stride,
                invalid_decay_mode_index,
            ).flatten()
            decay_mode_probabilities = count_probabilities.new_zeros(
                (batch_size, 5 * num_queries + 2)
            ).scatter_add(
                1,
                decay_mode_indices.unsqueeze(0).expand(batch_size, -1),
                count_probabilities.flatten(1),
            )
            decay_mode_loss = F.cross_entropy(
                decay_mode_probabilities.clamp_min(1e-8).log(),
                (
                    target_parent_decay_mode.to(device=device, dtype=torch.long)
                    + decay_mode_stride
                ).masked_fill(~signal_mask, self.ignore_index),
                reduction="none",
                ignore_index=self.ignore_index,
            )
            loss_soft_parent_decay_mode = self._weighted_mean(
                decay_mode_loss, parent_weights
            )
        if self.loss_soft_parent_decay_mode_weight > 0:
            total_loss = (
                total_loss
                + self.loss_soft_parent_decay_mode_weight
                * loss_soft_parent_decay_mode
            )

        raw_losses = {
            "objectness": loss_objectness,
            "tau_id": loss_tau_id,
            "kinematics": loss_kinematics,
            "charge": loss_charge,
            "meson_class": loss_meson_class,
            "consistency": loss_consistency,
            "charge_count": loss_charge_count,
            "parent_kinematics": loss_parent_kinematics,
            "parent_charge": loss_parent_charge,
            "parent_decay_mode": loss_parent_decay_mode,
            "soft_parent_kinematics": loss_soft_parent_kinematics,
            "soft_parent_charge": loss_soft_parent_charge,
            "soft_parent_decay_mode": loss_soft_parent_decay_mode,
        }
        if self.loss_weighting is not None:
            signal_count = signal_mask.sum()
            supervision_counts = {
                "objectness": signal_count,
                "tau_id": (
                    batch_size
                    if "is_tau" in outputs and target_is_tau is not None
                    else 0
                ),
                "kinematics": num_matched,
                "charge": num_charge_supervised,
                "meson_class": num_meson_class_supervised,
                "consistency": signal_count,
                "charge_count": signal_count,
                "parent_kinematics": signal_count,
                "parent_charge": signal_count,
                "parent_decay_mode": signal_count,
                "soft_parent_kinematics": signal_count,
                "soft_parent_charge": signal_count,
                "soft_parent_decay_mode": signal_count,
            }
            total_loss = self.loss_weighting(raw_losses, supervision_counts)
            effective_weights = self.loss_weighting.effective_weights()
        else:
            effective_weights = {
                name: raw_losses[name].new_tensor(weight)
                for name, weight in self.configured_loss_weights.items()
                if weight > 0
            }

        weighted_losses = {
            f"weighted_loss/{name}": effective_weights[name] * raw_losses[name]
            for name in effective_weights
        }

        return {
            "loss": total_loss,
            "loss_objectness": loss_objectness,
            "loss_tau_id": loss_tau_id,
            "loss_kinematics": loss_kinematics,
            "kinematics_log_pt_loss": kin_components["log_pt"],
            "kinematics_delta_eta_loss": kin_components["delta_eta"],
            "kinematics_phi_chord_loss": kin_components["phi_chord"],
            "kinematics_log_mass_loss": kin_components["log_mass"],
            "loss_charge": loss_charge,
            "loss_meson_class": loss_meson_class,
            "loss_consistency": loss_consistency,
            "loss_charge_count": loss_charge_count,
            "loss_parent_kinematics": loss_parent_kinematics,
            "loss_parent_charge": loss_parent_charge,
            "loss_parent_decay_mode": loss_parent_decay_mode,
            "loss_soft_parent_kinematics": loss_soft_parent_kinematics,
            "loss_soft_parent_charge": loss_soft_parent_charge,
            "loss_soft_parent_decay_mode": loss_soft_parent_decay_mode,
            "num_matched": pred_logits.new_tensor(float(num_matched)),
            "num_charge_supervised": num_charge_supervised.to(pred_logits.dtype),
            "num_meson_class_supervised": num_meson_class_supervised.to(
                pred_logits.dtype
            ),
            **weighted_losses,
            **{k: pred_logits.new_tensor(v) for k, v in self.matcher.last_cost_terms.items()},
        }


# Config subtrees ParTauDETRModule.__init__ reads. `output_dir` is included
# because training.input_scaling.scaler_path interpolates it.
_HPARAM_CFG_KEYS = ("model", "dataset", "training", "output_dir")


class ParTauDETRModule(L.LightningModule):
    """
    Lightning module for ParTauDETR with Hungarian matching and mixed losses.

    Expected target keys from dataloader:
      - particles_kinematics: [B, T, K]
      - particles_charge_ohe: [B, T, 3]
      - particles_meson_class_ohe: [B, T, C]
      - particles_mask: [B, T]
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: fully composed Hydra config. Every architecture choice is read
                from it -- nothing about the network is hardcoded here, so a
                checkpoint's architecture is fully described by its config.
        """
        super().__init__()
        self.cfg = cfg
        self.ignore_index = -100
        # Persist cfg into the checkpoint so `load_from_checkpoint(path)` rebuilds
        # the exact architecture without the caller having to supply a config.
        # Only the subtrees __init__ reads are kept: loggers flatten hparams and
        # resolve every interpolation, so carrying unrelated config (e.g. the
        # metrics plotting tree) turns any unresolvable key elsewhere into a
        # crash at fit() time, and floods the hyperparameter table.
        self.save_hyperparameters(
            {
                "cfg": OmegaConf.masked_copy(
                    cfg, [k for k in _HPARAM_CFG_KEYS if k in cfg]
                )
            }
        )

        arch = cfg.model
        encoder_cfg = arch.encoder
        detr_cfg = arch.detr

        num_charge_classes = int(arch.num_charge_classes)
        if num_charge_classes != 3:
            raise ValueError("This module expects 3 charge classes for {-1, 0, +1}.")

        meson_classes = get_meson_classes(cfg.dataset.tau_daughter_pdg_ids)
        self.num_meson_classes = len(meson_classes)

        # The matcher returns min(num_queries, n_targets) pairs per jet, so
        # with fewer queries than target slots the surplus targets are never
        # supervised, silently. The config ties the two by interpolation; this
        # catches a command-line override of one of them alone.
        num_queries = int(arch.num_queries)
        max_tau_daughters = int(cfg.dataset.max_tau_daughters)
        if num_queries != max_tau_daughters:
            raise ValueError(
                f"model.num_queries ({num_queries}) must equal "
                f"dataset.max_tau_daughters ({max_tau_daughters}). Override "
                "dataset.max_tau_daughters and let num_queries follow it."
            )

        self.tau_loss = TauLoss.from_config(arch.tau_loss, owner="ParTauDETR")
        self.num_kinematics_components = int(arch.num_kinematics_components)

        embed_dims = [int(d) for d in encoder_cfg.embed_dims]
        num_heads = int(encoder_cfg.num_heads)
        decoder_num_heads = detr_cfg.get("decoder_num_heads", None)
        decoder_num_heads = (
            num_heads if decoder_num_heads is None else int(decoder_num_heads)
        )
        embed_dim = embed_dims[-1]
        for label, heads in (("encoder", num_heads), ("decoder", decoder_num_heads)):
            if embed_dim % heads != 0:
                raise ValueError(
                    f"{label} num_heads ({heads}) must divide the model dimension "
                    f"embed_dims[-1] ({embed_dim})."
                )

        self.ParTauDETR = ParTauDETR(
            input_dim=int(cfg.dataset.num_features),
            num_queries=int(arch.num_queries),
            num_charge_classes=num_charge_classes,
            num_meson_classes=self.num_meson_classes,
            num_kinematics_components=self.num_kinematics_components,
            # encoder
            num_layers=int(encoder_cfg.num_layers),
            num_heads=num_heads,
            num_cls_layers=int(encoder_cfg.num_cls_layers),
            embed_dims=embed_dims,
            pair_embed_dims=[int(d) for d in encoder_cfg.pair_embed_dims],
            pair_input_dim=int(encoder_cfg.pair_input_dim),
            use_pre_activation_pair=bool(encoder_cfg.use_pre_activation_pair),
            remove_self_pair=bool(encoder_cfg.remove_self_pair),
            activation=str(encoder_cfg.activation),
            metric=str(encoder_cfg.metric),
            # DETR decoder and heads
            decoder_num_layers=int(detr_cfg.decoder_num_layers),
            decoder_num_heads=decoder_num_heads,
            decoder_ffn_ratio=int(detr_cfg.decoder_ffn_ratio),
            decoder_dropout=float(detr_cfg.decoder_dropout),
            append_global_token=bool(detr_cfg.append_global_token),
            tau_id_head=bool(detr_cfg.tau_id_head),
            head_dropout=float(detr_cfg.head_dropout),
            for_inference=False,
            use_amp=False,
        )

        self.matcher = HungarianMatcher(
            cost_objectness=float(detr_cfg.matcher.cost_objectness),
            cost_kinematics_l1=float(detr_cfg.matcher.cost_kinematics_l1),
            cost_charge_ce=float(detr_cfg.matcher.cost_charge),
            cost_meson_class_ce=float(detr_cfg.matcher.cost_meson_class_ce),
            object_class_index=0,
            ignore_index=self.ignore_index,
            kinematics_component_weights=list(
                detr_cfg.matcher.kinematics_component_weights
            ),
            classification_cost=str(
                detr_cfg.matcher.get("classification_cost", "clipped_nll")
            ),
            classification_cost_clip=float(
                detr_cfg.matcher.get("classification_cost_clip", 5.0)
            ),
        )

        self.criterion = SetCriterion(
            matcher=self.matcher,
            tau_loss=self.tau_loss,
            meson_classes=meson_classes,
            # Read strictly: a `.get(key, default)` here would silently fall back
            # to a hidden default if the key were renamed or misspelled.
            loss_objectness_weight=float(detr_cfg.loss.weight_objectness),
            loss_tau_id_weight=float(detr_cfg.loss.weight_tau_id),
            loss_kinematics_weight=float(detr_cfg.loss.weight_kinematics),
            loss_charge_weight=float(detr_cfg.loss.weight_charge),
            loss_meson_class_weight=float(detr_cfg.loss.weight_meson_class),
            loss_consistency_weight=float(detr_cfg.loss.weight_consistency),
            loss_charge_count_weight=float(detr_cfg.loss.weight_charge_count),
            loss_parent_kinematics_weight=float(detr_cfg.loss.weight_parent_kinematics),
            loss_parent_charge_weight=float(detr_cfg.loss.weight_parent_charge),
            loss_parent_decay_mode_weight=float(detr_cfg.loss.weight_parent_decay_mode),
            loss_soft_parent_kinematics_weight=float(detr_cfg.loss.weight_soft_parent_kinematics),
            loss_soft_parent_charge_weight=float(detr_cfg.loss.weight_soft_parent_charge),
            loss_soft_parent_decay_mode_weight=float(detr_cfg.loss.weight_soft_parent_decay_mode),
            parent_objectness_temperature=float(detr_cfg.loss.parent_objectness_temperature),
            automatic_weight_optimization=bool(
                detr_cfg.loss.get("automatic_weight_optimization", False)
            ),
            automatic_weight_priorities={
                str(name): float(priority)
                for name, priority in detr_cfg.loss.get(
                    "automatic_weight_priorities", {}
                ).items()
            },
            no_object_class_index=1,
            object_class_index=0,
            eos_coef=float(detr_cfg.loss.eos_coef),
            ignore_index=self.ignore_index,
        )

        # Starting value and fallback for the objectness threshold; the
        # calibrated value lives in the `score_threshold_calibrated` buffer.
        self.score_threshold = float(detr_cfg.inference.score_threshold)
        # p(tau) above which a jet's predicted daughters are kept at prediction
        # time. Objectness is trained on signal jets only, so the tauID head is
        # the only thing that can say "this jet has no daughters at all".
        self.tau_id_threshold = float(detr_cfg.inference.get("tau_id_threshold", 0.5))

        # One representative |PDG| per meson class, for the jet-level
        # evaluation: the decay mode is derived by ml-tau-data from PDG ids
        # classified by property, so a class is handed over as the hadron that
        # represents it -- 211 for a charged class, 130 for a neutral one, the
        # same convention the ntupelizer uses when it writes the daughter ids.
        # Not persisted: it is derived from the config, not learned.
        self.register_buffer(
            "meson_class_repr_pdg",
            torch.tensor(
                [130 if set(c.charges) == {0} else 211 for c in meson_classes],
                dtype=torch.long,
            ),
            persistent=False,
        )

        # Every val metric at the epoch with the lowest val_losses/loss. The
        # checkpoint callback records the best SCORE, but nothing records the
        # per-head breakdown that produced it: trainer.callback_metrics at the
        # end of the run is the LAST validation pass, which is a different epoch
        # whenever the run stopped improving early. Snapshotted here so a
        # scaling study can read the components that belong to the best loss.
        self.best_val_metrics: dict[str, float] | None = None

        # Jet-level validation performance: the losses say how well the
        # criterion is minimised, these say whether the reconstructed tau is
        # right. Capped, because a validation split can be millions of jets and
        # the distributions converge long before that.
        self.val_jets = s2s.SetToSetAccumulator(
            max_jets=int(cfg.training.get("jet_level_eval_jets", 200_000))
        )

        # The objectness threshold is calibrated, not assumed. inference
        # .score_threshold is only the starting value and the fallback: the
        # right operating point moves as the objectness head sharpens, so it is
        # rescanned before every validation on recent TRAIN jets. Using train
        # data for this matters -- picking the operating point on the same split
        # the model is then scored on would flatter every number that follows.
        scan_cfg = cfg.training.get("threshold_scan", None)
        self.threshold_scan_enabled = (
            True if scan_cfg is None else bool(scan_cfg.get("enabled", True))
        )
        self.threshold_scan_objective = (
            str(scan_cfg.get("objective", "decay_mode")) if scan_cfg else "decay_mode"
        )
        self.threshold_buffer_every = max(
            1, int(scan_cfg.get("buffer_every_n_steps", 20)) if scan_cfg else 20
        )
        self.threshold_buffer = s2s.ThresholdCalibrationBuffer(
            max_jets=int(scan_cfg.get("jets", 100_000)) if scan_cfg else 100_000
        )
        self.validation_threshold_buffer = s2s.ThresholdCalibrationBuffer(
            max_jets=int(scan_cfg.get("jets", 100_000)) if scan_cfg else 100_000
        )
        self.threshold_grid = (
            np.linspace(
                float(scan_cfg.get("low", 0.5)),
                float(scan_cfg.get("high", 0.9)),
                int(scan_cfg.get("points", 9)),
            )
            if scan_cfg
            else np.linspace(0.3, 0.9, 13)
        )
        # The calibrated objectness threshold. A registered buffer, not a
        # plain attribute, so it travels with the checkpoint: validation,
        # predict_step and anything that loads the model then use the SAME
        # operating point. Before the first scan it is the config value.
        self.register_buffer(
            "score_threshold_calibrated",
            torch.tensor(float(detr_cfg.inference.score_threshold)),
        )
        # Consecutive non-finite training losses seen; see training_step.
        self._non_finite_steps = 0

        # Gradient-outlier guard and the clip value it is logged against; see
        # on_before_optimizer_step. Read here rather than from the Trainer so a
        # module built outside a Trainer still has the documented defaults.
        _opt_cfg = cfg.training.get("optimizer", None) or {}
        _trainer_cfg = cfg.training.get("trainer", None) or {}
        self.grad_skip_norm = float(_opt_cfg.get("grad_skip_norm", 50.0))
        self.grad_skip_patience = int(_opt_cfg.get("grad_skip_patience", 50))
        self.grad_skip_warmup_steps = int(
            _opt_cfg.get("grad_skip_warmup_steps", 500)
        )
        self.grad_clip_val = float(_trainer_cfg.get("gradient_clip_val", 1.0))
        self._consecutive_skips = 0
        # One warning per run for the start-up transient, not one per step.
        self._warned_warmup_grad = False

        # Best decay-mode accuracy any threshold scan has reached, so a later
        # scan that collapses can be recognised as such; see on_validation_start.
        self._best_scan_accuracy = 0.0

    def _log_automatic_loss_weights(self) -> None:
        if self.criterion.loss_weighting is None:
            return
        effective_weights = self.criterion.loss_weighting.effective_weights()
        for name, log_variance in self.criterion.loss_weighting.log_variances.items():
            self.log(
                f"loss_log_variances/{name}",
                log_variance.detach(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"loss_weights/{name}",
                effective_weights[name].detach(),
                on_step=False,
                on_epoch=True,
            )

    def _log_per_loss_gradient_norms(
        self, losses: dict[str, torch.Tensor]
    ) -> None:
        parameters = tuple(
            parameter
            for parameter in self.ParTauDETR.parameters()
            if parameter.requires_grad
        )
        for key, weighted_loss in losses.items():
            if not key.startswith("weighted_loss/"):
                continue
            if weighted_loss.requires_grad:
                gradients = torch.autograd.grad(
                    weighted_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                squared_norms = [
                    gradient.detach().float().square().sum()
                    for gradient in gradients
                    if gradient is not None
                ]
                norm = (
                    torch.stack(squared_norms).sum().sqrt()
                    if squared_norms
                    else weighted_loss.new_zeros(())
                )
            else:
                norm = weighted_loss.new_zeros(())
            name = key.removeprefix("weighted_loss/")
            self.log(
                f"grad/loss_norm/{name}",
                norm,
                on_step=True,
                on_epoch=False,
            )

    @staticmethod
    def _ohe_to_class_indices(
        one_hot: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Convert one-hot [B, T, C] to class indices [B, T].
        All-zero rows map to ignore_index.
        """
        cls = one_hot.argmax(dim=-1)
        has_label = one_hot.sum(dim=-1) > 0
        cls = cls.to(torch.long)
        cls = cls.masked_fill(~has_label, ignore_index)
        return cls

    def _extract_set_targets(
        self, targets: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = [
            "particles_kinematics",
            "particles_charge_ohe",
            "particles_meson_class_ohe",
            "particles_mask",
        ]
        missing = [k for k in required if k not in targets]
        if len(missing) > 0:
            raise KeyError(
                f"Missing required DETR target keys: {missing}. Available keys: {list(targets.keys())}"
            )

        target_kinematics = targets["particles_kinematics"].float()
        target_mask = targets["particles_mask"].bool()

        if (
            target_kinematics.ndim != 3
            or target_kinematics.size(-1) != self.num_kinematics_components
        ):
            raise ValueError(
                f"Expected particles_kinematics shape [B, T, {self.num_kinematics_components}], "
                f"got {tuple(target_kinematics.shape)}"
            )

        charge_ohe = targets["particles_charge_ohe"].float()
        meson_class_ohe = targets["particles_meson_class_ohe"].float()

        if charge_ohe.ndim != 3 or charge_ohe.shape[:2] != target_mask.shape:
            raise ValueError(
                f"Expected particles_charge_ohe shape [B, T, C], got {tuple(charge_ohe.shape)}"
            )
        if charge_ohe.size(-1) != 3:
            raise ValueError(
                f"Expected particles_charge_ohe last dim = 3, got {charge_ohe.size(-1)}"
            )

        if (
            meson_class_ohe.ndim != 3
            or meson_class_ohe.shape[:2] != target_mask.shape
        ):
            raise ValueError(
                "Expected particles_meson_class_ohe shape [B, T, C], got "
                f"{tuple(meson_class_ohe.shape)}"
            )
        if meson_class_ohe.size(-1) != self.num_meson_classes:
            raise ValueError(
                "Expected particles_meson_class_ohe last dim = "
                f"{self.num_meson_classes}, got {meson_class_ohe.size(-1)}"
            )

        target_charge_cls = self._ohe_to_class_indices(charge_ohe, self.ignore_index)
        target_meson_class = self._ohe_to_class_indices(
            meson_class_ohe, self.ignore_index
        )

        # Ignore padded slots in class losses/matching costs.
        target_charge_cls = target_charge_cls.masked_fill(
            ~target_mask, self.ignore_index
        )
        target_meson_class = target_meson_class.masked_fill(
            ~target_mask, self.ignore_index
        )

        # Jet-level tau-tagging label. Prefer an explicit `is_tau` target when the
        # dataloader provides it; otherwise fall back to "has at least one valid
        # daughter", which is the same signal/background distinction.
        if "is_tau" in targets:
            target_is_tau = targets["is_tau"].long()
        else:
            target_is_tau = target_mask.any(dim=1).long()
        if target_is_tau.ndim != 1 or target_is_tau.size(0) != target_mask.size(0):
            raise ValueError(
                f"Expected is_tau shape [{target_mask.size(0)}], got {tuple(target_is_tau.shape)}"
            )

        return (
            target_kinematics,
            target_charge_cls,
            target_meson_class,
            target_mask,
            target_is_tau,
            targets["gen_jet_tau_charge"].long(),
            targets["gen_jet_tau_decaymode"].long(),
        )

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        outputs = self.ParTauDETR(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        return (
            outputs,
            inputs.target,
            inputs.weight,
            inputs.gen_jet_tau_p4s,
            inputs.reco_jet_p4s,
        )

    def training_step(self, batch, _batch_idx):
        outputs, targets, weights, gen_jet_tau_p4, kinematics_reference_p4 = self.forward(batch)
        (
            target_kinematics,
            target_charge_cls,
            target_meson_class,
            target_mask,
            target_is_tau,
            target_parent_charge,
            target_parent_decay_mode,
        ) = self._extract_set_targets(targets)

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_meson_class=target_meson_class,
            target_mask=target_mask,
            target_parent_charge=target_parent_charge,
            target_parent_decay_mode=target_parent_decay_mode,
            target_parent_p4=gen_jet_tau_p4,
            kinematics_reference_p4=kinematics_reference_p4,
            target_is_tau=target_is_tau,
            jet_weights=weights,
            objectness_threshold=self.score_threshold_calibrated,
        )
        self._log_automatic_loss_weights()

        # A non-finite loss must not reach the optimizer: one backward of a NaN
        # writes NaN into every weight, and the run then continues producing
        # NaN until its wall clock (the 250-epoch run of 2026-09-16 spent 19 h
        # that way). Returning None skips this step; several in a row means
        # the model is gone, so stop the run and let the checkpoints stand.
        if not bool(torch.isfinite(losses["loss"])):
            self._non_finite_steps += 1
            self.log("train/non_finite_steps", float(self._non_finite_steps),
                     on_step=True, on_epoch=False)
            step = getattr(self.trainer, "global_step", "?") if self.trainer is not None else "?"
            warnings.warn(
                f"non-finite training loss at step {step} "
                f"({self._non_finite_steps} in a row); skipping the optimizer step."
            )
            if self._non_finite_steps >= 5 and self.trainer is not None:
                warnings.warn("5 consecutive non-finite losses: stopping the run.")
                self.trainer.should_stop = True
            return None
        self._non_finite_steps = 0
        self._log_per_loss_gradient_norms(losses)

        self.log("train_losses/loss", losses["loss"], on_step=False, on_epoch=True)
        self.log(
            "train_losses/objectness",
            losses["loss_objectness"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/tau_id",
            losses["loss_tau_id"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics",
            losses["loss_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_pt_loss",
            losses["kinematics_log_pt_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_delta_eta_loss",
            losses["kinematics_delta_eta_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_phi_chord_loss",
            losses["kinematics_phi_chord_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/kinematics_log_mass_loss",
            losses["kinematics_log_mass_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/charge", losses["loss_charge"], on_step=False, on_epoch=True
        )
        self.log(
            "train_losses/meson_class",
            losses["loss_meson_class"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/consistency",
            losses["loss_consistency"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/charge_count",
            losses["loss_charge_count"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/parent_kinematics",
            losses["loss_parent_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/parent_charge",
            losses["loss_parent_charge"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/parent_decay_mode",
            losses["loss_parent_decay_mode"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/soft_parent_kinematics",
            losses["loss_soft_parent_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/soft_parent_charge",
            losses["loss_soft_parent_charge"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_losses/soft_parent_decay_mode",
            losses["loss_soft_parent_decay_mode"],
            on_step=False,
            on_epoch=True,
        )
        # Matched pairs per step: if the daughter losses move because the matcher
        # stopped finding targets, this moves with them. on_step as well as
        # on_epoch, because an epoch mean hides a collapse that lasts a chunk.
        self.log("counts/num_matched", losses["num_matched"],
                 on_step=True, on_epoch=True)
        if self.threshold_scan_enabled and self.global_step % self.threshold_buffer_every == 0:
            self._buffer_for_threshold_scan(batch, outputs, targets)
        # What the matcher is actually deciding on. cost/class_share rising over
        # training is the signature of the assignment being driven by predicted
        # identity rather than position.
        for key, value in losses.items():
            if key.startswith("cost/"):
                self.log(key, value, on_step=False, on_epoch=True)
            elif key.startswith("weighted_loss/"):
                name = key.removeprefix("weighted_loss/")
                self.log(
                    f"train_weighted_losses/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                )
        self.log("counts/num_charge_supervised", losses["num_charge_supervised"],
                 on_step=False, on_epoch=True)
        self.log("counts/num_meson_class_supervised",
                 losses["num_meson_class_supervised"], on_step=False, on_epoch=True)

        return losses["loss"]

    def _buffer_for_threshold_scan(
        self, batch, outputs, targets, buffer=None
    ) -> None:
        """
        Stash what the dR matching needs from this training batch.

        Signal jets only. Objectness is trained on signal jets alone -- whether a
        jet is a tau at all is the tauID head's decision, not objectness's -- so
        on background jets the objectness scores are untrained and arbitrary.
        Letting them into the scan would count every such query above threshold
        as a false positive against zero true daughters, and the background
        jets would then decide the threshold rather than the matching quality
        on taus.

        Throttled to every `threshold_scan.buffer_every_n_steps` training step:
        the ring buffer holds ~17 batches, so buffering every step copied six
        tensors to the host per step (each forcing a device sync) and discarded
        97% of them.
        """
        reco_jet = batch[6]
        target_mask = targets["particles_mask"].bool()
        if "is_tau" in targets:
            signal = targets["is_tau"].bool()
        else:
            signal = target_mask.any(dim=1)
        if not bool(signal.any()):
            return
        target_kinematics = targets["particles_kinematics"][signal]
        target_mask = target_mask[signal]
        outputs = {
            k: v[signal]
            for k, v in outputs.items()
            if k in ("pred_logits", "pred_kinematics", "pred_meson_class_logits")
        }
        reco_jet = {k: v[signal] for k, v in reco_jet.items()}
        with torch.no_grad():
            # Charged or neutral, per predicted and per true daughter: the
            # decay-mode objective of the scan counts prongs and neutrals.
            charged_class = self.meson_class_repr_pdg.to(self.device) == 211
            pred_charged = charged_class[outputs["pred_meson_class_logits"].argmax(-1)]
            true_charged = charged_class[targets["particles_meson_class_ohe"][signal].argmax(-1)]
            scores = torch.softmax(outputs["pred_logits"].float(), dim=-1)[..., 0]
            pred_eta = outputs["pred_kinematics"][..., 1].float() + reco_jet["eta"][:, None]
            pred_phi = reco_jet["phi"][:, None] + torch.atan2(
                outputs["pred_kinematics"][..., 2].float(),
                outputs["pred_kinematics"][..., 3].float(),
            )
            true_eta = target_kinematics[..., 1].float() + reco_jet["eta"][:, None]
            true_phi = reco_jet["phi"][:, None] + torch.atan2(
                target_kinematics[..., 2].float(), target_kinematics[..., 3].float()
            )
        destination = self.threshold_buffer if buffer is None else buffer
        destination.add(
            scores, pred_eta, pred_phi, pred_charged, true_eta, true_phi, target_mask, true_charged
        )

    def on_validation_start(self) -> None:
        """Recalibrate the objectness threshold on recent training jets."""
        if not self.threshold_scan_enabled or self.trainer is None:
            return
        if self.trainer.sanity_checking:
            return
        try:
            best, by_threshold = self.threshold_buffer.scan(
                self.threshold_grid, objective=self.threshold_scan_objective
            )
        except Exception as exc:  # pragma: no cover - never fail a run on this
            warnings.warn(f"threshold scan failed: {exc}")
            return
        if best is None:
            return
        # The scan is a bare argmax over the grid, so it always returns
        # something -- including when the model is broken and every threshold is
        # equally worthless, in which case the least-bad option is to suppress
        # every prediction and the argmax runs to the grid's upper edge. The
        # 2026-09-20 run wrote 0.9 into the checkpoint that way, which then read
        # as a tuning result rather than as the wreckage it was. Refuse the
        # update when the objective has collapsed against the best this run has
        # ever scanned, and flag an optimum sitting on a grid edge: that means
        # either the grid is too narrow or the model is broken, and either way
        # the value should not pass silently.
        accuracy = float(by_threshold[best]["decay_mode_accuracy"])
        if accuracy < 0.5 * self._best_scan_accuracy:
            warnings.warn(
                f"threshold scan objective collapsed "
                f"(decay_mode_accuracy {accuracy:.3f} vs best "
                f"{self._best_scan_accuracy:.3f} this run); keeping threshold "
                f"{float(self.score_threshold_calibrated):.2f} instead of {best:.2f}"
            )
        else:
            self._best_scan_accuracy = max(self._best_scan_accuracy, accuracy)
            self.score_threshold_calibrated.fill_(float(best))
            if bool(
                np.isclose(best, self.threshold_grid[0])
                or np.isclose(best, self.threshold_grid[-1])
            ):
                warnings.warn(
                    f"threshold scan optimum at grid edge ({best:.2f}); the grid "
                    f"[{self.threshold_grid[0]:.2f}, {self.threshold_grid[-1]:.2f}] "
                    f"may be too narrow, or the model may be diverging"
                )
        # The threshold actually in use, which after a refused update is the
        # previous one, not the scan's argmax. Both are logged: a gap between
        # them is the guard having fired.
        self.log("threshold/score_threshold",
                 float(self.score_threshold_calibrated), on_step=False, on_epoch=True)
        self.log("threshold/scan_argmax", float(best), on_step=False, on_epoch=True)
        self.log("threshold/best_decay_mode_accuracy",
                 accuracy, on_step=False, on_epoch=True)
        self.log("threshold/best_f1", float(by_threshold[best]["f1"]),
                 on_step=False, on_epoch=True)

    def _accumulate_jet_level(self, batch, outputs, targets) -> None:
        """
        Collapse this batch's predicted and true daughter sets to jet level.

        Predictions are selected by the same objectness threshold inference
        uses, so what is logged is what a downstream user would actually get,
        not an oracle-selected best case. Meson classes are handed to the
        evaluators as their representative PDG id (see meson_class_repr_pdg),
        which is what ml-tau-data's decay-mode classifier expects.
        """
        reco_jet = batch[6]
        (
            target_kinematics,
            target_charge_cls,
            target_meson_class,
            target_mask,
            target_is_tau,
            _target_parent_charge,
            _target_parent_decay_mode,
        ) = self._extract_set_targets(targets)

        charge_lut = torch.tensor([-1, 0, 1], device=self.device)
        repr_pdg = self.meson_class_repr_pdg.to(self.device)

        scores = torch.softmax(outputs["pred_logits"].float(), dim=-1)[..., 0]
        pred = s2s.daughters_to_jet_level(
            kin=outputs["pred_kinematics"].float(),
            charge=charge_lut[outputs["pred_charge_logits"].argmax(-1)],
            pdg=repr_pdg[outputs["pred_meson_class_logits"].argmax(-1)],
            valid=scores >= float(self.score_threshold_calibrated),
            reco_jet=reco_jet,
        )
        # Padded slots carry ignore_index; clamp before the lookup and let the
        # mask decide what counts.
        true = s2s.daughters_to_jet_level(
            kin=target_kinematics.float(),
            charge=charge_lut[target_charge_cls.clamp_min(0)],
            pdg=repr_pdg[target_meson_class.clamp_min(0)],
            valid=target_mask,
            reco_jet=reco_jet,
        )
        tau_score = None
        if "is_tau" in outputs:
            tau_score = torch.softmax(outputs["is_tau"].float(), dim=-1)[:, 1]
        # The evaluators bin efficiencies against the gen tau and the reco jet,
        # so those travel with the predictions.
        self.val_jets.update(
            pred,
            true,
            target_is_tau,
            tau_score,
            {
                "gen_jet_tau_p4s": batch[5],
                "reco_jet_p4s": batch[6],
                "gen_jet_p4s": batch[7],
            },
        )

    def on_validation_epoch_end(self) -> None:
        """Turn the accumulated jets into figures and scalars, then start over."""
        if self.trainer is None or self.trainer.sanity_checking:
            self.val_jets.reset()
            self.validation_threshold_buffer.reset()
            return
        tb_logger = None
        for logger in self.trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "add_figure"):
                tb_logger = experiment
                break
        try:
            scalars = s2s.log_set_to_set_metrics(
                self.val_jets, tb_logger, self.cfg, self.current_epoch, dataset="val"
            )
        except Exception as exc:  # pragma: no cover - never fail a run on a plot
            warnings.warn(f"jet-level validation logging failed: {exc}")
            scalars = {}
        for name, value in scalars.items():
            self.log(f"val_jet/{name}", value, on_step=False, on_epoch=True)
        if self.threshold_scan_enabled:
            try:
                best, by_threshold = self.validation_threshold_buffer.scan(
                    self.threshold_grid, objective=self.threshold_scan_objective
                )
            except Exception as exc:  # pragma: no cover - never fail a run on this
                warnings.warn(f"validation threshold scan failed: {exc}")
            else:
                if best is not None:
                    self.log("threshold/validation_argmax", float(best), on_step=False, on_epoch=True)
                    self.log(
                        "threshold/validation_best_decay_mode_accuracy",
                        float(by_threshold[best]["decay_mode_accuracy"]),
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        "threshold/validation_best_f1",
                        float(by_threshold[best]["f1"]),
                        on_step=False,
                        on_epoch=True,
                    )
        self.val_jets.reset()
        self.validation_threshold_buffer.reset()

    def validation_step(self, batch, _batch_idx):
        outputs, targets, weights, gen_jet_tau_p4, kinematics_reference_p4 = self.forward(batch)
        (
            target_kinematics,
            target_charge_cls,
            target_meson_class,
            target_mask,
            target_is_tau,
            target_parent_charge,
            target_parent_decay_mode,
        ) = self._extract_set_targets(targets)

        losses = self.criterion(
            outputs=outputs,
            target_kinematics=target_kinematics,
            target_charge_cls=target_charge_cls,
            target_meson_class=target_meson_class,
            target_mask=target_mask,
            target_parent_charge=target_parent_charge,
            target_parent_decay_mode=target_parent_decay_mode,
            target_parent_p4=gen_jet_tau_p4,
            kinematics_reference_p4=kinematics_reference_p4,
            target_is_tau=target_is_tau,
            jet_weights=weights,
            objectness_threshold=self.score_threshold_calibrated,
        )

        self.log("val_losses/loss", losses["loss"], on_step=False, on_epoch=True)
        self.log(
            "val_losses/objectness",
            losses["loss_objectness"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/tau_id",
            losses["loss_tau_id"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics",
            losses["loss_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_log_pt_loss",
            losses["kinematics_log_pt_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_delta_eta_loss",
            losses["kinematics_delta_eta_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_phi_chord_loss",
            losses["kinematics_phi_chord_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/kinematics_log_mass_loss",
            losses["kinematics_log_mass_loss"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/charge", losses["loss_charge"], on_step=False, on_epoch=True
        )
        self.log(
            "val_losses/meson_class",
            losses["loss_meson_class"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/consistency",
            losses["loss_consistency"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/charge_count",
            losses["loss_charge_count"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/parent_kinematics",
            losses["loss_parent_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/parent_charge",
            losses["loss_parent_charge"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/parent_decay_mode",
            losses["loss_parent_decay_mode"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/soft_parent_kinematics",
            losses["loss_soft_parent_kinematics"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/soft_parent_charge",
            losses["loss_soft_parent_charge"],
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_losses/soft_parent_decay_mode",
            losses["loss_soft_parent_decay_mode"],
            on_step=False,
            on_epoch=True,
        )
        for key, value in losses.items():
            if key.startswith("weighted_loss/"):
                name = key.removeprefix("weighted_loss/")
                self.log(
                    f"val_weighted_losses/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                )

        if not self.trainer.sanity_checking:
            self._accumulate_jet_level(batch, outputs, targets)
            if self.threshold_scan_enabled:
                self._buffer_for_threshold_scan(
                    batch, outputs, targets, self.validation_threshold_buffer
                )
        return losses["loss"]

    def on_before_optimizer_step(self, optimizer) -> None:
        """
        Per-step optimizer diagnostics, logged because nothing else records them.

        A late-training blow-up in every head at once, in train AND val, with a
        smoothly decaying learning rate, cannot be attributed from the loss
        curves alone: they show the damage, not the cause. These three do.

        grad/total_norm is the 2-norm BEFORE clipping (Lightning calls this hook
        after unscaling and before gradient_clip_val applies), so it says whether
        the optimizer saw an outlier gradient and whether clipping was active at
        all. A norm that starts oscillating and growing while the loss does the
        same is progressive sharpening -- the step size outrunning the curvature
        -- and points at the schedule. A single isolated spike points at one bad
        batch instead.

        grad/amp_scale matters because with 16-mixed a shrinking scale means
        GradScaler is catching infinities and silently skipping steps, which
        looks like a plateau rather than an error.

        optim/beta1 is logged because OneCycleLR CYCLES it by default: nothing in
        this repository asked for momentum to sweep 0.95 -> 0.85 -> 0.95, but it
        does, and the second half of that sweep coincides with the unstable
        phase.
        """
        if self.trainer is None:
            return
        if self.criterion.loss_weighting is not None:
            self.criterion.loss_weighting.freeze_inactive_log_variances()
            for name, log_variance in (
                self.criterion.loss_weighting.log_variances.items()
            ):
                if log_variance.grad is not None:
                    # Signed: positive lowers s and raises w under gradient
                    # descent; negative raises s and lowers w.
                    self.log(
                        f"grad/log_variance/{name}",
                        log_variance.grad.detach(),
                        on_step=True,
                        on_epoch=False,
                    )
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total = torch.sqrt(
                torch.stack([(g.detach() ** 2).sum() for g in grads]).sum()
            )
            self.log("grad/total_norm", total, on_step=True, on_epoch=False)
            # Against the configured clip value, not a hardcoded 1.0: raising
            # gradient_clip_val used to leave this metric silently reporting
            # against the old bound, so it stopped meaning "clipping fired".
            self.log(
                "grad/clipped",
                (total > self.grad_clip_val).float(),
                on_step=True,
                on_epoch=True,
            )

            # Outlier guard. Clipping is NOT one: it renormalises a gradient of
            # norm 1e11 to the clip value and applies it anyway, in whatever
            # direction that garbage points, and once the norm overflows to inf
            # the clip coefficient is 1/inf = 0, so every step silently becomes
            # a no-op and the run looks alive while only weight decay still
            # moves the weights. That is exactly how the 2026-09-20 run spent
            # its last 11 600 steps. Zeroing the gradient here skips the step
            # instead, which is only possible because grad_skip_norm sits well
            # above the healthy distribution (steps 150-55 000 of that run:
            # p50 2.8, p99 10.0, max 13.4) and therefore means something when
            # it fires.
            #
            # The threshold describes TRAINED weights. Fresh ones have a norm
            # in the hundreds -- ~270 on this model -- and absorbing that is
            # gradient_clip_val's job, not the guard's. Arming the guard from
            # step 0 is therefore not conservative but a deadlock: the gradient
            # is zeroed, so the weights do not move, so the next norm is just
            # as large, so every step skips until grad_skip_patience aborts the
            # run. Run 61024711 (2026-09-21) died that way at step 51 of epoch
            # 0, having never applied a single real update. Hence the warm-up
            # grace, during which a large but finite norm is left to the clip.
            # A NON-FINITE norm still skips at any step: it is never legitimate,
            # and bf16-mixed has no GradScaler to catch it.
            #
            # Caveat worth knowing: zeroing the gradient keeps the outlier out
            # of AdamW's m and v and out of the clip, but it does not freeze the
            # weights. AdamW's update is lr * m_hat / (sqrt(v_hat) + eps), so a
            # zeroed step still moves along the existing momentum and still
            # applies decoupled weight decay. That is the point -- the step
            # stays bounded by the HEALTHY history instead of following a
            # gradient of norm 1e11 -- but it is a damped step, not a true skip.
            if self.grad_skip_norm > 0.0:
                total_value = float(total)  # one host sync; logging forces one
                warming_up = (
                    self.trainer.global_step < self.grad_skip_warmup_steps
                )
                if not math.isfinite(total_value):
                    skipped = True
                elif total_value > self.grad_skip_norm:
                    skipped = not warming_up
                    if warming_up and not self._warned_warmup_grad:
                        self._warned_warmup_grad = True
                        warnings.warn(
                            f"gradient norm {total_value:.1f} above "
                            f"{self.grad_skip_norm} at step "
                            f"{self.trainer.global_step}, within the first "
                            f"{self.grad_skip_warmup_steps} steps: left to "
                            f"gradient_clip_val, not skipped. Expected once at "
                            f"start-up; if grad/total_norm has not fallen below "
                            f"the threshold by the end of the grace window, the "
                            f"guard will start skipping and the run will abort."
                        )
                else:
                    skipped = False
                if skipped:
                    for parameter in self.parameters():
                        if parameter.grad is not None:
                            parameter.grad.zero_()
                    self._consecutive_skips += 1
                    if self._consecutive_skips > self.grad_skip_patience:
                        raise RuntimeError(
                            f"gradient norm non-finite or above "
                            f"{self.grad_skip_norm} for "
                            f"{self._consecutive_skips} consecutive steps "
                            f"(last: {total_value}); the run has diverged."
                        )
                else:
                    self._consecutive_skips = 0
                self.log(
                    "grad/skipped", float(skipped), on_step=True, on_epoch=True
                )

        scaler = getattr(
            getattr(self.trainer, "precision_plugin", None), "scaler", None
        )
        if scaler is not None:
            self.log("grad/amp_scale", float(scaler.get_scale()),
                     on_step=True, on_epoch=False)

        for group in optimizer.param_groups:
            betas = group.get("betas")
            if betas is not None:
                self.log("optim/beta1", float(betas[0]), on_step=True, on_epoch=False)
            break

    def on_validation_end(self) -> None:
        """
        Keep the whole val metric set from the best epoch.

        Deliberately `on_validation_end` rather than `on_validation_epoch_end`:
        the epoch-end reduction of `self.log(..., on_epoch=True)` values has not
        happened yet in the latter, so `trainer.callback_metrics` would still
        hold the previous epoch's numbers.
        """
        if self.trainer is None or self.trainer.sanity_checking:
            return
        metrics = {
            name: float(value)
            for name, value in self.trainer.callback_metrics.items()
            if name.startswith("val_losses/") and hasattr(value, "item")
        }
        current = metrics.get("val_losses/loss")
        if current is None:
            return
        best = self.best_val_metrics
        if best is None or current < best["val_losses/loss"]:
            self.best_val_metrics = {
                **metrics,
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
            }

    def predict_step(self, batch, _batch_idx):
        outputs, _, _, _, _ = self.forward(batch)

        object_scores = torch.softmax(outputs["pred_logits"], dim=-1)[..., 0]
        # The calibrated threshold from the checkpoint, not the static config
        # value: validation reports every val_jet/* number at this operating
        # point, so prediction must use the same one.
        pred_mask = object_scores >= self.score_threshold_calibrated.to(object_scores.dtype)

        charge_class = outputs["pred_charge_logits"].argmax(dim=-1)
        charge_value_lut = outputs["pred_charge_logits"].new_tensor(
            [-1, 0, 1], dtype=torch.long
        )
        pred_charge = charge_value_lut[charge_class]

        meson_class = outputs["pred_meson_class_logits"].argmax(dim=-1)

        result = {
            "pred_kinematics": outputs["pred_kinematics"],
            "pred_charge_logits": outputs["pred_charge_logits"],
            "pred_meson_class_logits": outputs["pred_meson_class_logits"],
            "pred_logits": outputs["pred_logits"],
            "pred_scores": object_scores,
            "pred_mask": pred_mask,
            "pred_charge": pred_charge,
            "pred_meson_class": meson_class,
        }

        if "is_tau" in outputs:
            result["is_tau_logits"] = outputs["is_tau"]
            result["is_tau"] = torch.softmax(outputs["is_tau"], dim=-1)[..., 1]
            # Whether the jet is a tau at all is the tauID head's decision, not
            # objectness's (see SetCriterion): on a background jet the
            # objectness scores are untrained. pred_mask is the gated set that
            # downstream code should use; the raw objectness selection is kept
            # alongside for studies on true tau jets.
            result["pred_mask_objectness"] = pred_mask
            tagged = result["is_tau"] >= self.tau_id_threshold
            result["pred_mask"] = pred_mask & tagged[:, None]

        return result

    def test_step(self, batch, _batch_idx):
        return self.predict_step(batch, _batch_idx)

    def configure_optimizers(self) -> Any:
        base_lr = self.cfg.training.lr
        opt_cfg = self.cfg.training.get("optimizer", None) or {}
        weight_decay = float(opt_cfg.get("weight_decay", 1e-2))

        # Weight decay is a prior towards zero that makes sense for weight
        # matrices and none for biases, LayerNorm gains, the DETR query
        # embeddings, the cls token or learned log variances. The latter get an
        # optional third parameter group when automatic weighting is enabled.
        skip = set(self.ParTauDETR.no_weight_decay())
        decay, no_decay = [], []
        for name, parameter in self.ParTauDETR.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name in skip or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        parameter_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.criterion.loss_weighting is not None:
            parameter_groups.append(
                {
                    "params": self.criterion.loss_weighting.parameters(),
                    "weight_decay": 0.0,
                }
            )
        optimizer = torch.optim.AdamW(parameter_groups, lr=base_lr)

        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if estimated_steps is None or estimated_steps <= 0:
            max_epochs = self.cfg.training.trainer.max_epochs
            estimated_steps_per_epoch = 500
            total_steps = max_epochs * estimated_steps_per_epoch
            print(
                f"Warning: Using estimated total_steps={total_steps} (estimated_stepping_batches not available)"
            )
        else:
            total_steps = estimated_steps
            print(
                f"Using calculated total_steps={total_steps} from estimated_stepping_batches"
            )

        # pct_start is the fraction of total_steps spent ramping UP, from
        # max_lr/div_factor (div_factor defaults to 25) to max_lr; the rest is
        # the cosine decay. It does not change the peak height -- that is
        # training.lr -- and it barely changes how long the schedule dwells near
        # the peak, because the ramp side shortening is cancelled by the decay
        # side lengthening (measured on this run's 184 750 steps: 11 854 steps
        # within 1% of max_lr at 0.3, 11 806 at 0.1). What it changes is WHEN
        # that dwell happens. At the default 0.3 the peak landed at step 55 425
        # and the near-peak window was steps 51 818-63 671, i.e. late, on a model
        # sharp enough that the step size outran the curvature -- the 2026-09-20
        # run diverged at step 60 449, inside that window. At 0.1 the peak is at
        # step 18 475, while the model is still under-trained and the loss
        # surface flatter, and 90% of the run is annealing.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=float(opt_cfg.get("pct_start", 0.1)),
            anneal_strategy="cos",
            # Default True sweeps AdamW's beta1 0.95 -> 0.85 -> 0.95 alongside
            # the learning rate; nothing here asked for that, and beta1 was at
            # its minimum where the 250-epoch run diverged. Off by default.
            cycle_momentum=bool(opt_cfg.get("cycle_momentum", False)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
