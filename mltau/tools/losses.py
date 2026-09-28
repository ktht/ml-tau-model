import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss



def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Mean of `values` weighted by `weights`: sum(w*v) / sum(w).

    NOT mean(w*v). The latter's magnitude scales with the mean of the weights,
    i.e. with the class composition of the batch, so the reported loss moves
    even when the model does not -- a batch of only background and a batch of
    only signal differ by the weight ratio alone. cls_weight is built so that
    signal and background contribute equally within each theta-p bin, which is
    a statement about the weighted mean; mean(w*v) does not implement it.
    """
    if values.numel() == 0:
        return values.new_zeros(())
    return (values * weights).sum() / (weights.sum() + 1e-8)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss"""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Class weights (tensor of size num_classes)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class indices
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get predicted probabilities for true classes
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Usage in your model:
# self.focal_loss = FocalLoss(alpha=torch.tensor([0.1, 1.0, 1.0, 1.0]), gamma=2.0)  # Lower weight for background


class SigmoidFocalLoss(nn.Module):
    """Wrapper to make sigmoid_focal_loss behave like a module for consistency."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets.float(),  # targets need to be float for sigmoid
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class TauLoss(nn.Module):
    """Unified loss module for Tau tagging, charge, decay mode, and kinematics."""

    def __init__(
        self,
        l_m=0.2,
        label_smoothing=0.1,
        kinematics_weights=None,
        kinematics_scales=None,
    ):
        super().__init__()
        self.l_m = l_m
        # Per-component weights for the kinematics regression loss. l_m is kept
        # as the legacy mass weight and is used when kinematics_weights is None.
        self.kinematics_weights = dict(
            kinematics_weights
            or {"log_pt": 1.0, "delta_eta": 1.0, "phi_chord": 1.0, "log_mass": l_m}
        )
        # Residual scales, one per component, in the units of that component.
        # Each residual is divided by its scale before the Huber, so what the
        # loss sees is a dimensionless "error in units of the spread of this
        # target" and every component is comparable by construction. Defaults of
        # 1.0 reproduce the unscaled behaviour.
        # None means "not measured": the loss then falls back to the legacy
        # unscaled form (see _compute_kinematics_loss_per_sample) rather than
        # pretending a scale of 1 is a measurement. Use from_config().
        self.kinematics_scales = (
            dict(kinematics_scales) if kinematics_scales else None
        )
        # Tagging: all jets (background=0, signal=1)
        self.tag_loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )
        # Charge: signal taus only
        self.charge_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # Decay Mode: signal taus only
        self.dm_loss_fn = nn.CrossEntropyLoss(reduction="none")
        # Kinematics: signal taus only
        self.kin_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    @classmethod
    def from_config(cls, tau_loss_cfg=None, owner: str = "model") -> "TauLoss":
        """
        Build from a `tau_loss` config block. Every key is optional:

            tau_loss:
                l_m: 0.2                 # legacy mass weight, used if no weights
                label_smoothing: 0.1
                kinematics_scales:  {log_pt: .., delta_eta: .., phi_chord: .., log_mass: ..}
                kinematics_weights: {log_pt: 1, delta_eta: 1, phi_chord: 1, log_mass: 0.2}

        ParTauDETR, MultiParTau and SingleParTau all construct their loss here,
        so the three are configured the same way. The scales are properties of
        the data -- the spread of each regression target -- and cannot be
        defaulted sensibly: daughter-level targets (DETR) and tau-level targets
        (ParT) differ by an order of magnitude in delta_eta and delta_phi.
        Without them the loss uses its legacy unscaled form and says so;
        measure them with mltau/scripts/measure_kinematics_scales.py.
        """
        cfg = tau_loss_cfg if tau_loss_cfg is not None else {}
        weights = cfg.get("kinematics_weights", None)
        scales = cfg.get("kinematics_scales", None)
        if not scales:
            warnings.warn(
                f"[{owner}] tau_loss.kinematics_scales is not set. The kinematics "
                "loss runs in its legacy unscaled form (Huber in raw units, linear "
                "phi chord), in which the angular components carry almost no "
                "gradient. Measure the scales with "
                "`python3 mltau/scripts/measure_kinematics_scales.py` and put "
                "them in the config.",
                stacklevel=2,
            )
        return cls(
            l_m=float(cfg.get("l_m", 0.2)),
            label_smoothing=float(cfg.get("label_smoothing", 0.1)),
            kinematics_weights=(
                {k: float(v) for k, v in weights.items()} if weights else None
            ),
            kinematics_scales=(
                {k: float(v) for k, v in scales.items()} if scales else None
            ),
        )

    def compute_tagging_loss(self, predictions, targets, weights):
        """CrossEntropy loss for background vs signal classification."""
        loss = self.tag_loss_fn(predictions, targets.long())
        return weighted_mean(loss, weights)

    def compute_charge_loss(self, predictions, targets, weights):
        """BCE loss for charge classification (+1 vs -1)."""
        # Map physical charges {-1, 1} to binary labels {0, 1}.
        # (targets == 1) maps +1 -> 1 and -1 -> 0.
        binary_targets = (targets == 1).float()
        loss = self.charge_loss_fn(predictions, binary_targets)
        return weighted_mean(loss, weights)

    def compute_decay_mode_loss(self, predictions, targets, weights):
        """CrossEntropy loss for decay mode classification."""
        # CrossEntropyLoss expects (N, C) float targets for probabilities or (N,) long targets for indices
        if targets.ndim > 1:
            loss = self.dm_loss_fn(predictions, targets.float())
        else:
            loss = self.dm_loss_fn(predictions, targets.long())
        return weighted_mean(loss, weights)

    def _compute_kinematics_loss_per_sample(self, predictions, targets):
        """
        Per-sample Huber loss for (log pt, deta, phi_chord, log m).

        Every residual is divided by its component's scale first, so all four
        enter the Huber in the same units: multiples of the spread of that
        target. That is what makes the weights below comparable.

        Scaling is not cosmetic. The components differ by more than an order of
        magnitude in natural size -- log_pt has a spread of ~0.9 while delta_eta
        has ~0.08 -- and Huber is quadratic below delta=1, so an unscaled
        delta_eta residual contributes ~100x less loss and gradient than an
        unscaled log_pt residual of the same relative size. Hand-tuned weights
        cannot fix that, because the ratio between a quadratic and a linear term
        keeps moving as the model improves; dividing by the scale fixes it once.

        Dividing by the scale also puts the Huber knee at one scale unit, which
        is where it belongs: quadratic for typical residuals, linear for the
        tail, instead of quadratic everywhere (delta_eta) or linear everywhere
        (a chord of order 1 at initialisation).
        """
        scaled = self.kinematics_scales is not None
        s = self.kinematics_scales or {
            "log_pt": 1.0, "delta_eta": 1.0, "phi_chord": 1.0, "log_mass": 1.0
        }
        log_pt_loss = self.kin_loss_fn(
            predictions[:, 0] / s["log_pt"], targets[:, 0] / s["log_pt"]
        )
        delta_eta_loss = self.kin_loss_fn(
            predictions[:, 1] / s["delta_eta"], targets[:, 1] / s["delta_eta"]
        )
        # Phi chord: (sin, cos) treated as a 2D unit-vector difference, i.e. the
        # chord length between the predicted and the true angle. Feeding the
        # scaled chord through the same Huber makes this component quadratic
        # near zero like the others, rather than linear everywhere.
        chord = torch.sqrt(
            (predictions[:, 2] - targets[:, 2]) ** 2
            + (predictions[:, 3] - targets[:, 3]) ** 2
            + 1e-8
        )
        if scaled:
            phi_chord = chord / s["phi_chord"]
            phi_chord_loss = self.kin_loss_fn(phi_chord, torch.zeros_like(phi_chord))
        else:
            # Legacy form, kept for a loss without measured scales: the chord
            # enters linearly. Putting an unscaled chord (~0.05 for tau-level
            # targets) through a Huber with its knee at 1 would make this term
            # quadratic everywhere and shrink its gradient twentyfold, which is
            # what silently happened to MultiParTau between 2026-09-15 and this
            # change.
            phi_chord_loss = chord
        if predictions.size(-1) == 5:
            log_mass_loss = self.kin_loss_fn(
                predictions[:, 4] / s["log_mass"], targets[:, 4] / s["log_mass"]
            )
        else:
            log_mass_loss = None

        # Combined per-sample loss, weighted per component and normalised by the
        # weight sum so the overall scale does not move when weights are retuned.
        w = self.kinematics_weights
        per_sample_loss = (
            w["log_pt"] * log_pt_loss
            + w["delta_eta"] * delta_eta_loss
            + w["phi_chord"] * phi_chord_loss
            + (w["log_mass"] * log_mass_loss if log_mass_loss is not None else 0)
           ) / (sum(w[key] for key in ("log_pt", "delta_eta", "phi_chord"))
               + (w["log_mass"] if log_mass_loss is not None else 0) + 1e-12)

        return per_sample_loss, {
            "log_pt": log_pt_loss,
            "delta_eta": delta_eta_loss,
            "phi_chord": phi_chord_loss,
            **({"log_mass": log_mass_loss} if log_mass_loss is not None else {}),
        }

    def compute_kinematics_loss(self, predictions, targets, weights):
        """Huber loss for (log pt, deta, phi_chord, log m)."""
        per_sample_loss, components_per_sample = self._compute_kinematics_loss_per_sample(
            predictions, targets
        )

        components = {
            k: weighted_mean(v, weights) for k, v in components_per_sample.items()
        }

        return weighted_mean(per_sample_loss, weights), components

    def compute_multi_task_losses(self, predictions_dict, targets_dict, sample_weights):
        """Helper for MultiParTau to compute all 4 task losses at once with masking."""
        is_tau_mask = targets_dict["is_tau"].bool()

        # 1. Tagging loss — all jets
        tag_loss = self.compute_tagging_loss(
            predictions_dict["is_tau"], targets_dict["is_tau"], sample_weights
        )

        if not is_tau_mask.any():
            zero = tag_loss.new_zeros(())
            return torch.stack([tag_loss, zero, zero, zero]), {}

        tau_weights = sample_weights[is_tau_mask]

        # 2. Decay Mode loss — signal only
        dm_loss = self.compute_decay_mode_loss(
            predictions_dict["decay_mode"][is_tau_mask],
            targets_dict["decay_mode"][is_tau_mask],
            tau_weights,
        )

        # 3. Charge loss — signal only
        charge_loss = self.compute_charge_loss(
            predictions_dict["charge"][is_tau_mask],
            targets_dict["charge"][is_tau_mask],
            tau_weights,
        )

        # 4. Kinematics loss — signal only
        kin_loss, kin_components = self.compute_kinematics_loss(
            predictions_dict["kinematics"][is_tau_mask],
            targets_dict["kinematics"][is_tau_mask],
            tau_weights,
        )

        return torch.stack([tag_loss, dm_loss, charge_loss, kin_loss]), kin_components

    def compute_combined_loss(
        self,
        predictions,
        targets,
        weights,
        w_tag=1.0,
        w_dm=1.0,
        w_charge=1.0,
        w_kin=1.0,
    ):
        """
        Compute a single scalar loss representing the weighted average of per-jet
        combined losses. This is the logic used for validation monitoring and
        combined-loss training.
        """
        is_tau_mask = targets["is_tau"].bool()

        # 1. Tagging loss — all jets
        tag_per_jet = self.tag_loss_fn(predictions["is_tau"], targets["is_tau"].long())
        combined_per_jet = w_tag * tag_per_jet

        if is_tau_mask.any():
            # 2. Decay Mode loss — signal only
            dm_target = targets["decay_mode"][is_tau_mask]
            if dm_target.ndim > 1:
                dm_per_jet = self.dm_loss_fn(
                    predictions["decay_mode"][is_tau_mask],
                    dm_target.float(),
                )
            else:
                dm_per_jet = self.dm_loss_fn(
                    predictions["decay_mode"][is_tau_mask],
                    dm_target.long(),
                )

            # 3. Charge loss — signal only
            charge_targets = (targets["charge"][is_tau_mask] == 1).float()
            charge_per_jet = self.charge_loss_fn(
                predictions["charge"][is_tau_mask],
                charge_targets,
            )

            # 4. Kinematics loss — signal only
            kin_per_jet, _ = self._compute_kinematics_loss_per_sample(
                predictions["kinematics"][is_tau_mask],
                targets["kinematics"][is_tau_mask],
            )

            # Add signal-only terms into combined per-jet loss
            combined_per_jet[is_tau_mask] += (
                w_dm * dm_per_jet + w_charge * charge_per_jet + w_kin * kin_per_jet
            )

        # Multiply each jet's combined loss by its weight, then average
        return weighted_mean(combined_per_jet, weights)
