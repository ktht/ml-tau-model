"""
Jet-level performance logging for the set-to-set (ParTauDETR) model.

The loss curves say how well the criterion is being minimised; they do not say
whether the reconstructed tau is right. MultiParTau answered that with the
evaluators in `mltau.tools.evaluation`, and this module reuses exactly those, so
the plots are the same ones, produced by the same code, and a DETR run can be
compared against a MultiParTau run panel by panel.

What has to be done here first is the derivation those evaluators assume. They
take per-jet quantities, and this model predicts a SET of daughters, so the
jet-level tau is built from the set the way physics says it is:

    tau charge = sum of the daughters' charges
    tau p4     = sum of the daughters' four-momenta
    decay mode = ml-tau-data's classification of the daughters' PDG ids

and then expressed in the same parameterisation the regression heads use
(log pt ratio, delta eta, sin/cos delta phi, log mass ratio, all relative to the
reco jet), which is what `log_all_kinematics_metrics` already knows how to
decode.

The decay mode is the one place that does NOT reuse the MultiParTau path. Its
DecayModeEvaluator takes a probability vector over 6 reduced classes, which a
set model has no equivalent of, and collapsing to 6 classes would throw away the
distinction between 1-prong-2-pi0 and 1-prong-3-pi0 that this model is able to
make. It uses the HPS confusion matrix from `decode_HPS` instead -- the same one
STS_model_eval_HPS.ipynb draws -- which takes integer labels and an arbitrary
category list, and carries a column for predictions outside it.
"""

import warnings

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mltau.tools.evaluation import charge_id as c
from mltau.tools.evaluation import decode_HPS as hps
from mltau.tools.evaluation import set_to_set_models as s2s_eval  # noqa: F401  (kept: jet-level helpers)

# The decay mode comes from ml-tau-data, the package that also writes
# gen_jet_tau_decaymode into the ntuples (see set_to_set_models.py for why).
# It is importable as the ml-tau-data submodule / a pip install / PYTHONPATH.
from ntupelizer.tools import tau_decaymode as tdm
from mltau.tools.logging import kinematics as kinematics_logging
from mltau.tools.logging import tagging as tagging_logging
from mltau.tools.logging.general import log_metrics_dict
from mltau.tools.partau_detr import decode_kinematics, decode_fractions, momentum_coordinates

# Decay mode is NOT derived here. It goes through ntupelizer.tools.tau_decaymode
# (ml-tau-data), which classifies each daughter by particle PROPERTY -- charged
# hadron, neutral hadron, lepton; photons and neutrinos ignored -- rather than by
# an enumerated PDG list, and which is the same code that writes the
# gen_jet_tau_decaymode column. The extended rare id (30) is used so that a
# four-prong reconstruction, which is 15 on the bare 5*(n_charged-1)+n_neutral
# grid, cannot be confused with a genuinely rare decay.


# Rows of the confusion matrix, fixed rather than taken from whatever happens
# to appear in the sample. np.unique(truth) would make the axes depend on the
# validation draw, so a class present in one epoch and absent in the next
# silently changes the matrix and the rows stop being comparable between epochs
# or runs.
#
# The upstream grid is 5*(n_charged - 1) + min(n_neutral, 4):
#   0-4    one prong,   0 to 4 neutrals
#   10-14  three prong, 0 to 4 neutrals
# Upstream folds more than three prongs into `rare`. Five-prong decays are
# pulled back out of that bucket here, from the prong count, into one row of
# their own: it is a per-mille branching fraction, but it is exactly the class
# a set model can express and a 6-class reduced scheme cannot, and it is the
# class most likely to vanish from a validation draw. It is one row rather than
# five because the neutral multiplicity of a per-mille class is not measurable.
# Leptonic decays carry their own upstream id and get a row as well.
# Even-prong reconstructions -- 2 or 4 charged daughters, which no tau decay
# produces -- are not rows: they are predictions that cannot be right, and they
# get predicted-only columns below.
_GRID_MODES = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
_FIVE_PRONG_ROW = len(_GRID_MODES)
_LEPTONIC_ROW = _FIVE_PRONG_ROW + 1
_RARE_ROW = _LEPTONIC_ROW + 1
DECAY_MODE_LABELS = (
    [f"1h{n}p" for n in range(5)]
    + [f"3h{n}p" for n in range(5)]
    + ["5h", "lep", "rare"]
)
# Columns that exist only on the PREDICTED axis. Pooling even-prong predictions
# into "other" would hide a failure that says something specific -- a lost or
# spurious track rather than an unreconstructable decay. Two columns rather
# than ten (5-9 and 15-19 on the grid) because the neutral multiplicity of an
# impossible prong count is not interesting.
EXTRA_PREDICTED_LABELS = ["2h", "4h", "other"]


def category_index(mode, n_charged, n_lepton, n_other):
    """
    Row of each jet in DECAY_MODE_LABELS, or -1 if it has none.

    `mode` is the upstream id with the extended rare value. The prong count is
    needed alongside because the id alone cannot say "five prongs": upstream
    returns `rare` for anything above three.
    """
    mode = np.asarray(mode)
    index = np.full(mode.shape, -1, dtype=np.int64)
    for row, grid_mode in enumerate(_GRID_MODES):
        index[mode == grid_mode] = row
    hadronic = (np.asarray(n_lepton) == 0) & (np.asarray(n_other) == 0)
    is_rare = mode == tdm.RARE_DECAY_MODE_EXT
    index[is_rare & hadronic & (np.asarray(n_charged) == 5)] = _FIVE_PRONG_ROW
    index[mode == tdm.LEPTONIC_DECAY_MODE] = _LEPTONIC_ROW
    index[is_rare & (index < 0)] = _RARE_ROW
    return index


def _rectangular_confusion(truth_row, pred_row, n_charged_pred):
    """
    Row-normalised matrix with extra predicted-only columns.

    Rows are the physical decay modes; columns are those plus 2-prong, 4-prong
    and everything else. Even prong counts take precedence over the row the
    prediction would otherwise land in: two charged hadrons sit on the bare grid
    as 5-9, but the impossible prong count is the more specific statement about
    what went wrong. Truth without a row (should not happen for generator
    daughters) is dropped, and the caller warns about it.
    """
    n_rows = len(DECAY_MODE_LABELS)
    n_cols = n_rows + len(EXTRA_PREDICTED_LABELS)

    column = np.where(pred_row >= 0, pred_row, n_cols - 1)  # no row -> "other"
    column[n_charged_pred == 2] = n_rows
    column[n_charged_pred == 4] = n_rows + 1
    column[n_charged_pred > 5] = n_cols - 1

    keep = truth_row >= 0
    counts = np.zeros((n_rows, n_cols), dtype=float)
    np.add.at(counts, (truth_row[keep], column[keep]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / np.where(totals == 0, np.nan, totals)


def _jagged(dense: np.ndarray, valid: np.ndarray) -> ak.Array:
    """Dense [N, S] plus a mask -> the jagged per-jet lists the helpers expect."""
    return ak.unflatten(dense[valid], valid.sum(axis=1))


def decay_modes(pdg: np.ndarray, valid: np.ndarray) -> dict:
    """
    Per-jet decay mode from the selected daughters, through ml-tau-data.

    Returns the upstream id (extended rare convention), the daughter counts by
    category, and the confusion-matrix row. The counts are returned because the
    id alone is ambiguous about the prong count once it says `rare`, and the
    even-prong columns need the count directly.

    A jet with no selected daughters has counts of zero and is `rare` upstream;
    for a prediction that is the honest label -- nothing was reconstructed.
    """
    rows = ak.to_list(_jagged(pdg, valid))
    counts = np.array(
        [tdm.count_daughters(row) for row in rows], dtype=np.int64
    ).reshape(-1, 4)
    n_charged, n_neutral, n_lepton, n_other = (counts[:, i] for i in range(4))
    mode = np.fromiter(
        (
            tdm.decay_mode_from_counts(*row, rare=tdm.RARE_DECAY_MODE_EXT)
            for row in counts
        ),
        dtype=np.int64,
        count=len(counts),
    )
    return {
        "decay_mode": mode,
        "n_charged": n_charged,
        "n_neutral": n_neutral,
        "n_lepton": n_lepton,
        "n_other": n_other,
        "category": category_index(mode, n_charged, n_lepton, n_other),
    }


def daughters_to_jet_level(kin, charge, pdg, valid, reco_jet, p4=None):
    """
    Collapse a set of daughters into per-jet quantities.

    Returns the summed charge, the reconstructed decay mode, and the summed tau
    expressed BOTH as a p4 and in the 5-component training parameterisation, the
    latter so that `log_all_kinematics_metrics` can be reused unchanged.

    Args:
        kin:    [B, S, 5] daughter kinematics in the training parameterisation
        charge: [B, S] integer charge in {-1, 0, +1}
        pdg:    [B, S] absolute PDG id
        valid:  [B, S] bool, which slots count
        reco_jet: dict of [B] tensors with pt, eta, phi, energy
    """
    eps = 1e-6
    pt_jet = reco_jet["pt"]
    eta_jet = reco_jet["eta"]
    phi_jet = reco_jet["phi"]

    if p4 is None:
        p4 = decode_kinematics(kin, pt_jet, eta_jet, phi_jet, reco_jet["energy"])
    px, py, pz, energy = (p4 * valid[..., None]).sum(1).unbind(-1)

    pt_tau = torch.sqrt(px**2 + py**2 + eps)
    eta_tau = torch.asinh(pz / pt_tau)
    phi_tau = torch.atan2(py, px)
    mass_tau = torch.sqrt(torch.clamp(energy**2 - (px**2 + py**2 + pz**2), min=0.0))

    mass_jet = torch.sqrt(
        torch.clamp(
            reco_jet["energy"] ** 2 - (pt_jet * torch.cosh(eta_jet)) ** 2, min=0.0
        )
    ).clamp_min(eps)

    d_phi = phi_tau - phi_jet
    # Same five components, same order, as the daughter regression target.
    kin5 = torch.stack(
        [
            torch.log(torch.clamp(pt_tau / pt_jet.clamp_min(eps), min=eps)),
            eta_tau - eta_jet,
            torch.sin(d_phi),
            torch.cos(d_phi),
            torch.log(torch.clamp(mass_tau / mass_jet, min=eps)),
        ],
        dim=-1,
    )

    return {
        "charge": (charge * valid.to(charge.dtype)).sum(-1),
        "kinematics": kin5,
        "n_daughters": valid.sum(-1),
        # Kept dense; the decay mode is derived from these at logging time by
        # the evaluation code rather than reimplemented per batch.
        "pdg": pdg,
        "valid": valid,
    }


class SetToSetAccumulator:
    """
    Collects what the evaluators need over a validation epoch.

    Per-jet scalars only, and only up to `max_jets`: a validation split can be
    millions of jets, the plots converge long before that, and the evaluators
    are not cheap. Everything is moved to CPU on arrival so no GPU memory is
    held between batches.
    """

    P4_FIELDS = ("pt", "eta", "phi", "energy")

    def __init__(self, max_jets: int = 200_000):
        self.max_jets = max_jets
        self.reset()

    def reset(self) -> None:
        self.keys = ("charge", "kinematics", "n_daughters", "pdg", "valid")
        self.pred = {key: [] for key in self.keys}
        self.true = {key: [] for key in self.keys}
        self.is_tau: list[np.ndarray] = []
        self.tau_score: list[np.ndarray] = []
        self.p4 = {
            name: {field: [] for field in self.P4_FIELDS}
            for name in ("gen_jet_tau_p4s", "reco_jet_p4s", "gen_jet_p4s")
        }
        self.n_jets = 0

    def update(self, pred, true, is_tau, tau_score, p4s) -> None:
        if self.n_jets >= self.max_jets:
            return
        room = self.max_jets - self.n_jets
        for key in self.pred:
            self.pred[key].append(pred[key][:room].detach().cpu().numpy())
            self.true[key].append(true[key][:room].detach().cpu().numpy())
        self.is_tau.append(is_tau[:room].detach().cpu().numpy())
        if tau_score is not None:
            self.tau_score.append(tau_score[:room].detach().cpu().numpy())
        for name, p4 in p4s.items():
            for field in self.P4_FIELDS:
                self.p4[name][field].append(p4[field][:room].detach().cpu().numpy())
        self.n_jets += int(min(room, len(is_tau)))

    def stacked(self):
        """MultiParTau-shaped `targets` and `predictions` dicts, plus the p4s."""
        if self.n_jets == 0:
            return None
        targets = {"is_tau": np.concatenate(self.is_tau)}
        predictions = {}
        for key in ("charge", "kinematics", "n_daughters"):
            targets[key] = np.concatenate(self.true[key])
            predictions[key] = np.concatenate(self.pred[key])
        # Decay mode from the selected daughters, through the evaluation code.
        for side, store in (("true", targets), ("pred", predictions)):
            derived = decay_modes(
                np.concatenate(getattr(self, side)["pdg"]),
                np.concatenate(getattr(self, side)["valid"]),
            )
            store["decay_mode"] = derived["decay_mode"]
            store["n_charged"] = derived["n_charged"]
            store["decay_mode_category"] = derived["category"]
        if self.tau_score:
            predictions["is_tau"] = np.concatenate(self.tau_score)
        p4s = {
            name: ak.Array(
                {field: np.concatenate(values) for field, values in fields.items()}
            )
            for name, fields in self.p4.items()
        }
        return targets, predictions, p4s


def log_set_to_set_metrics(accumulator, tb_logger, cfg, current_epoch, dataset="val"):
    """
    Draw the MultiParTau evaluator plots for the reconstructed tau.

    Each block is guarded independently: one evaluator failing on a degenerate
    epoch (no signal jets, a metric config a sample cannot fill) must not cost
    the others, and none of them may cost the training run.
    """
    stacked = accumulator.stacked()
    if stacked is None or tb_logger is None:
        return {}
    targets, predictions, p4s = stacked
    signal = targets["is_tau"] == 1
    scalars = {"n_jets_evaluated": float(len(signal))}

    # ---- tagging: the jet-level head, evaluated exactly as MultiParTau does --
    if "is_tau" in predictions and 0 < signal.sum() < len(signal):
        try:
            tagging_logging.log_all_tagging_metrics(
                targets=targets,
                predictions=predictions,
                gen_jet_p4s=p4s["gen_jet_p4s"],
                gen_jet_tau_p4s=p4s["gen_jet_tau_p4s"],
                reco_jet_p4s=p4s["reco_jet_p4s"],
                cfg=cfg,
                tb_logger=tb_logger,
                current_epoch=current_epoch,
                dataset=dataset,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"tagging logging failed: {exc}")

    # ---- charge: the SUM of the daughters' charges --------------------------
    # HardLabelChargeIdEvaluator rather than ChargeIdEvaluator: this model emits
    # an integer charge, not a score to threshold, and that class exists for
    # exactly this case. Charges outside {-1, 0, +1} are physical failures of
    # the reconstruction (too many charged daughters), so they are counted
    # rather than clipped away.
    if signal.sum() > 0:
        try:
            pred_charge = predictions["charge"][signal]
            true_charge = targets["charge"][signal]
            scalars["charge_accuracy"] = float((pred_charge == true_charge).mean())
            # A tau is charge +-1. Anything else -- 0 from an even number of
            # charged daughters just as much as +-2 from too many -- is a failed
            # reconstruction, so both count here rather than only the overflow.
            scalars["charge_out_of_range_frac"] = float(
                (np.abs(pred_charge) != 1).mean()
            )
            # The evaluator's hard labels are +1 / -1 / "neither". Mapping every
            # non-unit charge to 0 puts those jets in the denominator of the
            # efficiency without claiming a sign for them, which is what a
            # failed charge reconstruction should do to the number.
            as_label = lambda q: np.where(np.abs(q) == 1, q, 0)
            evaluator = c.HardLabelChargeIdEvaluator(
                predicted=as_label(pred_charge),
                truth=as_label(true_charge),
                gen_jet_tau_p4s=p4s["gen_jet_tau_p4s"][signal],
                reco_jet_p4s=p4s["reco_jet_p4s"][signal],
                cfg=cfg,
                sample="all",
                algorithm="all",
            )
            confusion_plot = c.ConfusionMatrixPlot()
            confusion_plot.add_data(evaluator)
            tb_logger.add_figure(
                f"{dataset}_charge_id/confusion_matrix",
                confusion_plot.fig,
                current_epoch,
            )
            plt.close(confusion_plot.fig)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"charge logging failed: {exc}")

    # ---- decay mode: the HPS confusion matrix, as in STS_model_eval_HPS ------
    if signal.sum() > 0:
        try:
            truth = np.asarray(targets["decay_mode"])[signal]
            predicted = np.asarray(predictions["decay_mode"])[signal]
            truth_row = np.asarray(targets["decay_mode_category"])[signal]
            pred_row = np.asarray(predictions["decay_mode_category"])[signal]
            n_charged_pred = np.asarray(predictions["n_charged"])[signal]

            even_prong = (n_charged_pred > 0) & (n_charged_pred % 2 == 0)
            scalars["decay_mode_accuracy"] = float((predicted == truth).mean())
            scalars["decay_mode_even_prong_frac"] = float(even_prong.mean())
            # Predictions no tau can produce, or that reconstruct nothing
            # classifiable: even prongs, off-grid, or upstream `rare` (which for
            # a prediction mostly means zero or more than five daughters).
            scalars["decay_mode_unphysical_frac"] = float(
                (even_prong | (pred_row < 0) | (pred_row == _RARE_ROW)).mean()
            )

            # A truth class without a row would be dropped silently by the
            # matrix, so say so instead.
            unknown_truth = truth_row < 0
            if unknown_truth.any():
                warnings.warn(
                    f"{unknown_truth.mean():.2%} of true decay modes have no row "
                    f"and are not shown: "
                    f"{sorted(set(truth[unknown_truth].tolist()))[:10]}"
                )

            matrix = _rectangular_confusion(truth_row, pred_row, n_charged_pred)
            figure, _ = hps.visualize_hps_confusion_matrix(
                histogram=matrix,
                categories=DECAY_MODE_LABELS,
                predicted_categories=DECAY_MODE_LABELS + EXTRA_PREDICTED_LABELS,
            )
            tb_logger.add_figure(
                f"{dataset}_decay_mode/confusion_matrix", figure, current_epoch
            )
            plt.close(figure)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"decay mode logging failed: {exc}")

    # ---- kinematics of the summed tau ---------------------------------------
    if signal.sum() > 0:
        try:
            kinematics_logging.log_all_kinematics_metrics(
                targets=targets,
                predictions=predictions,
                reco_jet_p4s=p4s["reco_jet_p4s"],
                gen_jet_tau_p4s=p4s["gen_jet_tau_p4s"],
                cfg=cfg,
                tb_logger=tb_logger,
                current_epoch=current_epoch,
                dataset=dataset,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"kinematics logging failed: {exc}")

    # ---- how many daughters the model produced ------------------------------
    # Not a MultiParTau quantity: it only exists for a set model, and it is the
    # first thing to look at when the kinematics plots go strange, since a tau
    # summed from the wrong number of daughters is wrong before any regression
    # error enters.
    if signal.sum() > 0:
        scalars["n_daughters_pred_mean"] = float(
            predictions["n_daughters"][signal].mean()
        )
        scalars["n_daughters_true_mean"] = float(targets["n_daughters"][signal].mean())
    if tb_logger is not None and scalars:
        log_metrics_dict(tb_logger, scalars, f"{dataset}_set_to_set", current_epoch)
    return scalars


# ---------------------------------------------------------------------------
# Objectness threshold
# ---------------------------------------------------------------------------
# The threshold that selects predicted daughters is not a constant of the model:
# it moves as the objectness head sharpens, so a value fixed in the config is
# right at best for the epoch it was tuned on. The inference notebook scans it
# and takes the best F1; the same is done here, on TRAIN jets, so validation is
# never used to pick the operating point it is then judged at.


def _assign(p_eta, p_phi, t_eta, t_phi, max_dr):
    """Hungarian match of one jet's predicted to true daughters by dR."""
    if p_eta.size == 0 or t_eta.size == 0:
        return 0
    deta = np.abs(p_eta[:, None] - t_eta[None, :])
    dphi = np.abs(p_phi[:, None] - t_phi[None, :])
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    dr = np.sqrt(deta**2 + dphi**2)
    rows, cols = linear_sum_assignment(dr)
    return int((dr[rows, cols] <= max_dr).sum())


def _decay_mode_correct(n_charged_pred, n_neutral_pred, n_charged_true, n_neutral_true):
    """
    Per-jet: does the predicted daughter set map to the true decay mode?

    Decay mode is 5 * (n_charged - 1) + min(n_neutral, 4) in ml-tau-data, so
    equality of the two counts (neutrals clamped at 4) is equality of the mode.
    Done on counts rather than through the classifier so the scan stays a few
    numpy operations per threshold.
    """
    return (n_charged_pred == n_charged_true) & (
        np.minimum(n_neutral_pred, 4) == np.minimum(n_neutral_true, 4)
    )


def _mean_f1(scores, threshold, pred_eta, pred_phi, true_eta, true_phi, true_valid, max_dr):
    """Mean per-jet F1 of dR-matched daughters at one threshold."""
    n_true = true_valid.sum(axis=1)
    keep = scores >= threshold
    n_pred = keep.sum(axis=1)
    matched = np.zeros(len(n_true), dtype=np.int64)
    for i in np.nonzero((n_pred > 0) & (n_true > 0))[0]:
        matched[i] = _assign(
            pred_eta[i][keep[i]], pred_phi[i][keep[i]],
            true_eta[i][true_valid[i]], true_phi[i][true_valid[i]], max_dr,
        )
    efficiency = np.divide(matched, n_true, out=np.zeros(len(n_true)), where=n_true > 0)
    purity = np.divide(matched, n_pred, out=np.zeros(len(n_true), dtype=float), where=n_pred > 0)
    denominator = efficiency + purity
    f1 = np.divide(2 * efficiency * purity, denominator, out=np.zeros_like(denominator),
                   where=denominator > 0)
    return float(f1.mean())


def scan_threshold(
    scores, pred_eta, pred_phi, pred_charged, true_eta, true_phi, true_valid, true_charged,
    thresholds=None, max_dr: float = 0.4, objective: str = "decay_mode",
    fraction_logits=None, parent_p4=None,
):
    """
    Objectness threshold maximising `objective` over true tau jets.

    objective = "decay_mode": fraction of jets whose predicted charged and
    neutral daughter counts both equal the truth, i.e. decay-mode accuracy.
    That is the deliverable, and it is all-or-nothing in the prong count: one
    spurious daughter changes the mode outright.

    objective = "f1": mean per-jet F1 of dR-matched daughters, the older figure
    of merit. It treats a spurious or missing daughter as a partial loss, so
    its optimum sits below the decay-mode one; on the 20k-step baseline it
    picked 0.75 where 0.85 was 1.7 points of accuracy better.

    Only jets with at least one true daughter take part -- background has no
    decay mode and would only dilute either figure. The F1 at the chosen
    threshold is returned alongside whichever objective was used, for
    continuity of the logged curves.

    Returns (best_threshold, {threshold: {"decay_mode_accuracy": .., "f1": ..}}),
    where "f1" is filled only at the best threshold unless it was the
    objective (the assignment is the expensive part of the scan).
    """
    if objective not in ("decay_mode", "f1"):
        raise ValueError(f"threshold_scan.objective must be 'decay_mode' or 'f1', got {objective!r}")
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.9, 13)
    signal = true_valid.any(axis=1)
    if not signal.any():
        return None, {}
    scores, pred_eta, pred_phi, pred_charged = (
        scores[signal], pred_eta[signal], pred_phi[signal], pred_charged[signal]
    )
    true_eta, true_phi, true_valid, true_charged = (
        true_eta[signal], true_phi[signal], true_valid[signal], true_charged[signal]
    )
    if fraction_logits is not None:
        fraction_logits = torch.as_tensor(fraction_logits[signal])
        parent_p4 = torch.as_tensor(parent_p4[signal])

    def directions(threshold):
        if fraction_logits is None:
            return pred_eta, pred_phi
        decoded = decode_fractions(fraction_logits, parent_p4, torch.as_tensor(scores >= threshold))
        coordinates = momentum_coordinates(decoded)
        return coordinates[..., 1].numpy(), coordinates[..., 2].numpy()

    n_charged_true = (true_valid & true_charged).sum(axis=1)
    n_neutral_true = (true_valid & ~true_charged).sum(axis=1)

    by_threshold = {}
    for threshold in thresholds:
        keep = scores >= threshold
        n_charged_pred = (keep & pred_charged).sum(axis=1)
        n_neutral_pred = (keep & ~pred_charged).sum(axis=1)
        entry = {
            "decay_mode_accuracy": float(
                _decay_mode_correct(n_charged_pred, n_neutral_pred, n_charged_true, n_neutral_true).mean()
            ),
        }
        if objective == "f1":
            current_eta, current_phi = directions(threshold)
            entry["f1"] = _mean_f1(scores, threshold, current_eta, current_phi, true_eta, true_phi, true_valid, max_dr)
        by_threshold[float(threshold)] = entry

    key = "decay_mode_accuracy" if objective == "decay_mode" else "f1"
    best = max(by_threshold, key=lambda t: by_threshold[t][key])
    if "f1" not in by_threshold[best]:
        current_eta, current_phi = directions(best)
        by_threshold[best]["f1"] = _mean_f1(
            scores, best, current_eta, current_phi, true_eta, true_phi, true_valid, max_dr
        )
    return best, by_threshold


class ThresholdCalibrationBuffer:
    """
    Holds batches for an objectness threshold scan.

    Only what the two objectives need is kept: objectness scores, directions
    for the dR matching, and whether each predicted / true daughter is of a
    charged class for the decay-mode count.
    """

    def __init__(self, max_jets: int = 100_000):
        self.max_jets = max_jets
        self.batches: list[dict] = []

    def reset(self) -> None:
        self.batches.clear()

    def add(self, scores, pred_eta, pred_phi, pred_charged, true_eta, true_phi, true_valid, true_charged,
            fraction_logits=None, parent_p4=None) -> None:
        self.batches.append({
            "scores": scores.detach().float().cpu().numpy(),
            "pred_eta": pred_eta.detach().float().cpu().numpy(),
            "pred_phi": pred_phi.detach().float().cpu().numpy(),
            "pred_charged": pred_charged.detach().bool().cpu().numpy(),
            "true_eta": true_eta.detach().float().cpu().numpy(),
            "true_phi": true_phi.detach().float().cpu().numpy(),
            "true_valid": true_valid.detach().bool().cpu().numpy(),
            "true_charged": true_charged.detach().bool().cpu().numpy(),
        })
        if fraction_logits is not None:
            self.batches[-1]["fraction_logits"] = fraction_logits.detach().float().cpu().numpy()
            self.batches[-1]["parent_p4"] = parent_p4.detach().float().cpu().numpy()
        while sum(len(b["scores"]) for b in self.batches) > self.max_jets and len(self.batches) > 1:
            self.batches.pop(0)

    def scan(self, thresholds=None, max_dr: float = 0.4, objective: str = "decay_mode"):
        if not self.batches:
            return None, {}
        merged = {
            key: np.concatenate([b[key] for b in self.batches])
            for key in self.batches[0]
        }
        return scan_threshold(
            merged["scores"], merged["pred_eta"], merged["pred_phi"], merged["pred_charged"],
            merged["true_eta"], merged["true_phi"], merged["true_valid"], merged["true_charged"],
            thresholds=thresholds, max_dr=max_dr, objective=objective,
            fraction_logits=merged.get("fraction_logits"), parent_p4=merged.get("parent_p4"),
        )
