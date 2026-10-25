"""Paper-style evaluation of a ParTauDETR checkpoint, written as one JSON.

Computes the object-level numbers arXiv 2606.18460 reports for MultiParTau --
visible-pT response resolution as IQR/median, median DeltaR(pred, gen),
decay-mode accuracy/confusion from the predicted daughter species, tau charge
from the summed daughter charges -- plus daughter-level efficiency/purity/F1
and matched-daughter residuals on signal jets, and the tau-ID performance of
the tagging head (ROC, AUC, misID at fixed signal efficiencies) on signal vs
background jets. One JSON per checkpoint, so runs compare number by number;
the notebooks (ParTauDETR_inference/_evaluation) give the plots.

The training config is read from the checkpoint itself (the module saves its
model/dataset/training subtrees), so the inputs are built exactly as in
training: the same feature definitions, the same input scaler (refused if it
was fitted on differently defined features) and the same mass decoding. Only
the data directory can be redirected (--data-dir).

Runs on CPU (evaluation does not belong on a GPU allocation):
  ./run.sh python3 mltau/scripts/ParTauDETR_eval.py <ckpt> --out metrics.json
"""

import argparse
import glob
import json
import os

import awkward as ak
import numpy as np
import torch

from mltau.models.ParTauDETR_module import ParTauDETRModule
from mltau.tools.evaluation.decode_ParTauDETR import (
    get_predicted_particles,
    get_true_particles,
    match_particles,
    sum_p4_components,
    tau_scores,
)
from mltau.tools.io.ParTauDETR_dataloader import ParticleTransformerDETRDataset
from mltau.tools.evaluation import set_to_set_models as s2s_eval

PT_BINS = ((15, 25), (25, 35), (35, 50), (40, 45))
ROC_EFFICIENCIES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def iqr_resolution(response: np.ndarray) -> dict:
    """Paper-convention resolution: IQR of the response normalised to the
    median (same as kinematics.RegressionEvaluator: resolution/response)."""
    response = response[np.isfinite(response)]
    if response.size == 0:
        return {"median": float("nan"), "iqr_over_med": float("nan"), "n": 0}
    q25, q50, q75 = np.percentile(response, [25, 50, 75])
    return {
        "median": float(q50),
        "iqr_over_med": float((q75 - q25) / q50) if q50 else float("nan"),
        "n": int(response.size),
    }


def to_device(batch, device):
    return tuple(
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        if isinstance(b, dict)
        else (b.to(device) if torch.is_tensor(b) else b)
        for b in batch
    )


def read_files(pattern: str, max_files: int | None):
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    return files


def run_model(model, ds, files, chunk, device, keep_reconstruction: bool):
    """
    Forward every jet of `files` through the model, chunk by chunk.

    Always keeps p(tau) and the is_tau label per jet. With keep_reconstruction
    the daughter-level outputs, targets and reco-jet references are kept too
    (signal sample); for the background sample only the tagger matters, and
    keeping the set outputs of millions of jets would only cost memory.
    """
    out_keys = (
        "pred_logits",
        "pred_kinematics",
        "pred_charge_logits",
        "pred_meson_class_logits",
    )
    tgt_keys = (
        "particles_kinematics",
        "particles_charge_ohe",
        "particles_meson_class_ohe",
        "particles_mask",
    )
    outs = {k: [] for k in out_keys}
    tgts = {k: [] for k in tgt_keys}
    refs = {k: [] for k in ("pt", "eta", "phi", "energy")}
    scores, labels = [], []
    n_jets = 0
    with torch.no_grad():
        for path in files:
            data = ak.from_parquet(path)
            for start in range(0, len(data), chunk):
                batch = to_device(ds.build_tensors(data[start : start + chunk]), device)
                outputs, targets, _w, _, _ = model.forward(batch)
                score = tau_scores(outputs)
                if score is not None:
                    scores.append(score.cpu())
                labels.append(targets["is_tau"].float().cpu())
                if keep_reconstruction:
                    for k in out_keys:
                        outs[k].append(outputs[k].float().cpu())
                    for k in tgt_keys:
                        tgts[k].append(targets[k].cpu())
                    reco_jet_p4s = batch[6]
                    for k in refs:
                        refs[k].append(torch.as_tensor(reco_jet_p4s[k]).cpu())
            n_jets += len(data)
            print(f"  {os.path.basename(path)}: {n_jets:,} jets", flush=True)
    result = {
        "tau_score": torch.cat(scores).numpy() if scores else None,
        "is_tau": torch.cat(labels).numpy().astype(bool),
        "n_jets": n_jets,
    }
    if keep_reconstruction:
        result["outputs"] = {k: torch.cat(v) for k, v in outs.items()}
        result["targets"] = {k: torch.cat(v) for k, v in tgts.items()}
        result["reco_jet_p4s"] = {k: torch.cat(v) for k, v in refs.items()}
    return result


def tau_id_metrics(sig_scores: np.ndarray, bkg_scores: np.ndarray) -> dict:
    """
    ROC of the tagging head: signal efficiency vs background misID rate.

    Unweighted jet counts (the training's cls_weight decorrelation weights are
    not applied), signal = genuine taus, background = jets without one. The
    misID at a fixed efficiency takes the score threshold at that signal
    quantile; its statistical precision is ~1/sqrt(misID * n_bkg).
    """
    thresholds = np.quantile(sig_scores, 1.0 - np.asarray(ROC_EFFICIENCIES))
    at_efficiency = {}
    for eff, thr in zip(ROC_EFFICIENCIES, thresholds):
        n_pass = int((bkg_scores >= thr).sum())
        at_efficiency[f"{eff:.2f}"] = {
            "threshold": float(thr),
            "misid": n_pass / len(bkg_scores),
            "n_bkg_pass": n_pass,
        }

    # AUC = P(score_sig > score_bkg), from ranks (ties count half).
    scores = np.concatenate([sig_scores, bkg_scores])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # average ranks of tied values
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    n_s, n_b = len(sig_scores), len(bkg_scores)
    auc = (ranks[:n_s].sum() - n_s * (n_s + 1) / 2.0) / (n_s * n_b)

    # Compact curve for plotting: misID at 1%-spaced efficiencies.
    effs = np.linspace(0.01, 0.99, 99)
    curve_thr = np.quantile(sig_scores, 1.0 - effs)
    bkg_sorted = np.sort(bkg_scores)
    curve_misid = 1.0 - np.searchsorted(bkg_sorted, curve_thr, side="left") / n_b
    return {
        "n_signal": int(n_s),
        "n_background": int(n_b),
        "auc": float(auc),
        "at_efficiency": at_efficiency,
        "roc_curve": {
            "efficiency": effs.round(4).tolist(),
            "misid": curve_misid.tolist(),
        },
    }


def reconstruction_metrics(sig, threshold: float, mass_from_class: bool, tau_daughter_pdg_ids) -> dict:
    """Object- and daughter-level reconstruction on the genuine tau jets."""
    keep = torch.as_tensor(sig["is_tau"])
    outputs = {k: v[keep] for k, v in sig["outputs"].items()}
    targets = {k: v[keep] for k, v in sig["targets"].items()}
    reco_jet_p4s = {k: v[keep] for k, v in sig["reco_jet_p4s"].items()}

    true_p4, true_charge, true_meson_class = get_true_particles(targets, reco_jet_p4s)
    pred_p4, pred_charge, pred_meson_class = get_predicted_particles(
        outputs, reco_jet_p4s, obj_cls_trsh=threshold, mass_from_class=mass_from_class,
        tau_daughter_pdg_ids=tau_daughter_pdg_ids,
    )

    # ---- object level: visible tau from summed daughters ----
    pred_tau = sum_p4_components(pred_p4)
    true_tau = sum_p4_components(true_p4)
    pred_pt = ak.to_numpy(pred_tau.pt)
    true_pt = ak.to_numpy(true_tau.pt)
    has_pred = ak.to_numpy(ak.num(pred_p4)) > 0
    has_true = ak.to_numpy(ak.num(true_p4)) > 0
    both = has_pred & has_true

    response = np.where(both, pred_pt / np.where(true_pt > 0, true_pt, np.nan), np.nan)
    results = {
        "n_tau_jets": int(keep.sum()),
        "frac_no_prediction": float((~has_pred & has_true).mean()),
        "pt_response": iqr_resolution(response),
        "pt_response_binned": {
            f"{lo}-{hi}": iqr_resolution(response[both & (true_pt >= lo) & (true_pt < hi)])
            for lo, hi in PT_BINS
        },
    }
    deta = ak.to_numpy(pred_tau.eta) - ak.to_numpy(true_tau.eta)
    dphi_raw = ak.to_numpy(pred_tau.phi) - ak.to_numpy(true_tau.phi)
    dphi = np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw))
    dr = np.sqrt(deta**2 + dphi**2)[both]
    results["dR_median"] = float(np.median(dr)) if dr.size else float("nan")

    pred_pdg = s2s_eval.meson_class_to_pdg(pred_meson_class, tau_daughter_pdg_ids)
    true_pdg = s2s_eval.meson_class_to_pdg(true_meson_class, tau_daughter_pdg_ids)
    pred_ch = ak.to_numpy(ak.sum(pred_pdg == s2s_eval.CHARGED_HADRON_PDG, axis=1))
    pred_mode = ak.to_numpy(s2s_eval.get_decay_mode(pred_pdg))
    true_mode = ak.to_numpy(s2s_eval.get_decay_mode(true_pdg))
    valid = has_true
    results["decay_mode_accuracy"] = float((pred_mode[valid] == true_mode[valid]).mean())
    results["decay_mode_unphysical_frac"] = float(
        (
            (pred_mode[valid] == s2s_eval.RARE_DECAY_MODE_EXT)
            | ((pred_ch[valid] > 0) & (pred_ch[valid] % 2 == 0))
        ).mean()
    )
    main_modes = [0, 1, 2, 10, 11, s2s_eval.ELECTRON_DECAY_MODE, s2s_eval.MUON_DECAY_MODE]
    results["decay_mode_labels"] = {
        str(s2s_eval.ELECTRON_DECAY_MODE): "electron",
        str(s2s_eval.MUON_DECAY_MODE): "muon",
    }
    confusion = {}
    for tm in main_modes:
        sel = true_mode == tm
        if sel.sum() == 0:
            continue
        row = {str(pm): float((pred_mode[sel] == pm).mean()) for pm in main_modes}
        row["other"] = float(1.0 - sum(row.values()))
        row["n"] = int(sel.sum())
        confusion[str(tm)] = row
    results["decay_mode_confusion"] = confusion

    # ---- tau charge from summed daughter charges ----
    pred_q = ak.to_numpy(ak.sum(pred_charge, axis=1))
    true_q = ak.to_numpy(ak.sum(true_charge, axis=1))
    qsel = np.abs(true_q) == 1
    results["charge_accuracy"] = float((pred_q[qsel] == true_q[qsel]).mean())

    # ---- daughter level ----
    matches = match_particles(
        pred_p4, true_p4, pred_charge, true_charge,
        pred_meson_class, true_meson_class, max_dr=0.4,
    )
    n_matched = ak.to_numpy(ak.num(matches.pred_idx)).astype(float)
    n_pred = ak.to_numpy(ak.num(pred_p4)).astype(float)
    n_true = ak.to_numpy(ak.num(true_p4)).astype(float)
    eff = np.divide(n_matched, n_true, out=np.zeros_like(n_true), where=n_true > 0)
    pur = np.divide(n_matched, n_pred, out=np.zeros_like(n_pred), where=n_pred > 0)
    denom = eff + pur
    f1 = np.divide(2 * eff * pur, denom, out=np.zeros_like(denom), where=denom > 0)
    daughter = {
        "efficiency_mean": float(eff[n_true > 0].mean()),
        "purity_mean": float(pur[n_true > 0].mean()),
        "f1_mean": float(f1[n_true > 0].mean()),
    }
    matched_pred_pt = ak.to_numpy(ak.flatten(pred_p4[matches.pred_idx].pt))
    matched_true_pt = ak.to_numpy(ak.flatten(true_p4[matches.true_idx].pt))
    daughter["pt_response"] = iqr_resolution(
        matched_pred_pt / np.where(matched_true_pt > 0, matched_true_pt, np.nan)
    )
    matched_pred_q = ak.to_numpy(ak.flatten(pred_charge[matches.pred_idx]))
    matched_true_q = ak.to_numpy(ak.flatten(true_charge[matches.true_idx]))
    daughter["charge_accuracy_matched"] = float((matched_pred_q == matched_true_q).mean())
    matched_pred_cls = ak.to_numpy(ak.flatten(pred_meson_class[matches.pred_idx]))
    matched_true_cls = ak.to_numpy(ak.flatten(true_meson_class[matches.true_idx]))
    daughter["meson_class_accuracy_matched"] = float(
        (matched_pred_cls == matched_true_cls).mean()
    )
    results["daughter"] = daughter
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="test-file directory; default: the checkpoint's dataset.data_dir",
    )
    ap.add_argument("--signal-sample", default="z")
    ap.add_argument("--background-sample", default="qq")
    ap.add_argument(
        "--signal-files", type=int, default=1,
        help="signal test files to read (~100k jets each); 0 = all",
    )
    ap.add_argument(
        "--background-files", type=int, default=4,
        help="background test files for tau-ID (~100k jets each); 0 = all, "
        "-1 = skip tau-ID",
    )
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="objectness threshold; default: the calibrated value in the checkpoint",
    )
    ap.add_argument(
        "--mass-from-class", choices=("auto", "yes", "no"), default="auto",
        help="decode daughter masses from the meson class; auto = as trained",
    )
    args = ap.parse_args()
    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = ParTauDETRModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    cfg = model.cfg
    data_dir = os.path.expanduser(args.data_dir or str(cfg.dataset.data_dir))

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(getattr(model, "score_threshold_calibrated", 0.5))
    )
    mass_from_class = (
        bool(cfg.model.detr.loss.get("mass_from_meson_class", False))
        if args.mass_from_class == "auto"
        else args.mass_from_class == "yes"
    )
    print(
        f"objectness threshold {threshold:.4f}, mass_from_class {mass_from_class}, "
        f"data {data_dir}",
        flush=True,
    )

    # Builds the inputs exactly as in training, scaler included (and refused
    # if fitted on differently defined features).
    ds = ParticleTransformerDETRDataset.for_arrays(cfg)

    sig_files = read_files(
        os.path.join(data_dir, f"{args.signal_sample}_test*.parquet"),
        args.signal_files or None,
    )
    print(f"signal: {len(sig_files)} file(s)", flush=True)
    sig = run_model(model, ds, sig_files, args.chunk, device, keep_reconstruction=True)

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "data_dir": data_dir,
        "threshold": threshold,
        "mass_from_class": mass_from_class,
        "signal_files": [os.path.basename(f) for f in sig_files],
        "n_signal_jets": sig["n_jets"],
        "reconstruction": reconstruction_metrics(
            sig, threshold, mass_from_class, cfg.dataset.tau_daughter_pdg_ids
        ),
    }

    if args.background_files >= 0 and sig["tau_score"] is not None:
        bkg_files = read_files(
            os.path.join(data_dir, f"{args.background_sample}_test*.parquet"),
            args.background_files or None,
        )
        print(f"background: {len(bkg_files)} file(s)", flush=True)
        bkg = run_model(model, ds, bkg_files, args.chunk, device, keep_reconstruction=False)
        sig_scores = sig["tau_score"][sig["is_tau"]]
        bkg_scores = np.concatenate(
            [bkg["tau_score"][~bkg["is_tau"]], sig["tau_score"][~sig["is_tau"]]]
        )
        results["background_files"] = [os.path.basename(f) for f in bkg_files]
        results["tau_id"] = tau_id_metrics(sig_scores, bkg_scores)
    elif sig["tau_score"] is None:
        print("model has no tauID head: tau-ID skipped", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    summary = {k: v for k, v in results.items() if k != "tau_id"}
    if "tau_id" in results:
        summary["tau_id"] = {
            k: v for k, v in results["tau_id"].items() if k != "roc_curve"
        }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
