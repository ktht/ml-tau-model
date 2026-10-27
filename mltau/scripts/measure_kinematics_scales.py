"""
Measure the residual scales for the kinematics regression loss (TauLoss).

TauLoss divides each kinematics residual by a scale before the Huber, so every
component enters the loss as "error in units of the spread of its own target"
(see mltau/tools/losses.py, _compute_kinematics_loss_per_sample). The scales
are properties of the data, not tuning knobs: this script reads the training
targets exactly as the dataloader produces them and prints their spreads, plus
a block to paste into tau_loss.kinematics_scales.

    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model ParT
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model DETR
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model DETR --daughters single
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model DETR --daughters multiple
    # extra Hydra overrides go after --, e.g. another production:
    ./run.sh python3 mltau/scripts/measure_kinematics_scales.py --model ParT -- \\
        dataset.data_dir=/scratch/persistent/laurits/ml-tau/<production>/

Targets (same 5-vector for both models, built in the dataloaders):
    [log(pt / pt_jet), eta - eta_jet, sin(dphi), cos(dphi), log(m / m_jet)]
  ParT:  the visible tau of signal jets (is_tau == 1), cfg main.yaml.
  DETR:  supervised decoder daughters, cfg main_ParTauDETR.yaml.
  DETR --daughters single: the sole visible daughter (the full visible tau),
         using the direct-head targets for DM0/electron/muon decays. When the
         loader does not export these targets, the measurement adapter derives
         them from the full truth visible tau and reconstructed jet four-momenta.
The two differ by an order of magnitude in the angular spreads, so they need
separate scales.

--daughters single selects decays with exactly one original visible truth
daughter; multiple selects more than one; all (default) does not filter.
Counts are taken before cuts, meson filtering and padding, not from the number
of charged prongs or surviving decoder targets. This selection intersects the
automatically selected head's supervision mask; it does not change which
classes train it. For DETR, single selects direct-head targets, while multiple
and all select decoder targets. All preserves the default decoder measurement;
it does not combine the two heads' targets into a shared scale estimate.
DETR single measurements print model.direct_tau_loss.kinematics_scales.

Measurement defaults to single-process loading to avoid worker tensor/file-
descriptor transfer failures. Training settings are unchanged. To opt into
workers, pass -- training.dataloader.num_dataloader_workers=<n>.

The phi component of the loss is the chord between the predicted and the true
(sin dphi, cos dphi). For the trivial prediction "along the jet axis" the chord
is 2|sin(dphi/2)| ~ |dphi|, so its natural scale is the spread of dphi itself;
the script also prints sigma(sin dphi) and the RMS chord, since earlier configs
quoted those.

Several spread estimators are printed per component because the targets have
tails (the log ratios are clamped at +-5 by the dataloader): std, a robust
sigma (half the 16-84 % interquantile range, equal to the std for a Gaussian)
and the RMS. The YAML block uses the estimator chosen with --estimator.
"""

import argparse
import inspect
import os
import sys

import awkward as ak
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

# Checkpoints/configs carry OmegaConf objects; harmless here, mirrors train.py.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_DIR)
LOG_CLAMP = 5.0  # the dataloaders clip the log-ratio targets to +-LOG_CLAMP

from mltau.tools.io.ParT_dataloader import ParTDataModule, ParticleTransformerDataset
from mltau.tools.io.ParTauDETR_dataloader import (
    ParTauDETRDataModule,
    ParticleTransformerDETRDataset,
)


def _with_truth_daughter_counts(data, batch):
    """Attach original multiplicity before the dataset slices or shuffles rows."""
    column = "gen_jet_tau_vis_daughter_pdgs"
    if column not in data.fields:
        raise ValueError(f"Daughter-count measurement requires parquet column {column!r}.")
    counts = torch.from_numpy(ak.to_numpy(ak.num(data[column], axis=-1)).astype(np.int64))
    targets = {**batch[2], "num_tau_daughters": counts}
    return (*batch[:2], targets, *batch[3:])


def _with_direct_tau_targets(data, batch):
    """Supply missing direct targets using the training parameterization."""
    if {"is_direct_tau", "direct_kinematics"}.issubset(batch[2]):
        return batch

    daughter_pdgs = abs(data.gen_jet_tau_vis_daughter_pdgs)
    electron_count = ak.to_numpy(ak.sum(daughter_pdgs == 11, axis=-1))
    muon_count = ak.to_numpy(ak.sum(daughter_pdgs == 13, axis=-1))
    if np.any(electron_count + muon_count > 1):
        raise ValueError("Expected at most one electron or muon among truth tau daughters.")
    direct_mask = (
        (ak.to_numpy(data.gen_jet_tau_decaymode) == 0)
        | (electron_count == 1)
        | (muon_count == 1)
    )
    truth = {name: np.asarray(batch[5][name]) for name in ("pt", "eta", "phi", "energy")}
    reco = {name: np.asarray(batch[6][name]) for name in ("pt", "eta", "phi", "energy")}
    truth_mass = np.sqrt(np.maximum(
        truth["energy"] ** 2 - (truth["pt"] * np.cosh(truth["eta"])) ** 2, 0.0
    ))
    reco_mass = np.sqrt(np.maximum(
        reco["energy"] ** 2 - (reco["pt"] * np.cosh(reco["eta"])) ** 2, 0.0
    ))
    epsilon = 1e-6
    delta_phi = truth["phi"] - reco["phi"]
    kinematics = np.stack([
        np.clip(np.log(np.maximum(truth["pt"] / np.maximum(reco["pt"], epsilon), epsilon)), -LOG_CLAMP, LOG_CLAMP),
        truth["eta"] - reco["eta"],
        np.sin(delta_phi),
        np.cos(delta_phi),
        np.clip(np.log(np.maximum(truth_mass / np.maximum(reco_mass, epsilon), epsilon)), -LOG_CLAMP, LOG_CLAMP),
    ], axis=-1).astype(np.float32)
    kinematics[~direct_mask] = 0.0
    targets = {
        **batch[2],
        "is_direct_tau": torch.from_numpy(direct_mask),
        "direct_kinematics": torch.from_numpy(kinematics),
    }
    return (*batch[:2], targets, *batch[3:])


class MeasurementParTDataset(ParticleTransformerDataset):
    _NEEDED_COLUMNS = list(dict.fromkeys([
        *ParticleTransformerDataset._NEEDED_COLUMNS, "gen_jet_tau_vis_daughter_pdgs"
    ]))

    def build_tensors(self, data):
        return _with_truth_daughter_counts(data, super().build_tensors(data))


class MeasurementDETRDataset(ParticleTransformerDETRDataset):
    _NEEDED_COLUMNS = list(dict.fromkeys([
        *ParticleTransformerDETRDataset._NEEDED_COLUMNS,
        "gen_jet_tau_vis_daughter_pdgs", "gen_jet_tau_decaymode",
    ]))

    def build_tensors(self, data):
        batch = _with_truth_daughter_counts(data, super().build_tensors(data))
        return _with_direct_tau_targets(data, batch)


def load_targets(
    model: str, split: str, max_entries: int, overrides: list,
    daughters: str = "all",
) -> np.ndarray:
    """(N, 5) supervised kinematic targets, filtered by truth decay multiplicity."""
    if daughters not in ("all", "single", "multiple"):
        raise ValueError("daughters must be all, single or multiple")
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    config_name = "main" if model == "ParT" else "main_ParTauDETR"
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=["training.dataloader.num_dataloader_workers=0", *overrides],
        )
    if model == "ParT":
        datamodule = ParTDataModule(cfg=cfg, debug_run=False)
        if daughters != "all":
            datamodule.dataset_cls = MeasurementParTDataset
    else:
        datamodule = ParTauDETRDataModule(cfg=cfg, debug_run=False)
        if daughters != "all":
            datamodule.dataset_cls = MeasurementDETRDataset
    datamodule.setup("fit")
    loader = {"train": datamodule.train_dataloader, "val": datamodule.val_dataloader}[split]()
    if loader is None:
        raise RuntimeError(f"No {split} dataloader available; check the dataset split.")
    print(f"  DataLoader workers: {loader.num_workers}")

    chunks, n = [], 0
    for batch in loader:
        targets = batch[2]
        if not chunks:
            required = {"is_tau"}
            if daughters != "all":
                required.add("num_tau_daughters")
            if model == "ParT":
                required.add("kinematics")
            elif daughters == "single":
                required.update(("is_direct_tau", "direct_kinematics"))
            else:
                required.update(("particles_mask", "particles_kinematics"))
            missing = required.difference(targets)
            if missing:
                dataset = loader.dataset
                raise RuntimeError(
                    f"Scale measurement is missing target fields: {sorted(missing)}.\n"
                    f"Loaded dataset: {type(dataset).__name__} from {inspect.getfile(type(dataset))}\n"
                    f"Available targets: {sorted(targets)}\n"
                    "Update mltau/tools/io/ParTauDETR_dataloader.py and "
                    "mltau/tools/io/ParT_dataloader.py from the same checkout as this script, "
                    "then restart the command. num_tau_daughters must count original "
                    "gen_jet_tau_vis_daughter_pdgs before cuts or truncation; "
                    "particles_mask.sum() is not a valid substitute."
                )
        keep = torch.ones_like(targets["is_tau"], dtype=torch.bool)
        if daughters == "single":
            keep = targets["num_tau_daughters"] == 1
        elif daughters == "multiple":
            keep = targets["num_tau_daughters"] > 1
        if model == "ParT":
            keep = keep & targets["is_tau"].bool()
            kin = targets["kinematics"][keep]
        elif daughters == "single":
            keep = keep & targets["is_direct_tau"].bool()
            kin = targets["direct_kinematics"][keep]
        else:
            mask = targets["particles_mask"].bool() & keep[:, None]
            kin = targets["particles_kinematics"][mask]
        chunks.append(kin.double().numpy())
        n += len(kin)
        print(f"\r  read {n:,} targets", end="", flush=True)
        if n >= max_entries:
            break
    print()
    if not chunks or n == 0:
        sys.exit(
            f"No targets read for model={model}, daughters={daughters}; "
            "check the data and selection. Hybrid DETR excludes DM0/leptonic decays "
            "from decoder targets; use --model DETR --daughters single for single-daughter decays."
        )
    return np.concatenate(chunks)[:max_entries]


def spreads(x: np.ndarray) -> dict:
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return {
        "mean": x.mean(),
        "median": q50,
        "std": x.std(),
        "robust": 0.5 * (q84 - q16),
        "rms": np.sqrt(np.mean(x**2)),
    }


def summarize(kin: np.ndarray) -> dict:
    """Spreads of every quantity the loss components are built from."""
    dphi = np.arctan2(kin[:, 2], kin[:, 3])
    quantities = {
        "log_pt": kin[:, 0],
        "delta_eta": kin[:, 1],
        "dphi": dphi,
        "sin_dphi": kin[:, 2],
        "chord_to_axis": 2.0 * np.abs(np.sin(dphi / 2.0)),  # chord of the trivial prediction
        "log_mass": kin[:, 4],
    }
    table = {name: spreads(x) for name, x in quantities.items()}
    for name in ("log_pt", "log_mass"):
        table[name]["clamped"] = float(np.mean(np.abs(quantities[name]) >= LOG_CLAMP - 1e-6))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["ParT", "DETR"], required=True)
    parser.add_argument("--daughters", choices=["all", "single", "multiple"], default="all",
                        help="filter by original visible truth-daughter count; DETR single uses direct targets, all/multiple use decoder targets")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--estimator", choices=["std", "robust", "rms"], default="std",
                        help="estimator used for the printed YAML block")
    args, overrides = parser.parse_known_args()
    if args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    overrides = [o for o in overrides if o != "--"]

    print(f"[{args.model}] reading {args.split} targets (daughters={args.daughters}, max {args.max_entries:,})")
    kin = load_targets(args.model, args.split, args.max_entries, overrides,
                       daughters=args.daughters)
    table = summarize(kin)

    what = "visible taus" if args.model == "ParT" or args.daughters == "single" else "tau daughters"
    print(f"\n{len(kin):,} {what}\n")
    print(f"{'quantity':15s} {'mean':>9s} {'median':>9s} {'std':>9s} {'robust':>9s} {'rms':>9s}  at clamp")
    for name, s in table.items():
        clamp = f"{100 * s['clamped']:.2f}%" if "clamped" in s else ""
        print(f"{name:15s} {s['mean']:9.4f} {s['median']:9.4f} {s['std']:9.4f} "
              f"{s['robust']:9.4f} {s['rms']:9.4f}  {clamp}")

    e = args.estimator
    loss_config = "direct_tau_loss" if args.model == "DETR" and args.daughters == "single" else "tau_loss"
    print(f"\n# model.{loss_config}.kinematics_scales ({what}, {args.split}, daughters: {args.daughters}, {len(kin):,} entries, estimator: {e})")
    print("kinematics_scales:")
    print(f"    log_pt: {table['log_pt'][e]:.3g}")
    print(f"    delta_eta: {table['delta_eta'][e]:.3g}")
    print(f"    phi_chord: {table['dphi'][e]:.3g}    # spread of dphi; chord ~ |dphi| for collimated targets")
    print(f"    log_mass: {table['log_mass'][e]:.3g}")


if __name__ == "__main__":
    main()
