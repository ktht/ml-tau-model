from dataclasses import dataclass

import awkward as ak
import numpy as np
import torch
import tqdm
import vector
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from mltau.models.ParTauDETR_module import ParTauDETRModule
from mltau.tools.general import reinitialize_p4
from mltau.tools.io.ParTauDETR_dataloader import ParticleTransformerDETRDataset
from mltau.tools.partau_detr import decode_kinematics as decode_kinematics_p4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# from hydra import compose, initialize
# from omegaconf import OmegaConf

# with initialize(version_base=None, config_path="../config", job_name="test_app"):
#     cfg = compose(config_name="main_ParTauDETR")


def p4_from_components(components: torch.Tensor) -> ak.Array:
    """Convert 4-momentum from PyTorch tensor to Awkward array."""
    return vector.awk(
        ak.zip(
            {
                "px": components[..., 0],
                "py": components[..., 1],
                "pz": components[..., 2],
                "energy": components[..., 3],
            }
        )
    )


def sum_p4_components(p4: ak.Array) -> ak.Array:
    """Sum of 4-momenta in Cartesian coordinates."""
    total = vector.awk(
        ak.zip(
            {
                "px": ak.sum(p4.px, axis=1),
                "py": ak.sum(p4.py, axis=1),
                "pz": ak.sum(p4.pz, axis=1),
                "energy": ak.sum(p4.energy, axis=1),
            }
        )
    )
    return total


def _to_numpy(tensor) -> np.ndarray:
    """Dense numpy view of a torch tensor, wherever it lives."""
    return tensor.detach().cpu().numpy()


def _flat_view(arr: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat buffer plus per-event offsets for a jagged array.

    Leaving awkward ONCE is the whole point. Iterating a jagged array in Python
    costs ~2.5 ms per event here, because every element access rebuilds a
    record and every `np.asarray(evt.eta)` goes back through vector's behaviour
    dispatch; the same arithmetic on numpy slices costs ~30 us. That is the
    difference between a 21-point threshold scan taking an hour and a half and
    taking a minute.
    """
    counts = ak.to_numpy(ak.num(arr))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ak.to_numpy(ak.flatten(arr)), offsets


def _assign(
    p_eta, p_phi, p_ch, p_cls, t_eta, t_phi, t_ch, t_cls, max_dr, mismatch_penalty
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hungarian match for ONE event. All arguments are small 1-D numpy arrays.

    Returns within-event indices, so the result still indexes the caller's
    per-event lists.
    """
    if p_eta.size == 0 or t_eta.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    deta = np.abs(p_eta[:, None] - t_eta[None, :])
    dphi = np.abs(p_phi[:, None] - t_phi[None, :])
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    dr = np.sqrt(deta**2 + dphi**2)  # [n_pred, n_true]

    ch_mismatch = (p_ch[:, None] != t_ch[None, :]).astype(np.float64)
    cls_mismatch = (p_cls[:, None] != t_cls[None, :]).astype(np.float64)

    pred_idx, true_idx = linear_sum_assignment(
        dr + mismatch_penalty * (ch_mismatch + cls_mismatch)
    )
    valid = dr[pred_idx, true_idx] <= max_dr
    return pred_idx[valid].astype(np.int64), true_idx[valid].astype(np.int64)


def _predicted_components(outputs, reco_jet_p4s):
    """
    Dense [N, Q] predictions, before any objectness threshold is applied.

    Split out because none of it depends on the threshold: a scan would
    otherwise redo the softmaxes, the argmaxes and the kinematics decode once
    per point.
    """
    object_probs = torch.softmax(outputs["pred_logits"], dim=-1)
    pred_scores = object_probs[..., 0]

    pred_charge_probs = torch.softmax(outputs["pred_charge_logits"], dim=-1)
    pred_charge_cls = pred_charge_probs.argmax(dim=-1)
    charge_lut = outputs["pred_charge_logits"].new_tensor([-1, 0, 1], dtype=torch.long)
    pred_charge = charge_lut[pred_charge_cls]

    pred_meson_class = outputs["pred_meson_class_logits"].argmax(dim=-1)

    pred_p4_tensor = decode_kinematics_p4(
        outputs["pred_kinematics"],
        reco_jet_p4s["pt"],
        reco_jet_p4s["eta"],
        reco_jet_p4s["phi"],
        reco_jet_p4s["energy"],
    )
    pred_p4 = p4_from_components(pred_p4_tensor)
    return pred_scores, pred_p4, pred_charge, pred_meson_class


def tau_scores(outputs):
    """p(tau) per jet from the tauID head, or None if the model has no head."""
    if "is_tau" not in outputs:
        return None
    return torch.softmax(outputs["is_tau"].float(), dim=-1)[..., 1]


def get_predicted_particles(
    outputs,
    reco_jet_p4s,
    obj_cls_trsh: float = 0.5,
    tau_scores=None,
    tau_threshold: float | None = None,
):
    """
    Predicted daughters per jet: the queries whose objectness passes
    `obj_cls_trsh`, optionally only on jets the tauID head accepts.

    Whether a jet is a tau at all is the tauID head's decision. Objectness is
    trained on signal jets only, so on a background jet its scores are
    untrained and the selection they give is meaningless. Pass `tau_scores`
    (p(tau) per jet, see `tau_scores(outputs)`) together with `tau_threshold`
    to empty the daughter set of rejected jets; leave them out for a study
    restricted to true tau jets, where the tagger has nothing to add.
    """
    pred_scores, pred_p4, pred_charge, pred_meson_class = _predicted_components(
        outputs, reco_jet_p4s
    )
    pred_mask = pred_scores >= obj_cls_trsh
    if tau_scores is not None and tau_threshold is not None:
        tagged = torch.as_tensor(tau_scores).to(pred_mask.device) >= tau_threshold
        pred_mask = pred_mask & tagged[:, None]

    pred_p4 = ak.drop_none(ak.mask(pred_p4, pred_mask))
    pred_charge = ak.drop_none(ak.mask(pred_charge, pred_mask))
    pred_meson_class = ak.drop_none(ak.mask(pred_meson_class, pred_mask))
    return pred_p4, pred_charge, pred_meson_class


def get_true_particles(targets, reco_jet_p4s):
    target_mask = targets["particles_mask"].bool()

    target_charge_cls = targets["particles_charge_ohe"].argmax(dim=-1)
    charge_lut = target_charge_cls.new_tensor([-1, 0, 1], dtype=torch.long)
    target_charge = charge_lut[target_charge_cls]
    target_charge = ak.drop_none(ak.mask(target_charge, target_mask))

    target_meson_class = targets["particles_meson_class_ohe"].argmax(dim=-1)
    target_meson_class = ak.drop_none(ak.mask(target_meson_class, target_mask))

    true_p4_tensor = decode_kinematics_p4(
        targets["particles_kinematics"],
        reco_jet_p4s["pt"],
        reco_jet_p4s["eta"],
        reco_jet_p4s["phi"],
        reco_jet_p4s["energy"],
    )
    true_p4 = p4_from_components(true_p4_tensor)
    true_p4 = ak.drop_none(ak.mask(true_p4, target_mask))
    return true_p4, target_charge, target_meson_class




def match_particles(
    pred_p4: ak.Array,
    true_p4: ak.Array,
    pred_charge: ak.Array,
    true_charge: ak.Array,
    pred_meson_class: ak.Array,
    true_meson_class: ak.Array,
    max_dr: float = 0.4,
    mismatch_penalty: float = 5.0,
) -> ak.Array:
    """Match predicted to true particles per event via Hungarian matching.

    Cost = ΔR + penalty * (charge_mismatch + meson_class_mismatch).
    Unmatched particles are excluded.

    Args:
        pred_p4: jagged ak.Array of predicted 4-momenta per event.
        true_p4: jagged ak.Array of true 4-momenta per event.
        pred_charge, true_charge: charge arrays.
        pred_meson_class, true_meson_class: meson class arrays (0 charged, 1 neutral).
        max_dr: maximum ΔR for a valid match.
        mismatch_penalty: additive penalty for charge or PDG mismatch.
            A value of 5.0 means the matcher prefers a correct-identity
            match at ΔR=0.5 over a wrong-identity match at ΔR=0.0.

    Returns:
        ak.Array with fields ``pred_idx``, ``true_idx`` (jagged int per event).
    """
    p_eta, p_off = _flat_view(pred_p4.eta)
    p_phi, _ = _flat_view(pred_p4.phi)
    p_ch, _ = _flat_view(pred_charge)
    p_cls, _ = _flat_view(pred_meson_class)
    t_eta, t_off = _flat_view(true_p4.eta)
    t_phi, _ = _flat_view(true_p4.phi)
    t_ch, _ = _flat_view(true_charge)
    t_cls, _ = _flat_view(true_meson_class)

    pred_idx_list = []
    true_idx_list = []
    for i in range(len(p_off) - 1):
        a, b = p_off[i], p_off[i + 1]
        c, d = t_off[i], t_off[i + 1]
        pred_idx, true_idx = _assign(
            p_eta[a:b], p_phi[a:b], p_ch[a:b], p_cls[a:b],
            t_eta[c:d], t_phi[c:d], t_ch[c:d], t_cls[c:d],
            max_dr, mismatch_penalty,
        )
        pred_idx_list.append(pred_idx)
        true_idx_list.append(true_idx)

    return ak.Array(
        {"pred_idx": ak.Array(pred_idx_list), "true_idx": ak.Array(true_idx_list)}
    )


def verbose_true_pred_comparison(
    pred_meson_class: ak.Array,
    target_meson_class: ak.Array,
    pred_charge: ak.Array,
    target_charge: ak.Array,
    pred_p4: ak.Array,
    true_p4: ak.Array,
    data: ak.Array,
    matches: ak.Array,
    n_events: int = 20,
):
    total_pred_p4 = sum_p4_components(pred_p4)
    reduced_pred_p4 = sum_p4_components(pred_p4[matches.pred_idx])
    total_true_p4 = sum_p4_components(true_p4)
    reduced_true_p4 = sum_p4_components(true_p4[matches.true_idx])

    for i in range(n_events):
        print("--------------------------------------")
        print("--------------------------------------")
        print(f"------------- Event {i} -----------------")
        print("--------------------------------------")
        print(
            f"Number predicted particles: {len(pred_meson_class[i])}, \t Number true particles: {len(target_meson_class[i])}"
        )
        print("Best matches:")
        print("[Meson class: 0=charged, 1=neutral]")
        print(
            f"Pred: {pred_meson_class[matches.pred_idx][i]}\t True: {target_meson_class[matches.true_idx][i]}"
        )
        print(
            f"AllPred: {pred_meson_class[i]} \t AllTrue: {target_meson_class[i]}"
        )
        print("[Ch]")
        print(
            f"Pred: {pred_charge[matches.pred_idx][i]}\t True: {target_charge[matches.true_idx][i]}"
        )
        print(f"AllPred: {pred_charge[i]} \t AllTrue: {target_charge[i]}")
        print("[pT]")
        print(
            f"Pred: {pred_p4.pt[matches.pred_idx][i]}\t True: {true_p4.pt[matches.true_idx][i]}"
        )
        print(f"AllPred: {pred_p4.pt[i]} \t AllTrue: {true_p4.pt[i]}")
        print()
        print(f"RecoJet constituent PDGs: {data.reco_cand_pdgs[i]}")
        print(f"RecoJet constituent pTs: {reinitialize_p4(data.reco_cand_p4s).pt[i]}")
        print("--------------------------------------")
        print(
            r"Pred $\tau p_T$ (all predicted): ",
            total_pred_p4.pt[i],
            r"$\tau p_T$ (matched): ",
            reduced_pred_p4.pt[i],
        )
        print(
            r"True $\tau p_T$ (all true): ",
            total_true_p4.pt[i],
            r"$\tau p_T$ (matched): ",
            reduced_true_p4.pt[i],
        )
        print()
        print(
            r"Pred $\tau \phi$ (all predicted): ",
            total_pred_p4.phi[i],
            r"$\tau \phi$ (matched): ",
            reduced_pred_p4.phi[i],
        )
        print(
            r"True $\tau \phi$ (all true): ",
            total_true_p4.phi[i],
            r"$\tau \phi$ (matched): ",
            reduced_true_p4.phi[i],
        )
        print()
        print(
            r"Pred $\tau \eta$ (all predicted): ",
            total_pred_p4.eta[i],
            r"$\tau \eta$ (matched): ",
            reduced_pred_p4.eta[i],
        )
        print(
            r"True $\tau \eta$ (all true): ",
            total_true_p4.eta[i],
            r"$\tau \eta$ (matched): ",
            reduced_true_p4.eta[i],
        )
        print()
        print(
            r"Pred $\tau mass$ (all predicted): ",
            total_pred_p4.mass[i],
            r"$\tau mass$ (matched): ",
            reduced_pred_p4.mass[i],
        )
        print(
            r"True $\tau mass$ (all true): ",
            total_true_p4.mass[i],
            r"$\tau mass$ (matched): ",
            reduced_true_p4.mass[i],
        )


def _f1_from_counts(n_matched, n_pred, n_true) -> float:
    """Mean per-jet F1 of matched daughters. One definition, two call paths."""
    efficiency = np.divide(
        n_matched, n_true, out=np.zeros_like(n_true, dtype=float), where=n_true != 0
    )
    purity = np.divide(
        n_matched, n_pred, out=np.zeros_like(n_pred, dtype=float), where=n_pred != 0
    )
    num = 2 * (efficiency * purity)
    denom = efficiency + purity
    f1 = np.divide(num, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    return np.mean(f1)


def calculate_metrics(matches, target_meson_class, pred_meson_class):
    return _f1_from_counts(
        ak.num(matches.pred_idx).to_numpy(),
        ak.num(pred_meson_class).to_numpy(),
        ak.num(target_meson_class).to_numpy(),
    )


def load_model(checkpoint_path, cfg):
    model = ParTauDETRModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=DEVICE,
        cfg=cfg,
    )
    model.to(DEVICE)
    model.eval()
    return model


def model_inference(checkpoint_path, data_path, cfg, model=None):
    if model is None:
        model = load_model(checkpoint_path, cfg)
    data_paths = [data_path] if isinstance(data_path, (str, bytes)) else data_path
    data = ak.concatenate([ak.from_parquet(path) for path in data_paths])
    print(f"Read {len(data):,} jets from {len(data_paths)} parquet files.", flush=True)
    ds = ParticleTransformerDETRDataset.for_arrays(cfg)
    batch = ds.build_tensors(data)

    reco_jet_p4s = batch[6]

    with torch.no_grad():
        outputs, targets, _weights, _, _ = model.forward(batch)
    return outputs, targets, _weights, reco_jet_p4s, data


def create_predictions(
    outputs,
    targets,
    _weights,
    reco_jet_p4s,
    cfg,
    obj_cls_trsh=0.885,
    tau_threshold: float | None = None,
):
    """
    True and predicted daughter sets.

    `tau_threshold` gates the predicted set with the tauID head (see
    get_predicted_particles). Set it whenever the sample contains background
    jets; on a signal-only sample it only removes the true taus the tagger
    misses, which is a tagging inefficiency and not a reconstruction one.
    """
    targets = get_true_particles(targets, reco_jet_p4s)
    predictions = get_predicted_particles(
        outputs,
        reco_jet_p4s,
        obj_cls_trsh=obj_cls_trsh,
        tau_scores=tau_scores(outputs) if tau_threshold is not None else None,
        tau_threshold=tau_threshold,
    )
    true_daughters = TauDaughter(*targets)
    pred_daughters = TauDaughter(*predictions)
    return true_daughters, pred_daughters


def save_results(true_daughters, pred_daughters, output_path):
    results = ak.Array(
        {
            "pred_p4": pred_daughters.p4,
            "pred_charge": pred_daughters.charge,
            "pred_meson_class": pred_daughters.meson_class,
            "true_p4": true_daughters.p4,
            "true_charge": true_daughters.charge,
            "true_meson_class": true_daughters.meson_class,
        }
    )
    ak.to_parquet(results, output_path)


@dataclass
class TauDaughter:
    p4: torch.Tensor
    charge: torch.Tensor
    meson_class: torch.Tensor


def scan_thresholds(
    outputs,
    reco_jet_p4s,
    true_p4,
    target_charge,
    target_meson_class,
    cfg: DictConfig | None = None,
    thresholds=None,
    max_dr: float = 0.4,
    mismatch_penalty: float = 5.0,
) -> float:
    """
    Objectness threshold maximising the mean per-jet daughter F1.

    Everything that does not depend on the threshold is computed once, outside
    the loop: the softmaxes, the argmaxes and the kinematics decode (dense
    [N, Q], since no mask has been applied yet), and the truth side flattened
    to numpy. A point in the scan is then only a per-event boolean selection
    plus the assignment itself.

    Doing it the other way -- calling get_predicted_particles and
    match_particles per point, each walking jagged awkward arrays element by
    element -- costs about 4 minutes per point at 100k jets, so a 21-point scan
    ran for an hour and a half without tqdm ever advancing past 0/21.

    `cfg` is accepted for call-site compatibility and not used.
    """
    thresholds = np.linspace(0.9, 0.8, 21) if thresholds is None else np.asarray(thresholds)

    scores, pred_p4, pred_charge, pred_meson_class = _predicted_components(
        outputs, reco_jet_p4s
    )
    scores = _to_numpy(scores)  # [N, Q]
    p_eta = ak.to_numpy(pred_p4.eta)  # [N, Q]; dense, nothing masked yet
    p_phi = ak.to_numpy(pred_p4.phi)
    p_ch = _to_numpy(pred_charge)
    p_cls = _to_numpy(pred_meson_class)

    t_eta, t_off = _flat_view(true_p4.eta)
    t_phi, _ = _flat_view(true_p4.phi)
    t_ch, _ = _flat_view(target_charge)
    t_cls, _ = _flat_view(target_meson_class)
    n_true = np.diff(t_off)

    # True tau jets only. Objectness is trained on signal jets alone -- the
    # tauID head decides tau vs not -- so on background jets its scores are
    # untrained and every query above threshold would count as a false
    # positive against zero true daughters; with a 7:1 mix that would set the
    # threshold instead of the matching quality on taus. Every true tau has at
    # least one visible daughter, so n_true > 0 is the tag.
    signal = n_true > 0
    if not signal.all():
        print(
            f"threshold scan: {int(signal.sum()):,} jets with true daughters, "
            f"ignoring {int((~signal).sum()):,} without.",
            flush=True,
        )

    n_events = scores.shape[0]
    f1_scores = []
    for obj_cls_trsh in tqdm.tqdm(thresholds, desc="threshold scan", unit="point"):
        keep = scores >= obj_cls_trsh  # [N, Q]
        n_pred = keep.sum(axis=1)
        n_matched = np.zeros(n_events, dtype=np.int64)
        # Only jets that can produce a match are worth solving.
        for i in np.nonzero((n_pred > 0) & (n_true > 0))[0]:
            sel = keep[i]
            c, d = t_off[i], t_off[i + 1]
            pred_idx, _ = _assign(
                p_eta[i][sel], p_phi[i][sel], p_ch[i][sel], p_cls[i][sel],
                t_eta[c:d], t_phi[c:d], t_ch[c:d], t_cls[c:d],
                max_dr, mismatch_penalty,
            )
            n_matched[i] = pred_idx.size
        f1_scores.append(
            _f1_from_counts(n_matched[signal], n_pred[signal], n_true[signal])
        )

    best_thrsh = float(thresholds[int(np.argmax(f1_scores))])
    print(f"best objectness threshold {best_thrsh:.4f} (F1 {max(f1_scores):.4f})")
    return best_thrsh
