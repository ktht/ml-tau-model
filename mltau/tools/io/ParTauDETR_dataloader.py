import awkward as ak
import numpy as np
import torch
from omegaconf import DictConfig
from ntupelizer.tools.tau_decaymode import classify_decay_modes

from mltau.tools.io.ParT_dataloader import (
    ParTDataModule,
    ParticleTransformerDataset,
    p4_field,
    sample_name,
    sort_candidates_by_pt,
)


class ParticleTransformerDETRDataset(ParticleTransformerDataset):
    """
    ParT-style dataset for DETR set-to-set training.

    Inputs are kept identical to ParticleTransformerDataset:
      - cand_features: [N, 17, max_cands]
      - cand_kinematics: [N, 4, max_cands]
      - cand_mask: [N, 1, max_cands]

    Targets are replaced with daughter-level set targets:
      - particles_mask: [N, T] (True for valid daughter)
      - particles_kinematics: [N, T, 4] =
          [log(pt_dau/pt_jet), delta_eta(dau-jet), sin(delta_phi), cos(delta_phi)]
      - particles_charge_ohe: [N, T, 3] one-hot for charges [-1, 0, +1]

    where T = cfg.dataset.max_tau_daughters if provided, otherwise inferred from
    the currently loaded row-group.
    """

    _NEEDED_COLUMNS = [
        "reco_cand_p4s",
        "reco_cand_charges",
        "reco_cand_pdgs",
        "reco_cand_dz",
        "reco_cand_dz_error",
        "reco_cand_dxy",
        "reco_cand_dxy_error",
        "reco_jet_p4",
        "gen_jet_tau_p4",
        "gen_jet_p4",
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
        "gen_jet_tau_decaymode",
        "gen_jet_tau_charge",
        "cls_weight",
    ]

    # Charge classes are a structural constant, not configuration: the model
    # rejects anything but three classes and ParTauDETRModule.predict_step maps
    # them back through a fixed [-1, 0, +1] lookup table.
    CHARGE_CLASS_VALUES = [-1, 0, 1]
    CHARGE_TO_CLASS = {q: i for i, q in enumerate(CHARGE_CLASS_VALUES)}

    _FILTER_COLUMNS = [
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
        "gen_jet_tau_decaymode",
        "gen_jet_tau_charge",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.row_groups and (
            self.cfg.dataset.get("pion_filter", False)
            or self.cfg.dataset.get("quality_cuts", False)
        ):
            counted_units = []
            for filename, indices, _ in self.read_units:
                data = ak.from_parquet(
                    filename, row_groups=indices, columns=self._FILTER_COLUMNS
                )
                n_kept = int(ak.sum(self._selected_jets(data)))
                if n_kept:
                    counted_units.append((filename, indices, n_kept))
            self.read_units = counted_units
            self.num_rows = sum(unit[2] for unit in counted_units)
            self.reads_by_sample = {}
            for unit in counted_units:
                self.reads_by_sample.setdefault(sample_name(unit[0]), []).append(unit)

    def _acceptance_mask(self, data: ak.Array):
        daughters = data.gen_jet_tau_vis_daughter_p4s
        if not self.cfg.dataset.get("quality_cuts", False) or not daughters.fields:
            return ak.ones_like(data.gen_jet_tau_vis_daughter_pdgs, dtype=bool)
        pt = p4_field(daughters, "pt")
        eta = p4_field(daughters, "eta")
        return (pt > float(self.cfg.dataset.tau_daughter_pt_cut)) & (
            abs(np.tanh(eta)) < float(self.cfg.dataset.tau_daughter_max_abs_cos_theta)
        )

    def _selected_jets(self, data: ak.Array):
        signal = data.gen_jet_tau_decaymode != -1
        pdgs = data.gen_jet_tau_vis_daughter_pdgs
        selected = ak.ones_like(signal, dtype=bool)
        if self.cfg.dataset.get("pion_filter", False):
            all_pions = (abs(pdgs) == 211) | (abs(pdgs) == 111)
            selected = selected & (~signal | (ak.all(all_pions, axis=1) & ak.any(all_pions, axis=1)))
        if self.cfg.dataset.get("quality_cuts", False):
            accepted = self._acceptance_mask(data)
            surviving_pdgs = pdgs[accepted][signal]
            surviving_charges = data.gen_jet_tau_vis_daughter_charges[accepted][signal]
            charge_ok = ak.sum(surviving_charges, axis=1) == data.gen_jet_tau_charge[signal]
            decay_ok = ak.Array(classify_decay_modes(surviving_pdgs)) == data.gen_jet_tau_decaymode[signal]
            keep = ak.to_numpy(selected)
            keep[ak.to_numpy(signal)] &= ak.to_numpy(charge_ok & decay_ok)
            selected = ak.Array(keep)
        return selected


    @staticmethod
    def _pad_jagged(arr, max_len: int, fill=0.0, dtype=None):
        out = ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_len, clip=True), fill))
        return out.astype(dtype) if dtype is not None else out

    @staticmethod
    def _get_record_field(record_array, names: list[str]):
        for name in names:
            if name in record_array.fields:
                return record_array[name]
        raise KeyError(
            f"Could not find any of fields {names} in {record_array.fields}."
        )

    def _get_max_tau_daughters(self, n_daughters: np.ndarray) -> int:
        configured = self.cfg.dataset.get("max_tau_daughters", None)
        if configured is not None:
            return int(configured)
        if n_daughters.size == 0:
            return 0
        return int(np.max(n_daughters))

    @staticmethod
    def _sort_tau_daughters_by_pt(daughter_p4, daughter_pdg, daughter_charge):
        if len(daughter_p4.fields) == 0:
            return daughter_p4, daughter_pdg, daughter_charge
        order = ak.argsort(p4_field(daughter_p4, "pt"), axis=-1, ascending=False)
        return daughter_p4[order], daughter_pdg[order], daughter_charge[order]

    @classmethod
    def _charges_to_class_indices(cls, raw_charge: np.ndarray) -> np.ndarray:
        out = np.full(raw_charge.shape, -1, dtype=np.int64)
        q = np.rint(raw_charge).astype(np.int64)
        for val, idx in cls.CHARGE_TO_CLASS.items():
            out[q == val] = idx
        return out

    def build_tensors(self, data: ak.Array):
        if self.cfg.dataset.get("pion_filter", False) or self.cfg.dataset.get("quality_cuts", False):
            data = data[self._selected_jets(data)]
        if self.cfg.dataset.get("sort_by_pt", False):
            data = sort_candidates_by_pt(data)
        # -------------------------
        # Inputs (unchanged)
        # -------------------------
        max_cands = self.cfg.dataset.max_cands
        eps = 1e-6

        def pad_cand(arr, fill=0.0):
            return self._pad_jagged(arr, max_cands, fill=fill, dtype=np.float32)

        # Candidate-level quantities
        cand_pt = pad_cand(p4_field(data.reco_cand_p4s, "pt"))
        cand_eta = pad_cand(p4_field(data.reco_cand_p4s, "eta"))
        cand_phi = pad_cand(p4_field(data.reco_cand_p4s, "phi"))
        cand_en = pad_cand(p4_field(data.reco_cand_p4s, "energy"))
        cand_charge = pad_cand(data.reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(data.reco_cand_pdgs))
        cand_dz = pad_cand(data.reco_cand_dz)
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = pad_cand(data.reco_cand_dxy)
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        lengths = np.minimum(ak.to_numpy(ak.num(data.reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Jet-level p4 for feature engineering and bookkeeping
        jet_pt = ak.to_numpy(p4_field(data.reco_jet_p4, "pt")).astype(np.float32)
        jet_eta = ak.to_numpy(p4_field(data.reco_jet_p4, "eta")).astype(np.float32)
        jet_phi = ak.to_numpy(p4_field(data.reco_jet_p4, "phi")).astype(np.float32)
        jet_en = ak.to_numpy(p4_field(data.reco_jet_p4, "energy")).astype(np.float32)

        gen_tau_pt = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "pt")).astype(np.float32)
        gen_tau_eta = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "eta")).astype(np.float32)
        gen_tau_phi = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "phi")).astype(np.float32)
        gen_tau_energy = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "energy")).astype(np.float32)

        gen_jet_pt = ak.to_numpy(p4_field(data.gen_jet_p4, "pt")).astype(np.float32)
        gen_jet_eta = ak.to_numpy(p4_field(data.gen_jet_p4, "eta")).astype(np.float32)
        gen_jet_phi = ak.to_numpy(p4_field(data.gen_jet_p4, "phi")).astype(np.float32)
        gen_jet_energy = ak.to_numpy(p4_field(data.gen_jet_p4, "energy")).astype(np.float32)

        # 17 ParticleTransformer features
        jpt = jet_pt[:, None]
        jeta = jet_eta[:, None]
        jphi = jet_phi[:, None]
        jen = jet_en[:, None]

        cand_deta = np.abs(cand_eta - jeta)
        dphi_raw = cand_phi - jphi
        cand_dphi = np.abs(np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw)))
        cand_logpt = np.log(np.maximum(cand_pt, eps))
        cand_loge = np.log(np.maximum(cand_en, eps))
        cand_logptrel = np.log(np.maximum(cand_pt / np.maximum(jpt, eps), eps))
        cand_logerel = np.log(np.maximum(cand_en / np.maximum(jen, eps), eps))
        cand_dR = np.sqrt(cand_deta**2 + cand_dphi**2)

        is_electron = (cand_pdg_abs == 11).astype(np.float32)
        is_muon = (cand_pdg_abs == 13).astype(np.float32)
        is_photon = (cand_pdg_abs == 22).astype(np.float32)
        is_charged_hadron = (cand_pdg_abs == 211).astype(np.float32)
        is_neutral_hadron = (cand_pdg_abs == 130).astype(np.float32)

        cand_features_np = np.stack(
            [
                cand_deta,
                cand_dphi,
                cand_logpt,
                cand_loge,
                cand_logptrel,
                cand_logerel,
                cand_dR,
                cand_charge,
                is_electron,
                is_muon,
                is_photon,
                is_charged_hadron,
                is_neutral_hadron,
                cand_dz,
                cand_dz_err,
                cand_dxy,
                cand_dxy_err,
            ],
            axis=1,
        )
        cand_features_np *= mask_np[:, None, :]
        np.nan_to_num(cand_features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Candidate kinematics [px, py, pz, E]
        cand_px = cand_pt * np.cos(cand_phi)
        cand_py = cand_pt * np.sin(cand_phi)
        cand_pz = cand_pt * np.sinh(cand_eta)
        cand_kinematics_np = np.stack([cand_px, cand_py, cand_pz, cand_en], axis=1)
        cand_kinematics_np *= mask_np[:, None, :]

        # Optional per-jet training weight
        if "cls_weight" not in data.fields:
            weight_tensors = torch.ones(len(data), dtype=torch.float32)
        else:
            weight_tensors = torch.from_numpy(
                ak.to_numpy(data.cls_weight).astype(np.float32)
            )

        # -------------------------
        # DETR set targets
        # -------------------------
        daughter_pdg_jag = data.gen_jet_tau_vis_daughter_pdgs
        daughter_p4 = data.gen_jet_tau_vis_daughter_p4s
        daughter_charge_jag = data.gen_jet_tau_vis_daughter_charges

        if self.cfg.dataset.get("quality_cuts", False) and len(daughter_p4.fields) > 0:
            accepted = self._acceptance_mask(data)
            daughter_p4 = daughter_p4[accepted]
            daughter_pdg_jag = daughter_pdg_jag[accepted]
            daughter_charge_jag = daughter_charge_jag[accepted]

        daughter_p4, daughter_pdg_jag, daughter_charge_jag = (
            self._sort_tau_daughters_by_pt(
                daughter_p4, daughter_pdg_jag, daughter_charge_jag
            )
        )

        daughter_counts = ak.to_numpy(ak.num(daughter_pdg_jag)).astype(np.int64)
        max_tau_daughters = self._get_max_tau_daughters(daughter_counts)

        # Failsafe: background samples (and any jet with no visible daughters)
        # store daughter_p4 as an empty/unknown-typed array with no fields, so
        # there is nothing to extract.  Fall through to the zero-filled branch
        # instead of raising KeyError in _get_record_field.
        if max_tau_daughters > 0 and len(daughter_p4.fields) > 0:
            dau_pt = self._pad_jagged(
                p4_field(daughter_p4, "pt"),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )
            dau_eta = self._pad_jagged(
                p4_field(daughter_p4, "eta"),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )
            dau_phi = self._pad_jagged(
                p4_field(daughter_p4, "phi"),
                max_tau_daughters,
                fill=0.0,
                dtype=np.float32,
            )

            daughter_charge = self._pad_jagged(
                daughter_charge_jag,
                max_tau_daughters,
                fill=0,
                dtype=np.int64,
            )
            daughter_pdg = self._pad_jagged(
                daughter_pdg_jag,
                max_tau_daughters,
                fill=0,
                dtype=np.int64,
            )

            clipped_counts = np.minimum(daughter_counts, max_tau_daughters)
            daughter_mask_np = (
                np.arange(max_tau_daughters)[None, :] < clipped_counts[:, None]
            )

            # Daughter kinematic targets in the same spirit as ParT kinematics_tensor.
            _LOG_CLAMP = 5.0
            jet_pt_2d = np.maximum(jet_pt[:, None], eps)
            jet_eta_2d = jet_eta[:, None]
            jet_phi_2d = jet_phi[:, None]

            daughter_deta = dau_eta - jet_eta_2d
            daughter_dphi_raw = dau_phi - jet_phi_2d
            daughter_dphi = np.arctan2(
                np.sin(daughter_dphi_raw), np.cos(daughter_dphi_raw)
            )
            daughter_log_pt_ratio = np.clip(
                np.log(np.maximum(dau_pt / jet_pt_2d, eps)), -_LOG_CLAMP, _LOG_CLAMP
            )
            daughter_kinematics_np = np.stack(
                [
                    daughter_log_pt_ratio,
                    daughter_deta,
                    np.sin(daughter_dphi),
                    np.cos(daughter_dphi),
                ],
                axis=-1,
            )

            daughter_kinematics_np *= daughter_mask_np[..., None]
            np.nan_to_num(
                daughter_kinematics_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            # Keep the daughter axis at max_tau_daughters even with nothing to
            # put in it. A background read that emitted T=0 while a signal read
            # emitted T=8 cannot be concatenated, which breaks batches that mix
            # the two, and would make the target shape depend on which file a
            # batch happened to come from. The all-False mask already tells the
            # criterion that none of these slots carry supervision.
            n_jets = len(data)
            n_slots = max(int(max_tau_daughters), 0)
            daughter_mask_np = np.zeros((n_jets, n_slots), dtype=bool)
            daughter_kinematics_np = np.zeros((n_jets, n_slots, 4), dtype=np.float32)
            daughter_charge = np.zeros((n_jets, n_slots), dtype=np.int64)
            daughter_pdg = np.zeros((n_jets, n_slots), dtype=np.int64)

        charge_cls = self._charges_to_class_indices(daughter_charge)

        # Prepare one-hot targets; unknown classes stay all-zero.
        n_charge = len(self.CHARGE_CLASS_VALUES)
        charge_ohe = np.zeros((*charge_cls.shape, n_charge), dtype=np.float32)

        valid_charge = charge_cls >= 0
        if np.any(valid_charge):
            rows, cols = np.where(valid_charge)
            charge_ohe[rows, cols, charge_cls[rows, cols]] = 1.0

        # Zero out padded daughters in one-hot tensors too.
        charge_ohe *= daughter_mask_np[..., None]

        targets = {
            "particles_mask": torch.from_numpy(daughter_mask_np).bool(),
            "particles_kinematics": torch.from_numpy(daughter_kinematics_np).float(),
            "particles_charge_ohe": torch.from_numpy(charge_ohe).float(),
            # Jet-level tau-tagging label, following ParticleTransformerDataset:
            # -1 -> no genuine tau (background), >= 0 -> genuine tau (signal).
            "is_tau": torch.from_numpy(
                (ak.to_numpy(data.gen_jet_tau_decaymode) != -1).astype(np.int64)
            ),
            "gen_jet_tau_charge": torch.from_numpy(
                ak.to_numpy(data.gen_jet_tau_charge).astype(np.int64)
            ),
            "gen_jet_tau_decaymode": torch.from_numpy(
                ak.to_numpy(data.gen_jet_tau_decaymode).astype(np.int64)
            ),
        }

        return self._scaled((
            torch.from_numpy(cand_features_np),
            torch.from_numpy(cand_kinematics_np),
            targets,
            torch.from_numpy(mask_np).unsqueeze(1),
            weight_tensors.float(),
            {
                "pt": torch.from_numpy(gen_tau_pt),
                "eta": torch.from_numpy(gen_tau_eta),
                "phi": torch.from_numpy(gen_tau_phi),
                "energy": torch.from_numpy(gen_tau_energy),
            },
            {
                "pt": torch.from_numpy(jet_pt),
                "eta": torch.from_numpy(jet_eta),
                "phi": torch.from_numpy(jet_phi),
                "energy": torch.from_numpy(jet_en),
            },
            {
                "pt": torch.from_numpy(gen_jet_pt),
                "eta": torch.from_numpy(gen_jet_eta),
                "phi": torch.from_numpy(gen_jet_phi),
                "energy": torch.from_numpy(gen_jet_energy),
            },
        ))









class ParTauDETRDataModule(ParTDataModule):
    """
    DataModule variant using ParticleTransformerDETRDataset.

    File discovery, the train/val split, the scaler fit and the stratified
    batch composition are all the base class's; only the dataset class and the
    sample selection differ.
    """

    dataset_cls = ParticleTransformerDETRDataset

    def __init__(self, cfg: DictConfig, debug_run: bool = False):
        super().__init__(cfg=cfg, debug_run=debug_run)
        # Tau tagging is a binary signal-vs-background task, so we need the
        # background samples in addition to the signal samples. The base class
        # otherwise defaults to signal-only (`sample = "z"`) for set-to-set.
        if cfg.model.detr.tau_id_head:
            self.sample = "*"


