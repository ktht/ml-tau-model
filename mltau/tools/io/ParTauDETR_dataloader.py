import awkward as ak
import numpy as np
import torch
from omegaconf import DictConfig

from mltau.tools.io.ParT_dataloader import (
    ParTDataModule,
    ParticleTransformerDataset,
    has_p4_field,
    p4_field,
    sort_candidates_by_pt,
)
from mltau.tools.meson_classes import (
    get_meson_class_groups,
    pdg_to_meson_class_indices,
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
      - particles_kinematics: [N, T, 5] =
          [log(pt_dau/pt_jet), delta_eta(dau-jet), sin(delta_phi), cos(delta_phi), log(m_dau/m_jet)]
      - particles_charge_ohe: [N, T, 3] one-hot for charges [-1, 0, +1]
      - particles_meson_class_ohe: [N, T, C] one-hot over configured meson classes

    where T = cfg.dataset.max_tau_daughters if provided, otherwise inferred from
    the currently loaded row-group.
    """

    _NEEDED_COLUMNS = [
        "reco_cand_p4s",
        "reco_cand_charges",
        "reco_cand_pdgs",
        "reco_cand_signed_dz",
        "reco_cand_dz_error",
        "reco_cand_signed_dxy",
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

    @property
    def pdg_class_groups(self) -> tuple[tuple[int, ...], ...]:
        """Absolute PDG IDs grouped in configured class order."""
        return get_meson_class_groups(self.cfg.dataset.tau_daughter_pdg_ids)







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

    def _pdg_to_meson_class_indices(self, raw_pdg: np.ndarray) -> np.ndarray:
        return pdg_to_meson_class_indices(
            raw_pdg, self.cfg.dataset.tau_daughter_pdg_ids
        )

    def build_tensors(self, data: ak.Array):
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
        cand_dz = pad_cand(data.reco_cand_signed_dz)
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = pad_cand(data.reco_cand_signed_dxy)
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        # Reset sentinel values of track parameters to zero for neutral particles
        neutral_mask = cand_charge == 0
        charged_mask = ~neutral_mask
        for track_parameter in (cand_dz, cand_dz_err, cand_dxy, cand_dxy_err):
            track_parameter[neutral_mask] = 0.0

        impact_parameter_cfg = self.cfg.dataset.impact_parameter_features
        if impact_parameter_cfg.use_log_significance:
            cand_dz_err[charged_mask] = np.log(
                np.maximum(
                    np.abs(cand_dz[charged_mask])
                    / np.maximum(cand_dz_err[charged_mask], eps),
                    eps,
                )
            )
            cand_dxy_err[charged_mask] = np.log(
                np.maximum(
                    np.abs(cand_dxy[charged_mask])
                    / np.maximum(cand_dxy_err[charged_mask], eps),
                    eps,
                )
            )

        scale = float(impact_parameter_cfg.transform_scale)
        if scale > 0.0:
            cand_dz[charged_mask] = (
                2.0 * np.arctan(scale * cand_dz[charged_mask]) / np.pi
            )
            cand_dxy[charged_mask] = (
                2.0 * np.arctan(scale * cand_dxy[charged_mask]) / np.pi
            )

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

        cand_deta = cand_eta - jeta
        dphi_raw = cand_phi - jphi
        cand_dphi = np.arctan2(np.sin(dphi_raw), np.cos(dphi_raw))
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

        daughter_pt_cut = float(self.cfg.dataset.get("tau_daughter_pt_cut", -1.0))
        if daughter_pt_cut >= 0.0 and len(daughter_p4.fields) > 0:
            daughter_pt = self._get_record_field(daughter_p4, ["pt", "rho"])
            passes_pt_cut = daughter_pt >= daughter_pt_cut
            daughter_p4 = daughter_p4[passes_pt_cut]
            daughter_pdg_jag = daughter_pdg_jag[passes_pt_cut]
            daughter_charge_jag = daughter_charge_jag[passes_pt_cut]

        daughter_pdg_abs = abs(daughter_pdg_jag)
        supported_ids = [
            pdg_id for pdg_ids in self.pdg_class_groups for pdg_id in pdg_ids
        ]
        supported_pdg = daughter_pdg_abs == supported_ids[0]
        for pdg_id in supported_ids[1:]:
            supported_pdg = supported_pdg | (daughter_pdg_abs == pdg_id)

        daughter_p4 = daughter_p4[supported_pdg]
        daughter_pdg_jag = daughter_pdg_jag[supported_pdg]
        daughter_charge_jag = daughter_charge_jag[supported_pdg]

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

            # p4_field derives energy from any complete basis, including a
            # (pt, eta, phi, mass) record. What it cannot do is invent a fourth
            # coordinate, so a record carrying neither energy nor mass is the
            # one case left to handle here: treat those daughters as massless.
            if has_p4_field(daughter_p4, "energy") or has_p4_field(
                daughter_p4, "mass"
            ):
                dau_energy = self._pad_jagged(
                    p4_field(daughter_p4, "energy"),
                    max_tau_daughters,
                    fill=0.0,
                    dtype=np.float32,
                )
            else:
                dau_energy = dau_pt * np.cosh(dau_eta)

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

            dau_px = dau_pt * np.cos(dau_phi)
            dau_py = dau_pt * np.sin(dau_phi)
            dau_pz = dau_pt * np.sinh(dau_eta)
            daughter_p4_np = np.stack([dau_px, dau_py, dau_pz, dau_energy], axis=-1)

            # Daughter kinematic targets in the same spirit as ParT kinematics_tensor.
            _LOG_CLAMP = 5.0
            jet_pt_2d = np.maximum(jet_pt[:, None], eps)
            jet_eta_2d = jet_eta[:, None]
            jet_phi_2d = jet_phi[:, None]
            jet_mass = np.sqrt(
                np.maximum(jet_en**2 - (jet_pt * np.cosh(jet_eta)) ** 2, 0.0)
            )
            jet_mass_2d = np.maximum(jet_mass[:, None], eps)

            daughter_deta = dau_eta - jet_eta_2d
            daughter_dphi_raw = dau_phi - jet_phi_2d
            daughter_dphi = np.arctan2(
                np.sin(daughter_dphi_raw), np.cos(daughter_dphi_raw)
            )
            daughter_log_pt_ratio = np.clip(
                np.log(np.maximum(dau_pt / jet_pt_2d, eps)), -_LOG_CLAMP, _LOG_CLAMP
            )
            daughter_mass = np.sqrt(
                np.maximum(dau_energy**2 - (dau_pt * np.cosh(dau_eta)) ** 2, 0.0)
            )
            daughter_log_mass_ratio = np.clip(
                np.log(np.maximum(daughter_mass / jet_mass_2d, eps)),
                -_LOG_CLAMP,
                _LOG_CLAMP,
            )

            daughter_kinematics_np = np.stack(
                [
                    daughter_log_pt_ratio,
                    daughter_deta,
                    np.sin(daughter_dphi),
                    np.cos(daughter_dphi),
                    daughter_log_mass_ratio,
                ],
                axis=-1,
            )

            daughter_p4_np *= daughter_mask_np[..., None]
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
            daughter_p4_np = np.zeros((n_jets, n_slots, 4), dtype=np.float32)
            daughter_kinematics_np = np.zeros((n_jets, n_slots, 5), dtype=np.float32)
            daughter_charge = np.zeros((n_jets, n_slots), dtype=np.int64)
            daughter_pdg = np.zeros((n_jets, n_slots), dtype=np.int64)

        charge_cls = self._charges_to_class_indices(daughter_charge)
        meson_class = self._pdg_to_meson_class_indices(daughter_pdg)

        # Prepare one-hot targets; unknown classes stay all-zero.
        n_charge = len(self.CHARGE_CLASS_VALUES)
        n_meson_classes = len(self.pdg_class_groups)
        charge_ohe = np.zeros((*charge_cls.shape, n_charge), dtype=np.float32)
        meson_class_ohe = np.zeros(
            (*meson_class.shape, n_meson_classes), dtype=np.float32
        )

        valid_charge = charge_cls >= 0
        valid_meson_class = meson_class >= 0
        if np.any(valid_charge):
            rows, cols = np.where(valid_charge)
            charge_ohe[rows, cols, charge_cls[rows, cols]] = 1.0
        if np.any(valid_meson_class):
            rows, cols = np.where(valid_meson_class)
            meson_class_ohe[rows, cols, meson_class[rows, cols]] = 1.0

        # Zero out padded daughters in one-hot tensors too.
        charge_ohe *= daughter_mask_np[..., None]
        meson_class_ohe *= daughter_mask_np[..., None]

        targets = {
            "particles_mask": torch.from_numpy(daughter_mask_np).bool(),
            "particles_kinematics": torch.from_numpy(daughter_kinematics_np).float(),
            "particles_charge_ohe": torch.from_numpy(charge_ohe).float(),
            "particles_meson_class_ohe": torch.from_numpy(meson_class_ohe).float(),
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


