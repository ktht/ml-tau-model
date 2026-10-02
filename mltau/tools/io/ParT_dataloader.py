import glob
import math
import os
import warnings
from collections.abc import Sequence

import awkward as ak
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset

from mltau.tools import features as f
from mltau.tools import general as g
from mltau.tools.io import general as ig  # RowGroupDataset
from mltau.tools.io import input_scaling as scaling

np.random.seed(42)

# Four-vectors reach us under several different field namings depending on how
# the ntuple was written: vector's Momentum4D storage uses (rho, eta, phi, t),
# other producers use (pt, eta, phi, energy), and some write a Cartesian
# (px, py, pz, energy) record instead. Awkward's `arr["pt"]` looks up a FIELD,
# not vector's `pt` property, so the wrong naming raises FieldNotFoundError
# rather than being resolved by the behaviour.
#
# ParT_dataloader used to hardcode rho/t and ParTauDETR_dataloader pt/energy, so
# whichever layout the data had, one of them broke. Go through p4_field instead.
P4_FIELD_ALIASES = {
    "pt": ("pt", "rho"),
    "eta": ("eta",),
    "phi": ("phi",),
    "energy": ("energy", "t", "E", "e"),
    "mass": ("mass", "m", "tau"),
}


def has_p4_field(record_array, quantity: str) -> bool:
    """True if `quantity` is stored outright under one of its aliases.

    Only asks about stored fields, so it stays a cheap question about the
    layout: a Cartesian record has no `pt` field even though p4_field can hand
    one back.
    """
    fields = record_array.fields
    return any(alias in fields for alias in P4_FIELD_ALIASES[quantity])


def p4_field(record_array, quantity: str):
    """Read `quantity` from a p4 record, whatever layout it was written in.

    A stored field wins, because reading one is free and exact. Otherwise the
    record is in a basis that does not carry `quantity` at all -- Cartesian
    (px, py, pz, energy) has none of pt/eta/phi, and a (pt, eta, phi, mass)
    record has no energy -- so hand it to vector via `reinitialize_p4`, which
    picks whichever complete basis is present and derives the rest.
    """
    aliases = P4_FIELD_ALIASES[quantity]
    fields = record_array.fields
    for alias in aliases:
        if alias in fields:
            return record_array[alias]

    try:
        return getattr(g.reinitialize_p4(record_array), quantity)
    except Exception as exc:
        raise KeyError(
            f"no field for {quantity!r} in p4 record: tried {list(aliases)} and "
            f"deriving it from the stored basis; record has {fields}"
        ) from exc


_CANDIDATE_FIELDS = (
    "reco_cand_p4s",
    "reco_cand_charges",
    "reco_cand_pdgs",
    "reco_cand_signed_dz",
    "reco_cand_dz_error",
    "reco_cand_signed_dxy",
    "reco_cand_dxy_error",
)


def sort_candidates_by_pt(data: ak.Array) -> ak.Array:
    """
    Order every jet's candidates by descending pT, consistently across all
    candidate-level fields. Only per-jet fields are left alone.

    The ParT encoder is permutation invariant and gains nothing from this; the
    MLP-Mixer is not, so for it the order is part of the input definition.
    Controlled by `dataset.sort_by_pt`.
    """
    order = ak.argsort(p4_field(data.reco_cand_p4s, "pt"), axis=1, ascending=False)
    for field in _CANDIDATE_FIELDS:
        if field in data.fields:
            data = ak.with_field(data, data[field][order], field)
    return data


def sample_name(path: str) -> str:
    """
    Sample label from a `{sample}_train*.parquet` / `{sample}_test*.parquet` path.

    Module level rather than a DataModule method because the dataset needs the
    same rule: a read covers one file and therefore one sample, and the batch
    composition is built on knowing which.
    """
    base = os.path.basename(path)
    for split in ("_train", "_test"):
        if split in base:
            return base.split(split)[0]
    return os.path.splitext(base)[0]


def loader_kwargs(num_workers: int, prefetch_factor, debug_run: bool) -> dict:
    """
    DataLoader arguments that are only valid for a given worker count.

    With num_workers=0 PyTorch rejects both `persistent_workers=True` and
    `prefetch_factor`, so passing them unconditionally made
    `num_dataloader_workers=0` raise. That is the one configuration that
    isolates worker startup from everything else, so it has to work -- it is
    the first thing to try when a job hangs before the first batch.

    `multiprocessing_context` is decided from the RESOLVED worker count, not the
    configured one: a request of 6 clamped to 1 by the cpuset should not still
    select forkserver.
    """
    kwargs = {"num_workers": int(num_workers), "pin_memory": True}
    if num_workers > 0:
        kwargs["persistent_workers"] = not debug_run
        kwargs["prefetch_factor"] = None if debug_run else prefetch_factor
        if num_workers > 1:
            kwargs["multiprocessing_context"] = "forkserver"
    return kwargs


def resolve_num_workers(requested: int) -> int:
    """
    Clamp the worker count to the CPUs this process may actually use.

    `os.sched_getaffinity` reflects the Slurm cpuset, so this catches a job that
    asked for one cpu but configured several workers -- they would otherwise
    timeshare a single core and stall the first batch for minutes.
    """
    requested = int(requested)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    # Leave one core for the main process that feeds the GPU.
    usable = max(1, available - 1) if available > 1 else 1
    if requested > usable:
        warnings.warn(
            f"training.dataloader.num_dataloader_workers={requested} but only "
            f"{available} cpu(s) are available to this process; using {usable}. "
            "Request more cpus (e.g. #SBATCH --cpus-per-task=8) to use more "
            "workers.",
            stacklevel=2,
        )
        return usable
    return requested

class ParticleTransformerDataset(IterableDataset):
    """
    Streams jets from parquet row groups as ready-made batches.

    Reads are planned per file (`row_groups_per_read` row groups per read),
    sharded per sample across workers, and -- when more than one sample is
    present -- every batch is composed from all samples at once, so the class
    mix is a property of the loader rather than of which files happened to be
    read together. `__len__` is exact. Subclasses provide `_NEEDED_COLUMNS` and
    `build_tensors`; everything about reading and batching is shared.
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
        "gen_jet_tau_decaymode",
        "gen_jet_tau_charge",
        "cls_weight",
        ]

    def __init__(
        self,
        row_groups: Sequence[ig.RowGroup],
        cfg: DictConfig,
        batch_size: int = 1,
        shuffle: bool = False,
        row_groups_per_read: int = 1,
        mixing_reads: int = 1,
        cache_parquet_handles: bool = True,
        num_workers: int = 0,
        stratify_samples: bool = True,
    ):
        """
        Args:
            shuffle: reshuffle the read order and the jets inside each loaded
                chunk on every epoch.
            stratify_samples: draw every batch from all samples at once instead
                of concatenating whole reads and leaving the class mix to the
                draw. Turn it off only where the emission ORDER matters.
            row_groups_per_read: number of consecutive row groups pulled in a
                single `ak.from_parquet` call. Each such call re-opens the file
                and re-parses the whole Parquet footer (every row group x every
                column), so with small row groups that fixed cost dominates the
                actual payload and scales as O(n_row_groups^2) per epoch.
                Coalescing divides the number of footer parses by this factor.
            mixing_reads: number of reads held in memory at once. Values > 1 mix
                signal and background into the same batch, at the cost of
                proportionally more worker memory. A read covers one file and
                therefore one class, so with mixing_reads=2 about half of all
                chunks are still single-class; 4 brings that to ~12%.
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.row_groups = row_groups
        self.num_rows = sum(rg.num_rows for rg in self.row_groups)
        # Set by the DataModule, or by for_arrays, when training.input_scaling
        # is enabled. Applied inside build_tensors, which is the single place
        # these tensors are constructed, so every consumer -- training,
        # evaluation, notebooks -- standardises identically without having to
        # remember to ask.
        self.input_scaler = None
        # Counts __iter__ calls, so every epoch gets its own reproducible
        # shuffle (see _epoch_rng).
        self._epochs_started = 0
        if self.row_groups:
            print(
                f"There are {'{:,}'.format(self.num_rows)} jets in the dataset.",
                flush=True,
            )
        self.shuffle = shuffle
        self.cache_parquet_handles = bool(cache_parquet_handles)
        # Filled lazily inside the worker; see _parquet_handle.
        self._handles = None
        self.mixing_reads = max(1, int(mixing_reads))
        # Needed for an exact __len__: batches are counted per worker shard.
        self.num_workers = max(0, int(num_workers))
        self.stratify_samples = bool(stratify_samples)
        self.read_units = self._build_read_units(
            row_groups, max(1, int(row_groups_per_read))
        )
        # A read covers one file and therefore one sample, so this grouping is
        # what both the worker sharding and the batch composition are built on.
        self.reads_by_sample: dict[str, list] = {}
        for unit in self.read_units:
            self.reads_by_sample.setdefault(sample_name(unit[0]), []).append(unit)
        if row_groups:
            print(
                f"Grouped {len(row_groups):,} row groups into "
                f"{len(self.read_units):,} parquet read(s) over "
                + ", ".join(
                    f"{name}={len(units):,} read(s)"
                    for name, units in sorted(self.reads_by_sample.items())
                )
                + (
                    "; batches stratified across samples."
                    if self.stratify_samples and len(self.reads_by_sample) > 1
                    else "."
                ),
                flush=True,
            )
            self._warn_if_shards_lose_a_sample()


    def _epoch_rng(self) -> np.random.Generator:
        """
        Generator for this epoch's read order and in-chunk shuffles.

        Seeded, so a run is reproducible from `training.seed` (which the
        training scripts hand to seed_everything) -- an unseeded default_rng()
        here made the read order differ between two runs of the same config
        while the comment on `seed` claimed it covered shuffling. Three
        ingredients: the configured seed; torch's per-process seed, which the
        DataLoader derives from the global seed per worker so workers do not
        replay each other; and the epoch counter, so persistent workers do not
        replay the previous epoch.
        """
        self._epochs_started += 1
        cfg_seed = 42
        if "training" in self.cfg and "seed" in self.cfg.training:
            cfg_seed = int(self.cfg.training.seed)
        process_seed = int(torch.initial_seed() % (2**31 - 1))
        return np.random.default_rng([cfg_seed, process_seed, self._epochs_started])

    def set_input_scaler(self, scaler) -> None:
        """Attach a `tensors -> tensors` callable, or None to disable scaling."""
        self.input_scaler = scaler

    def _scaled(self, tensors):
        return tensors if self.input_scaler is None else self.input_scaler(tensors)

    @classmethod
    def for_arrays(cls, cfg: DictConfig):
        """
        Dataset bound to `cfg` alone, for running `build_tensors` on arrays that
        are already in memory.

        Inference reads its parquet file itself and needs only the
        tensor-building half of the dataset, so there are no row groups to plan
        reads over. Constructing that with `row_groups=[]` used to make the
        dataset announce "There are 0 jets in the dataset" about a file it had
        just read in full, which reads like data loss and is why the counts are
        now printed only when there is actually something to read.
        """
        dataset = cls(row_groups=[], cfg=cfg, batch_size=1)
        # Inference must standardise exactly as training did; reading the scaler
        # here means a caller cannot forget to.
        dataset.set_input_scaler(scaling.make_input_scaler(cfg))
        return dataset


    def _warn_if_shards_lose_a_sample(self) -> None:
        """
        Stratification is per worker, so every worker needs every sample.

        Reads are strided within each sample, so a sample with fewer reads than
        there are workers cannot reach all of them, and the workers that miss it
        fall back to emitting single-class batches -- silently undoing exactly
        what stratification is for.
        """
        workers = max(1, self.num_workers)
        if not (self.stratify_samples and len(self.reads_by_sample) > 1 and workers > 1):
            return
        short = {n: len(u) for n, u in self.reads_by_sample.items() if len(u) < workers}
        if short:
            warnings.warn(
                f"{short} read(s) available for sample(s) {sorted(short)} but "
                f"{workers} dataloader workers: those samples cannot reach every "
                "worker, so some workers will emit single-class batches. Lower "
                "training.dataloader.row_groups_per_read (more, smaller reads) or "
                "num_dataloader_workers.",
                stacklevel=2,
            )

    @staticmethod
    def _build_read_units(
        row_groups: Sequence[ig.RowGroup], row_groups_per_read: int
    ) -> list[tuple[str, list[int], int]]:
        """
        Group row groups into (filename, row_group_indices, num_rows) reads.

        Row groups are batched per file in ascending index order, up to
        `row_groups_per_read` each. Contiguity is deliberately NOT required:
        `get_dataset_rowgroups` shuffles and then splits train/val, so a train
        shard is a random ~87% subset whose indices have gaps every ~8 entries.
        Insisting on consecutive runs would cap reads at that gap spacing and
        undo the coalescing entirely. pyarrow accepts an arbitrary index list,
        and ascending order keeps enough locality; the cost being amortised here
        is the per-call Parquet footer parse, not seek time.
        """
        by_file: dict[str, list[ig.RowGroup]] = {}
        for rg in row_groups:
            by_file.setdefault(rg.filename, []).append(rg)

        units: list[tuple[str, list[int], int]] = []
        for filename, groups in by_file.items():
            groups.sort(key=lambda rg: rg.row_group)
            for start in range(0, len(groups), row_groups_per_read):
                block = groups[start : start + row_groups_per_read]
                units.append(
                    (
                        filename,
                        [rg.row_group for rg in block],
                        sum(rg.num_rows for rg in block),
                    )
                )
        return units

    def _shard_reads(self, worker_id: int, num_workers: int) -> list:
        """
        The reads one worker is responsible for.

        Strided WITHIN each sample rather than over the flat list. Striding the
        flat list would already balance the row counts, but it cannot promise
        that a worker receives any read of a given sample -- and a worker that
        holds only background can only ever emit background batches, which is
        exactly what stratification exists to prevent. Per-sample striding gives
        every worker the same class mix as the dataset, to within one read.

        Contiguous slicing is still avoided: with ceil() the last worker gets a
        short or empty shard while the others do a full share, so the epoch is
        paced by the slowest.
        """
        if num_workers <= 1:
            return list(self.read_units)
        shard: list = []
        for name in sorted(self.reads_by_sample):
            shard.extend(self.reads_by_sample[name][worker_id::num_workers])
        return shard

    def __len__(self):
        """
        Exact number of batches this dataset yields.

        Every read in a worker's shard is consumed exactly once and every batch
        is full except the shard's last, so a worker emits
        ceil(rows_in_its_shard / batch_size) batches. That holds for both the
        stratified and the plain path -- they differ in how jets are ordered,
        not in how many there are -- and sharding is deterministic, so this is
        exact rather than an estimate.

        Exactness matters beyond cosmetics. With `val_check_interval` unset,
        Lightning sets val_check_batch = len(dataloader) and triggers
        end-of-epoch validation via `(batch_idx + 1) % val_check_batch == 0`,
        overwriting its own is_last_batch default. An over-estimate here means
        that condition never fires and validation is silently skipped forever.
        """
        num_workers = max(1, self.num_workers)
        total = 0
        for worker in range(num_workers):
            rows = sum(num_rows for _, _, num_rows in self._shard_reads(worker, num_workers))
            if rows:
                total += math.ceil(rows / self.batch_size)
        return total

    @staticmethod
    def _allocate(batch_size: int, remaining: dict[str, int]) -> dict[str, int]:
        """
        Split one batch across samples in proportion to what each has left.

        Proportional to the REMAINING rows, not to the dataset totals, so the
        mix stays representative as samples drain at different rates and the
        last batches are not suddenly single-class. Largest-remainder rounding
        makes the parts sum exactly to the batch size; a sample is never asked
        for more than it still holds.
        """
        left = sum(remaining.values())
        target = min(batch_size, left)
        exact = {n: target * r / left for n, r in remaining.items() if r > 0}
        out = {n: min(int(v), remaining[n]) for n, v in exact.items()}
        short = target - sum(out.values())
        order = sorted(exact, key=lambda n: exact[n] - int(exact[n]), reverse=True)
        while short > 0:
            progressed = False
            for name in order:
                if short == 0:
                    break
                if out[name] < remaining[name]:
                    out[name] += 1
                    short -= 1
                    progressed = True
            if not progressed:  # every sample is exhausted; nothing left to give
                break
        return {n: k for n, k in out.items() if k > 0}

    def build_tensors(self, data: ak.Array):
        if self.cfg.dataset.get("sort_by_pt", False):
            data = sort_candidates_by_pt(data)
        max_cands = self.cfg.dataset.max_cands
        eps = 1e-6

        # ------------------------------------------------------------------
        # Helper: pad jagged awkward array → dense float32 [N, max_cands]
        # ------------------------------------------------------------------
        def pad_cand(arr, fill=0.0):
            return ak.to_numpy(
                ak.fill_none(ak.pad_none(arr, max_cands, clip=True), fill)
            ).astype(np.float32)

        # ------------------------------------------------------------------
        # Candidate p4 components: stored as (rho=pt, eta, phi, t=energy)
        # All other candidate fields — one padded extraction each
        # ------------------------------------------------------------------
        cand_pt = pad_cand(p4_field(data.reco_cand_p4s, "pt"))  # [N, max_cands]
        cand_eta = pad_cand(p4_field(data.reco_cand_p4s, "eta"))
        cand_phi = pad_cand(p4_field(data.reco_cand_p4s, "phi"))
        cand_en = pad_cand(p4_field(data.reco_cand_p4s, "energy"))  # energy
        cand_charge = pad_cand(data.reco_cand_charges)
        cand_pdg_abs = pad_cand(abs(data.reco_cand_pdgs))
        cand_dz = np.abs(pad_cand(data.reco_cand_signed_dz))
        cand_dz_err = pad_cand(data.reco_cand_dz_error)
        cand_dxy = np.abs(pad_cand(data.reco_cand_signed_dxy))
        cand_dxy_err = pad_cand(data.reco_cand_dxy_error)

        # Mask: True = real particle, False = padding  [N, max_cands]
        lengths = np.minimum(ak.to_numpy(ak.num(data.reco_cand_pdgs)), max_cands)
        mask_np = np.arange(max_cands)[None, :] < lengths[:, None]

        # Scalar jet p4s — read raw fields directly, no reinitialize_p4
        jet_pt = ak.to_numpy(p4_field(data.reco_jet_p4, "pt")).astype(np.float32)  # [N]
        jet_eta = ak.to_numpy(p4_field(data.reco_jet_p4, "eta")).astype(np.float32)
        jet_phi = ak.to_numpy(p4_field(data.reco_jet_p4, "phi")).astype(np.float32)
        jet_en = ak.to_numpy(p4_field(data.reco_jet_p4, "energy")).astype(np.float32)

        _pt_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "pt")).astype(np.float32)
        _eta_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "eta")).astype(np.float32)
        _phi_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "phi")).astype(np.float32)
        _energy_gen = ak.to_numpy(p4_field(data.gen_jet_tau_p4, "energy")).astype(np.float32)

        _pt_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "pt")).astype(np.float32)
        _eta_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "eta")).astype(np.float32)
        _phi_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "phi")).astype(np.float32)
        _energy_gen_jet = ak.to_numpy(p4_field(data.gen_jet_p4, "energy")).astype(np.float32)

        # ------------------------------------------------------------------
        # Compute 17 ParticleTransformer features in numpy (zero awkward)
        # ParticleTransformer features from https://arxiv.org/pdf/2202.03772, table 2
        # Broadcast jet scalars [N] → [N, 1] against candidates [N, max_cands]
        # ------------------------------------------------------------------
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

        isElectron = (cand_pdg_abs == 11).astype(np.float32)
        isMuon = (cand_pdg_abs == 13).astype(np.float32)
        isPhoton = (cand_pdg_abs == 22).astype(np.float32)
        isChargedHadron = (cand_pdg_abs == 211).astype(np.float32)
        isNeutralHadron = (cand_pdg_abs == 130).astype(np.float32)

        # Stack → [N, 17, max_cands], zero padded slots, fix nan/inf
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
                isElectron,
                isMuon,
                isPhoton,
                isChargedHadron,
                isNeutralHadron,
                cand_dz,
                cand_dz_err,
                cand_dxy,
                cand_dxy_err,
            ],
            axis=1,
        )  # [N, 17, max_cands]
        cand_features_np *= mask_np[:, None, :]
        np.nan_to_num(cand_features_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Cand kinematics: (px, py, pz, energy) → [N, 4, max_cands]
        cand_px = cand_pt * np.cos(cand_phi)
        cand_py = cand_pt * np.sin(cand_phi)
        cand_pz = cand_pt * np.sinh(cand_eta)
        cand_kinematics_np = np.stack([cand_px, cand_py, cand_pz, cand_en], axis=1)
        cand_kinematics_np *= mask_np[:, None, :]

        # ------------------------------------------------------------------
        # Weights, decay mode, charge
        # ------------------------------------------------------------------
        if "cls_weight" not in data.fields:
            weight_tensors = torch.ones(len(data), dtype=torch.float32)
        else:
            weight_tensors = torch.from_numpy(
                ak.to_numpy(data.cls_weight).astype(np.float32)
            )

        gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
        reduced_gen_decay_modes = g.get_reduced_decaymodes(gen_jet_tau_decaymode)
        ohe_prepared_decay_modes = g.prepare_one_hot_encoding(reduced_gen_decay_modes)
        gen_jet_tau_decaymode_reduced = torch.from_numpy(
            ohe_prepared_decay_modes.astype(np.int64)
        )
        gen_jet_tau_decaymode_ohe = torch.nn.functional.one_hot(
            gen_jet_tau_decaymode_reduced, 6
        ).float()
        gen_jet_tau_decaymode_exists = torch.from_numpy(
            (gen_jet_tau_decaymode != -1).astype(np.int64)
        )
        charge_tensor = torch.from_numpy(
            (ak.to_numpy(data.gen_jet_tau_charge).astype(np.int32) == 1).astype(
                np.float32
            )
        )

        # ------------------------------------------------------------------
        # Kinematics regression targets (pure numpy, no reinitialize_p4)
        # ------------------------------------------------------------------
        _deta = _eta_gen - jet_eta
        _dphi_raw = _phi_gen - jet_phi
        _dphi = np.arctan2(np.sin(_dphi_raw), np.cos(_dphi_raw))
        _vis_pt_ratio = np.maximum(_pt_gen / np.maximum(jet_pt, eps), eps)
        # m^2 = E^2 - pt^2 * cosh^2(eta)
        _mass_gen = np.sqrt(
            np.maximum(_energy_gen**2 - (_pt_gen * np.cosh(_eta_gen)) ** 2, 0.0)
        )
        _mass_reco = np.sqrt(
            np.maximum(jet_en**2 - (jet_pt * np.cosh(jet_eta)) ** 2, 0.0)
        )
        _vis_m_ratio = np.maximum(_mass_gen / np.maximum(_mass_reco, eps), eps)
        # Clamp log-ratio targets to ±5 (≈ factor-of-150 correction).
        # Without this, massless reco jets give log(_mass_gen/eps) ≈ 14, which
        # dominates the loss and causes GradNorm to suppress the kin head weight.
        _LOG_CLAMP = 5.0
        kinematics_tensor = torch.from_numpy(
            np.stack(
                [
                    np.clip(np.log(_vis_pt_ratio), -_LOG_CLAMP, _LOG_CLAMP),
                    _deta,
                    np.sin(_dphi),
                    np.cos(_dphi),
                    np.clip(np.log(_vis_m_ratio), -_LOG_CLAMP, _LOG_CLAMP),
                ],
                axis=-1,
            )
        )

        return self._scaled((
            torch.from_numpy(cand_features_np),
            torch.from_numpy(cand_kinematics_np),
            {
                "kinematics": kinematics_tensor.float(),
                "decay_mode": gen_jet_tau_decaymode_ohe.float(),
                "charge": charge_tensor.float(),
                "is_tau": gen_jet_tau_decaymode_exists.long(),
            },
            torch.from_numpy(mask_np).unsqueeze(1),  # [N, 1, max_cands]
            weight_tensors.float(),
            {
                "pt": torch.from_numpy(_pt_gen),
                "eta": torch.from_numpy(_eta_gen),
                "phi": torch.from_numpy(_phi_gen),
                "energy": torch.from_numpy(_energy_gen),
            },
            {
                "pt": torch.from_numpy(jet_pt),
                "eta": torch.from_numpy(jet_eta),
                "phi": torch.from_numpy(jet_phi),
                "energy": torch.from_numpy(jet_en),
            },
            {
                "pt": torch.from_numpy(_pt_gen_jet),
                "eta": torch.from_numpy(_eta_gen_jet),
                "phi": torch.from_numpy(_phi_gen_jet),
                "energy": torch.from_numpy(_energy_gen_jet),
            },
        ))



    def _parquet_handle(self, filename: str):
        """
        Return a cached pyarrow handle for `filename`.

        `ak.from_parquet(path, row_groups=...)` re-opens the file and re-parses
        the entire Parquet footer on every call, which with tens of thousands of
        row groups costs far more than the rows being read. A ParquetFile holds
        the parsed footer, so keeping one per file turns that into a one-off cost
        per worker.

        Handles are opened lazily here rather than in __init__ because __init__
        runs in the parent process and the dataset is pickled out to the workers;
        an open file handle must not cross that boundary.
        """
        import pyarrow.parquet as pq

        if self._handles is None:
            self._handles = {}
        handle = self._handles.get(filename)
        if handle is None:
            handle = pq.ParquetFile(filename)
            self._handles[filename] = handle
        return handle

    def _load_read_unit(self, read_unit):
        filename, row_group_indices, _ = read_unit
        if self.cache_parquet_handles:
            table = self._parquet_handle(filename).read_row_groups(
                row_group_indices, columns=self._NEEDED_COLUMNS
            )
            data = ak.from_arrow(table)
            del table
        else:
            data = ak.from_parquet(
                filename,
                row_groups=row_group_indices,
                columns=self._NEEDED_COLUMNS,
            )
        tensors = self.build_tensors(data)
        del data
        return tensors

    @staticmethod
    def _concat_tensors(parts: list[tuple]):
        """Concatenate several build_tensors() outputs along the jet axis."""
        if len(parts) == 1:
            return parts[0]
        def _cat(tensors, label):
            shapes = {t.shape[1:] for t in tensors}
            if len(shapes) > 1:
                raise RuntimeError(
                    f"Cannot concatenate '{label}' across reads: trailing shapes "
                    f"differ ({sorted(str(s) for s in shapes)}). All reads must "
                    "agree on every axis but the jet axis; check that "
                    "dataset.max_tau_daughters is set so signal and background "
                    "produce the same number of daughter slots."
                )
            return torch.cat(tensors, dim=0)

        out = []
        for field in range(len(parts[0])):
            if isinstance(parts[0][field], dict):
                out.append(
                    {
                        k: _cat([p[field][k] for p in parts], f"{field}.{k}")
                        for k in parts[0][field]
                    }
                )
            else:
                out.append(_cat([p[field] for p in parts], str(field)))
        return tuple(out)

    @staticmethod
    def _take(tensors: tuple, idx):
        return tuple(
            {k: v[idx] for k, v in t.items()} if isinstance(t, dict) else t[idx]
            for t in tensors
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            reads_to_process = self._shard_reads(0, 1)
        else:
            reads_to_process = self._shard_reads(worker_info.id, worker_info.num_workers)

        samples = {sample_name(unit[0]) for unit in reads_to_process}
        if self.stratify_samples and len(samples) > 1:
            yield from self._iter_stratified(reads_to_process)
        else:
            yield from self._iter_chunked(reads_to_process)

    def _iter_stratified(self, reads_to_process):
        """
        Emit batches composed from every sample at once.

        A read covers one file and therefore one class, so concatenating whole
        reads and shuffling inside the result only mixes classes when the reads
        that happened to land together came from different files. With an
        unbalanced file count that frequently fails: for a fraction f of reads
        in one class, a chunk of k reads is single-class with probability
        f^k + (1-f)^k, which at f=7/8 and k=4 is about 59%. The consequence is
        runs of tens of consecutive batches carrying one label, during which the
        tagging head simply drifts towards that label -- its loss falls to ~0
        inside a run, spikes on the flip, and the epoch mean carries no signal.

        Here each sample instead keeps its own buffer and every batch takes a
        share of each, proportional to what that sample has left. Both classes
        are then present in every batch by construction, whatever the file
        ratio, and the epoch composition is untouched: each read is still
        consumed exactly once, so `__len__` is unchanged.

        The memory budget is unchanged too. `mixing_reads` reads stay resident
        in total, now split across the samples rather than possibly all being
        the same class.
        """
        rng = self._epoch_rng()
        queues: dict[str, list] = {}
        for unit in reads_to_process:
            queues.setdefault(sample_name(unit[0]), []).append(unit)
        if self.shuffle:
            for units in queues.values():
                rng.shuffle(units)

        reads_per_refill = max(1, self.mixing_reads // len(queues))
        remaining = {name: sum(u[2] for u in units) for name, units in queues.items()}
        buffers: dict[str, list] = {}  # sample -> [tensors, cursor]

        def refill(name: str) -> bool:
            block = queues[name][:reads_per_refill]
            del queues[name][:reads_per_refill]
            if not block:
                return False
            tensors = self._concat_tensors([self._load_read_unit(u) for u in block])
            if self.shuffle:
                tensors = self._take(tensors, torch.randperm(tensors[0].shape[0]))
            # Rebinding drops the previous buffer, so one chunk per sample is
            # resident at a time.
            buffers[name] = [tensors, 0]
            return True

        while sum(remaining.values()) > 0:
            parts = []
            for name, wanted in self._allocate(self.batch_size, remaining).items():
                while wanted > 0:
                    buffer = buffers.get(name)
                    if buffer is None or buffer[1] >= buffer[0][0].shape[0]:
                        if not refill(name):
                            # Queue empty before the row count said so: stop
                            # asking this sample rather than spinning.
                            remaining[name] = 0
                            break
                        buffer = buffers[name]
                    tensors, cursor = buffer
                    take = min(wanted, tensors[0].shape[0] - cursor)
                    parts.append(self._take(tensors, slice(cursor, cursor + take)))
                    buffer[1] = cursor + take
                    remaining[name] -= take
                    wanted -= take
            if not parts:
                break
            batch = self._concat_tensors(parts)
            if self.shuffle:
                # Only cosmetic -- every consumer is permutation invariant --
                # but it keeps a truncated batch from being one class.
                batch = self._take(batch, torch.randperm(batch[0].shape[0]))
            yield batch

    def _iter_chunked(self, reads_to_process):
        """
        Emit batches by concatenating whole reads, preserving read order.

        Used when there is only one sample to draw from, and for the test and
        predict splits where the emission order is meaningful.
        """
        if self.shuffle:
            self._epoch_rng().shuffle(reads_to_process)

        # Rows left over from a chunk are carried into the next one instead of
        # being emitted as a short batch. That keeps every batch full except the
        # last of the shard, which is what makes __len__ exact.
        carry = None
        for start_read in range(0, len(reads_to_process), self.mixing_reads):
            chunk = reads_to_process[start_read : start_read + self.mixing_reads]
            tensors = self._concat_tensors([self._load_read_unit(u) for u in chunk])
            if carry is not None:
                tensors = self._concat_tensors([carry, tensors])
                carry = None
            n_rows = tensors[0].shape[0]

            if self.shuffle:
                tensors = self._take(tensors, torch.randperm(n_rows))

            n_full = (n_rows // self.batch_size) * self.batch_size
            for start in range(0, n_full, self.batch_size):
                yield self._take(tensors, slice(start, start + self.batch_size))
            if n_full < n_rows:
                carry = self._take(tensors, slice(n_full, n_rows))

        if carry is not None and carry[0].shape[0] > 0:
            yield carry


class ParTDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        debug_run: bool = False,
    ):
        """Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.

        """
        self.cfg = cfg
        use_bkg = (cfg.training.model.task == "is_tau") or (
            cfg.training.model.name == "MultiParTau"
        )
        self.debug_run = debug_run
        self.sample = "z" if not use_bkg else "*"
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_row_groups = 2 if debug_run else None
        self.save_hyperparameters()
        super().__init__()

    _sample_name = staticmethod(sample_name)
    # The dataset class this module instantiates. A subclass points it at a
    # dataset with different targets and keeps everything else.
    dataset_cls = ParticleTransformerDataset

    def _select_row_groups(
        self, row_groups: list, rng: np.random.Generator
    ) -> list:
        """
        Apply `cfg.dataset.max_jets_per_sample` to a flat row-group list.

        Selection is deterministic given `cfg.dataset.selection_seed`: row groups
        are ordered canonically by (filename, row-group index) before being
        shuffled by the seeded generator, so the same seed and the same files
        always yield the same subset regardless of glob order.

        Row groups are the smallest readable unit, so the kept jet count
        overshoots the limit by at most one row group.
        """
        limits = self.cfg.dataset.get("max_jets_per_sample", None)

        by_sample: dict[str, list] = {}
        for row_group in row_groups:
            by_sample.setdefault(self._sample_name(row_group.filename), []).append(
                row_group
            )

        selected = []
        for sample in sorted(by_sample):
            groups = sorted(
                by_sample[sample], key=lambda rg: (rg.filename, rg.row_group)
            )
            rng.shuffle(groups)

            limit = None if limits is None else limits.get(sample, None)
            available = sum(rg.num_rows for rg in groups)
            if limit is None:
                kept, n_jets = groups, available
                note = "all"
            else:
                kept, n_jets = [], 0
                for row_group in groups:
                    if n_jets >= int(limit):
                        break
                    kept.append(row_group)
                    n_jets += row_group.num_rows
                note = f"limit {int(limit):,}"
            print(
                f"[dataset] {sample}: {n_jets:,} / {available:,} jets "
                f"({len(kept):,} / {len(groups):,} row groups, {note})",
                flush=True,
            )
            selected.extend(kept)
        return selected

    def _resolve_input_paths(self, pattern: str, dataset_type: str) -> list:
        """
        Glob `pattern` and fail loudly if it matches nothing.

        An empty match previously produced a dataset of 0 jets and training
        continued, so the only symptom was "There are 0 jets in the dataset"
        followed by a job that appeared to hang. The three usual causes are a
        data_dir that is not bind-mounted into the container, filenames that do
        not contain "_train"/"_test", and files sitting one directory deeper.
        """
        paths = sorted(glob.glob(pattern))
        if paths:
            # Logged on success too: a silently wrong data_dir is otherwise only
            # visible as a surprising jet count much further down.
            print(
                f"[dataset] {dataset_type}: {len(paths):,} file(s) matched "
                f"{pattern}",
                flush=True,
            )
            return paths

        data_dir = os.path.dirname(pattern)
        lines = [
            f"No {dataset_type} files matched: {pattern}",
            f"  dataset.data_dir      : {self.cfg.dataset.data_dir}",
            f"  directory exists      : {os.path.isdir(data_dir)}",
        ]
        if os.path.isdir(data_dir):
            everything = sorted(os.listdir(data_dir))
            parquet = [f for f in everything if f.endswith(".parquet")]
            subdirs = [f for f in everything if os.path.isdir(os.path.join(data_dir, f))]
            lines += [
                f"  entries in directory  : {len(everything)}",
                f"  .parquet files there  : {len(parquet)}",
                f"  first few names       : {everything[:8]}",
                f"  subdirectories        : {subdirs[:8]}",
                "",
                "Files are matched as '{sample}_" + dataset_type + "*.parquet' with "
                f"sample='{self.sample}', so a name must contain '_{dataset_type}'.",
                "The sample label is the part before '_" + dataset_type + "', and it is "
                "what dataset.max_jets_per_sample keys on.",
            ]
        else:
            lines += [
                "",
                "The directory is not visible from inside the container. Check that "
                "it is covered by a -B bind mount in run-lumi.sh / run.sh.",
            ]
        raise FileNotFoundError("\n".join(lines))

    def get_dataset_rowgroups(self, dataset_type: str):
        if dataset_type == "test":
            test_paths_wcp = os.path.join(
                os.path.expanduser(os.path.expandvars(self.cfg.dataset.data_dir)),
                f"{self.sample}_test*.parquet",
            )
            test_paths = self._resolve_input_paths(test_paths_wcp, "test")
            test_rowgroups = ig.get_row_groups(input_paths=test_paths)
            # max_jets_per_sample is deliberately NOT applied here: truncating the
            # evaluation set would silently change every reported metric.
            np.random.default_rng(
                int(self.cfg.dataset.get("selection_seed", 42))
            ).shuffle(test_rowgroups)
            return test_rowgroups
        elif dataset_type == "train":
            total = sum(
                [
                    self.cfg.dataset.relative_sizes[dataset]
                    for dataset in ["train", "val"]
                ]
            )
            fractions = {
                dataset: self.cfg.dataset.relative_sizes[dataset] / total
                for dataset in ["train", "val"]
            }
            train_paths_wcp = os.path.join(
                os.path.expanduser(os.path.expandvars(self.cfg.dataset.data_dir)),
                f"{self.sample}_train*.parquet",
            )
            train_paths = self._resolve_input_paths(train_paths_wcp, "train")
            # A dedicated generator rather than the global numpy state, so the
            # train/val split and the per-sample subsampling cannot be perturbed
            # by unrelated random draws elsewhere in the process.
            rng = np.random.default_rng(
                int(self.cfg.dataset.get("selection_seed", 42))
            )
            all_train_rowgroups = self._select_row_groups(
                ig.get_row_groups(input_paths=train_paths), rng
            )
            rng.shuffle(all_train_rowgroups)
            n_train_rowgroups = int(len(all_train_rowgroups) * fractions["train"])
            train_rowgroups = all_train_rowgroups[:n_train_rowgroups]
            val_rowgroups = all_train_rowgroups[n_train_rowgroups:]
            return train_rowgroups, val_rowgroups
        else:
            return []


    def _dataset_kwargs(self, num_workers: int) -> dict:
        """Read/batching options every dataset of this module is built with."""
        dl_cfg = self.cfg.training.dataloader
        return {
            "row_groups_per_read": int(dl_cfg.get("row_groups_per_read", 1)),
            "mixing_reads": int(dl_cfg.get("mixing_reads", 1)),
            "cache_parquet_handles": bool(dl_cfg.get("cache_parquet_handles", True)),
            "num_workers": int(num_workers),
        }

    def make_fit_dataset(self, row_groups):
        """
        Dataset the scaler-fitting pass reads through.

        The same dataset class the module trains with, so the fit sees exactly
        the features training will see, and stratified: a read covers one file
        and therefore one class, so stopping after `fit_jets` jets without
        stratifying could take the whole subsample from background alone and
        bias every feature mean the fit produces. Not shuffled, so the scaler
        is reproducible.
        """
        return self.dataset_cls(
            row_groups=list(row_groups),
            cfg=self.cfg,
            batch_size=self.cfg.training.dataloader.batch_size,
            shuffle=False,
            stratify_samples=True,
            **self._dataset_kwargs(num_workers=0),
        )

    def resolve_input_scaler(self, row_groups, stage: str):
        """
        Return the `tensors -> tensors` scaler for this run, fitting it if needed.

        An existing .npz is reused rather than refitted, so a resumed run, the
        validation split and later evaluation all standardise with the SAME
        constants -- refitting per run would silently shift the inputs a trained
        checkpoint expects. Fitting only ever happens for the fit stage; test and
        predict require the file to already exist.

        The fit reads the training row groups itself, through an ordinary
        single-process dataset, and stops after `training.input_scaling.fit_jets`
        jets.
        """
        if not scaling.scaling_enabled(self.cfg):
            return None

        path = scaling.scaler_path(self.cfg)
        if not os.path.exists(path):
            if stage != "fit":
                raise RuntimeError(
                    f"Input scaling is enabled but no scaler exists at {path}. "
                    "Run training first, or point training.input_scaling.scaler_path "
                    "at the scaler that was fitted for this checkpoint."
                )
            fit_jets = int(
                self.cfg.training.input_scaling.get("fit_jets", 500_000)
            )
            print(
                f"[input scaling] No scaler at {path}; fitting on up to "
                f"{fit_jets:,} training jets.",
                flush=True,
            )
            scaling.fit_scaler(
                iter(self.make_fit_dataset(row_groups)), self.cfg, max_jets=fit_jets
            )
        return scaling.make_input_scaler(self.cfg)


    def setup(self, stage: str) -> None:
        batch_size = (
            self.cfg.training.dataloader.batch_size if not self.debug_run else 512
        )
        dl_cfg = self.cfg.training.dataloader
        if stage == "fit":
            train_row_groups, val_row_groups = self.get_dataset_rowgroups(
                dataset_type="train"
            )
            # The dataset needs the worker count actually used by the DataLoader,
            # so __len__ matches how the shards are really split.
            n_workers = (
                0
                if self.debug_run
                else resolve_num_workers(dl_cfg.num_dataloader_workers)
            )
            kwargs = self._dataset_kwargs(n_workers)
            self.train_dataset = self.dataset_cls(
                row_groups=train_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=True,
                **kwargs,
            )
            self.val_dataset = self.dataset_cls(
                row_groups=val_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=False,
                **kwargs,
            )
            scaler = self.resolve_input_scaler(train_row_groups, "fit")
            self.train_dataset.set_input_scaler(scaler)
            self.val_dataset.set_input_scaler(scaler)
            # batch_size=None: the dataset yields pre-batched slices, so the
            # DataLoader does no collation. loader_kwargs drops the arguments
            # that are only legal with workers.
            loader_args = loader_kwargs(n_workers, dl_cfg.prefetch_factor, self.debug_run)
            self.train_loader = DataLoader(self.train_dataset, batch_size=None, **loader_args)
            self.val_loader = DataLoader(self.val_dataset, batch_size=None, **loader_args)
        elif stage == "test" or stage == "predict":
            test_row_groups = self.get_dataset_rowgroups(dataset_type="test")
            n_workers = resolve_num_workers(dl_cfg.num_dataloader_workers)
            self.test_dataset = self.dataset_cls(
                row_groups=test_row_groups,
                cfg=self.cfg,
                batch_size=batch_size,
                shuffle=False,
                # Evaluation reads the emission order as meaningful, and a
                # gradient-free pass has nothing to gain from stratifying.
                stratify_samples=False,
                **self._dataset_kwargs(n_workers),
            )
            self.test_dataset.set_input_scaler(
                self.resolve_input_scaler(test_row_groups, stage)
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=None,
                **loader_kwargs(n_workers, dl_cfg.prefetch_factor, False),
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader
