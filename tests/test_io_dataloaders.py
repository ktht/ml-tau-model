"""
Structure and composition checks for the parquet dataloaders.

Runs against whatever `dataset.data_dir` points at, or a directory given as
MLTAU_TEST_DATA_DIR, so the same script works on the cluster and on a laptop
with a small local sample. Both data modules are exercised: the ParT one that
MultiParTau / SingleParTau train with, and the DETR one.

What is asserted:
  - the batch tuple has the eight expected members with the expected shapes;
  - every batch but the last of a worker shard has exactly batch_size jets,
    and the number of batches equals len(dataset);
  - when more than one sample is present, every batch contains both classes.

    ./run.sh python3 tests/test_io_dataloaders.py
    MLTAU_TEST_DATA_DIR=/tmp/newprod python3 tests/test_io_dataloaders.py
"""

import os
import sys
from pathlib import Path

import awkward as ak
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mltau.tools.io.ParT_dataloader import (  # noqa: E402
    ParTDataModule,
    sort_candidates_by_pt,
)
from mltau.tools.io.ParTauDETR_dataloader import (  # noqa: E402
    ParTauDETRDataModule,
    ParticleTransformerDETRDataset,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "mltau" / "config"


def compose(main: str, dataset: str, batch_size: int):
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "training.yaml"),
        OmegaConf.load(CONFIG_DIR / dataset),
        OmegaConf.load(CONFIG_DIR / main),
    )
    cfg.output_dir = os.environ.get("MLTAU_TEST_OUTPUT_DIR", "/tmp/mltau_test_output")
    data_dir = os.environ.get("MLTAU_TEST_DATA_DIR")
    if data_dir:
        cfg.dataset.data_dir = data_dir
    cfg.training.dataloader.batch_size = batch_size
    cfg.training.dataloader.num_dataloader_workers = 0
    cfg.training.dataloader.row_groups_per_read = 4
    cfg.training.dataloader.mixing_reads = 2
    cfg.training.input_scaling.enabled = False
    return cfg


def check_structure(batch, n_features: int, max_cands: int):
    assert len(batch) == 8, len(batch)
    assert batch[0].shape[1:] == (n_features, max_cands), batch[0].shape
    assert batch[1].shape[1:] == (4, max_cands), batch[1].shape
    assert isinstance(batch[2], dict) and "is_tau" in batch[2]
    assert batch[3].shape[1:] == (1, max_cands), batch[3].shape
    assert batch[4].ndim == 1
    for p4 in batch[5:8]:
        assert isinstance(p4, dict) and {"pt", "eta", "phi", "energy"} <= set(p4)


def check_batches(loader, batch_size: int, name: str):
    dataset = loader.dataset
    sizes, mixed, single = [], 0, 0
    for batch in loader:
        n = batch[0].shape[0]
        sizes.append(n)
        is_tau = batch[2]["is_tau"]
        frac = float(is_tau.float().mean())
        if 0.0 < frac < 1.0:
            mixed += 1
        else:
            single += 1
    total = sum(sizes)
    assert len(sizes) == len(dataset), f"{name}: {len(sizes)} batches yielded, len(dataset)={len(dataset)}"
    assert total == dataset.num_rows, f"{name}: {total} jets yielded, dataset has {dataset.num_rows}"
    assert all(s == batch_size for s in sizes[:-1]), f"{name}: short batch before the last: {sizes}"
    multi_sample = len(getattr(dataset, "reads_by_sample", {})) > 1
    if multi_sample and getattr(dataset, "stratify_samples", False):
        assert single == 0, f"{name}: {single} single-class batches with stratification on"
    print(
        f"  {name}: {len(sizes)} batches, {total:,} jets, batch sizes "
        f"{sorted(set(sizes))}, mixed {mixed}, single-class {single}"
    )


def check_tau_daughter_sorting():
    p4 = ak.Array(
        [
            [
                {"pt": 4.0, "eta": 0.1, "phi": 0.2, "mass": 0.14},
                {"pt": 12.0, "eta": 0.2, "phi": 0.3, "mass": 0.14},
                {"pt": 7.0, "eta": 0.3, "phi": 0.4, "mass": 0.14},
            ]
        ]
    )
    pdg = ak.Array([[11, 22, 33]])
    charge = ak.Array([[-1, 0, 1]])

    sorted_p4, sorted_pdg, sorted_charge = (
        ParticleTransformerDETRDataset._sort_tau_daughters_by_pt(p4, pdg, charge)
    )

    assert ak.to_list(sorted_p4.pt) == [[12.0, 7.0, 4.0]]
    assert ak.to_list(sorted_pdg) == [[22, 33, 11]]
    assert ak.to_list(sorted_charge) == [[0, 1, -1]]
    assert ParticleTransformerDETRDataset._pad_jagged(
        sorted_p4.pt, 2
    ).tolist() == [[12.0, 7.0]]
    assert ParticleTransformerDETRDataset._pad_jagged(
        sorted_pdg, 2
    ).tolist() == [[22, 33]]
    assert ParticleTransformerDETRDataset._pad_jagged(
        sorted_charge, 2
    ).tolist() == [[0, 1]]


def check_signed_candidate_sorting():
    data = ak.Array(
        [
            {
                "reco_cand_p4s": [
                    {"pt": 4.0, "eta": 0.1, "phi": 0.2, "mass": 0.14},
                    {"pt": 12.0, "eta": 0.2, "phi": 0.3, "mass": 0.14},
                ],
                "reco_cand_signed_dz": [0.4, -1.2],
                "reco_cand_signed_dxy": [-0.04, 0.12],
            }
        ]
    )
    sorted_data = sort_candidates_by_pt(data)

    assert ak.to_list(sorted_data.reco_cand_signed_dz) == [[-1.2, 0.4]]
    assert ak.to_list(sorted_data.reco_cand_signed_dxy) == [[0.12, -0.04]]


def run(main: str, dataset: str, module_cls, label: str, batch_size: int = 256):
    print(f"[{label}]")
    cfg = compose(main, dataset, batch_size)
    dm = module_cls(cfg=cfg, debug_run=False)
    dm.setup("fit")
    train = dm.train_dataloader()
    check_structure(next(iter(train)), int(cfg.dataset.num_features) if "num_features" in cfg.dataset else 17, int(cfg.dataset.max_cands))
    check_batches(train, batch_size, "train")
    check_batches(dm.val_dataloader(), batch_size, "val")
    print("  OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_tau_daughter_sorting()
    check_signed_candidate_sorting()
    run("main.yaml", "dataset.yaml", ParTDataModule, "ParT data module")
    run("main_ParTauDETR.yaml", "dataset_ParTauDETR.yaml", ParTauDETRDataModule, "DETR data module")
    print("\nAll dataloader tests passed.")
