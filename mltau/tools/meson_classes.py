from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


STORED_LEPTONIC_DECAY_MODE = 16
CHARGED_HADRON_PDG = 211
NEUTRAL_HADRON_PDG = 130
ELECTRON_PDG = 11
MUON_PDG = 13


@dataclass(frozen=True)
class MesonClass:
    name: str
    pdg_ids: tuple[int, ...]
    charges: tuple[int, ...]

    @property
    def representative_pdg(self) -> int:
        """Representative species for decay-mode decoding, preserving leptons."""
        species = set(self.pdg_ids)
        if species.intersection({ELECTRON_PDG, MUON_PDG}):
            if species not in ({ELECTRON_PDG}, {MUON_PDG}) or 0 in self.charges:
                raise ValueError(f"Class '{self.name}' must keep each lepton species separate and charged.")
            return self.pdg_ids[0]
        if set(self.charges) == {0}:
            return NEUTRAL_HADRON_PDG
        if 0 not in self.charges:
            return CHARGED_HADRON_PDG
        raise ValueError(f"Class '{self.name}' mixes neutral and charged particles.")

    @property
    def mass(self) -> float:
        """Representative daughter mass in GeV; hadron classes retain pion masses."""
        return {
            CHARGED_HADRON_PDG: 0.13957,
            NEUTRAL_HADRON_PDG: 0.13498,
            ELECTRON_PDG: 0.000510999,
            MUON_PDG: 0.1056583755,
        }[self.representative_pdg]


def get_meson_classes(configured_groups: Mapping) -> tuple[MesonClass, ...]:
    """Return validated meson classes in configured order."""
    if not isinstance(configured_groups, Mapping):
        raise ValueError(
            "dataset.tau_daughter_pdg_ids must map class names to pdg_ids and charges."
        )
    classes: list[MesonClass] = []
    seen: set[int] = set()
    for class_name, class_cfg in configured_groups.items():
        pdg_ids = tuple(abs(int(pdg_id)) for pdg_id in class_cfg["pdg_ids"])
        charges = tuple(int(charge) for charge in class_cfg["charges"])
        if not pdg_ids or 0 in pdg_ids:
            raise ValueError(f"Meson class '{class_name}' must contain nonzero PDG IDs.")
        if not charges or any(charge not in {-1, 0, 1} for charge in charges):
            raise ValueError(
                f"Meson class '{class_name}' must allow charges from {{-1, 0, 1}}."
            )
        duplicates = seen.intersection(pdg_ids)
        if duplicates:
            raise ValueError(
                f"PDG IDs {sorted(duplicates)} occur in more than one meson class."
            )
        seen.update(pdg_ids)
        classes.append(MesonClass(str(class_name), pdg_ids, charges))

    if not classes:
        raise ValueError("At least one meson class must be configured.")
    return tuple(classes)


def get_meson_class_groups(configured_groups: Mapping) -> tuple[tuple[int, ...], ...]:
    """Return absolute PDG IDs grouped in configured class order."""
    return tuple(cls.pdg_ids for cls in get_meson_classes(configured_groups))


def pdg_to_meson_class_indices(
    raw_pdg: np.ndarray, configured_groups: Mapping
) -> np.ndarray:
    """Map signed PDG IDs to configured meson-class indices; unsupported is -1."""
    groups = get_meson_class_groups(configured_groups)
    class_indices = np.full(raw_pdg.shape, -1, dtype=np.int64)
    pdg_abs = np.abs(raw_pdg.astype(np.int64))
    for class_index, pdg_ids in enumerate(groups):
        class_indices[np.isin(pdg_abs, pdg_ids)] = class_index
    return class_indices