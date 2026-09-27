from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


# Generic hadron IDs used when downstream decay-mode code needs only the
# charged-versus-neutral category, not the predicted meson species.
CHARGED_HADRON_PDG = 211
NEUTRAL_HADRON_PDG = 130


@dataclass(frozen=True)
class MesonClass:
    name: str
    pdg_ids: tuple[int, ...]


def get_meson_classes(configured_groups: Mapping) -> tuple[MesonClass, ...]:
    """Return validated meson classes in configured order."""
    if not isinstance(configured_groups, Mapping):
        raise ValueError(
            "dataset.tau_daughter_pdg_ids must map class names to PDG ID lists."
        )
    classes: list[MesonClass] = []
    seen: set[int] = set()
    for class_name, configured_pdgs in configured_groups.items():
        pdg_ids = tuple(abs(int(pdg_id)) for pdg_id in configured_pdgs)
        if not pdg_ids or 0 in pdg_ids:
            raise ValueError(f"Meson class '{class_name}' must contain nonzero PDG IDs.")
        duplicates = seen.intersection(pdg_ids)
        if duplicates:
            raise ValueError(
                f"PDG IDs {sorted(duplicates)} occur in more than one meson class."
            )
        seen.update(pdg_ids)
        classes.append(MesonClass(str(class_name), pdg_ids))

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