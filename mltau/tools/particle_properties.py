from collections.abc import Mapping
from dataclasses import dataclass

import torch

from mltau.tools.meson_classes import get_meson_classes


@dataclass(frozen=True)
class ParticleProperty:
    charge: int
    mass: float


@dataclass(frozen=True)
class MassOption:
    mass: float
    charge: int
    meson_class: int


def get_particle_properties(configured_properties: Mapping) -> dict[int, ParticleProperty]:
    """Return particle properties indexed by absolute PDG ID."""
    if not isinstance(configured_properties, Mapping):
        raise ValueError("particle_properties must map PDG IDs to charge and mass.")

    properties: dict[int, ParticleProperty] = {}
    for configured_pdg, values in configured_properties.items():
        pdg_id = abs(int(configured_pdg))
        if pdg_id == 0:
            raise ValueError("particle_properties cannot contain PDG ID 0.")
        charge = int(values["charge"])
        mass = float(values["mass"])
        if charge not in {0, 1}:
            raise ValueError(
                f"Particle {pdg_id} has charge={charge}; expected charge 0 or 1."
            )
        if mass <= 0:
            raise ValueError(f"Particle {pdg_id} must have a positive mass.")
        properties[pdg_id] = ParticleProperty(charge, mass)

    if not properties:
        raise ValueError("particle_properties must contain at least one particle.")
    return properties


def get_mass_options(
    configured_groups: Mapping, configured_properties: Mapping
) -> tuple[MassOption, ...]:
    """Return one mass entry per configured PDG ID, in meson-class order.

    Entries are not merged when their masses are close. For example, charged
    and neutral pions keep separate entries even though their masses differ by
    only about 0.005 GeV. Their probabilities may therefore be similar.
    """
    properties = get_particle_properties(configured_properties)
    options: list[MassOption] = []
    for class_index, meson_class in enumerate(get_meson_classes(configured_groups)):
        for pdg_id in meson_class.pdg_ids:
            if pdg_id not in properties:
                raise ValueError(
                    f"Meson class '{meson_class.name}' uses PDG ID {pdg_id}, but "
                    "particle_properties has no entry for it."
                )
            particle = properties[pdg_id]
            options.append(
                MassOption(
                    mass=particle.mass,
                    charge=particle.charge,
                    meson_class=class_index,
                )
            )
    return tuple(options)


def mass_logits(
    mass_output: torch.Tensor,
    mass_options: torch.Tensor,
    mass_width: float,
) -> torch.Tensor:
    """Convert one scalar per query into logits over configured mass entries.

    The formula is ``-0.5 * ((output - mass) / mass_width)^2``. For example, an
    output near 0.14 GeV gives both pion entries larger logits than the kaon and
    eta entries. ``mass_width`` is in GeV and controls how quickly logits fall
    as an entry moves away from the output.
    """
    if mass_width <= 0:
        raise ValueError("mass_width must be positive.")
    return -0.5 * (
        (mass_output.unsqueeze(-1) - mass_options) / mass_width
    ).square()


def mass_probabilities(
    mass_output: torch.Tensor,
    mass_options: torch.Tensor,
    mass_width: float,
) -> torch.Tensor:
    """Return probabilities over fixed masses for each scalar model output.

    Very close entries can both keep substantial probability. This is expected:
    their weighted mass remains close to both entries, while the charge and
    meson-class heads carry the other classification information.
    """
    return torch.softmax(
        mass_logits(mass_output, mass_options, mass_width), dim=-1
    )
