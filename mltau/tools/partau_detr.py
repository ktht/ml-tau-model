import math

import torch


def encode_p4(p4: torch.Tensor, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    """Express physical p4 in the existing jet-relative five-component basis."""
    ref = {name: value.to(p4) for name, value in reference.items()}
    while ref["pt"].ndim < p4.ndim - 1:
        ref = {name: value.unsqueeze(-1) for name, value in ref.items()}
    px, py, pz, energy = p4.unbind(-1)
    pt = (px.square() + py.square()).clamp_min(1e-24).sqrt()
    mass = (energy.square() - p4[..., :3].square().sum(-1)).clamp_min(1e-24).sqrt()
    ref_mass = (ref["energy"].square() - (ref["pt"] * ref["eta"].cosh()).square()).clamp_min(1e-12).sqrt()
    phi = torch.atan2(py, torch.where(pt > 1e-11, px, torch.ones_like(px))) - ref["phi"]
    return torch.stack((
        (pt / ref["pt"].clamp_min(1e-6)).clamp_min(1e-6).log().clamp(-5, 5),
        torch.asinh(pz / pt) - ref["eta"],
        phi.sin(), phi.cos(),
        (mass / ref_mass).clamp_min(1e-6).log().clamp(-5, 5),
    ), dim=-1)


def boost_p4(p4: torch.Tensor, frame: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Boost rest-frame p4 to a future-timelike frame (or back with inverse)."""
    frame_mass = (frame[..., 3].square() - frame[..., :3].square().sum(-1)).clamp_min(1e-24).sqrt()
    boost = frame[..., :3] / frame_mass.unsqueeze(-1)
    if inverse:
        boost = -boost
    gamma = frame[..., 3] / frame_mass
    while boost.ndim < p4.ndim:
        boost = boost.unsqueeze(-2)
        gamma = gamma.unsqueeze(-1)
    projection = (p4[..., :3] * boost).sum(-1)
    momentum = p4[..., :3] + boost * (p4[..., 3] + projection / (gamma + 1)).unsqueeze(-1)
    energy = gamma * p4[..., 3] + projection
    return torch.cat((momentum, energy.unsqueeze(-1)), dim=-1)


def fraction_daughters(
    parent_p4: torch.Tensor, coordinates: torch.Tensor, *, close: bool,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Positive energy fractions and independent subluminal velocities.

    Coordinate 0 is softmaxed over queries, never over the velocity components.
    With active_mask, only active queries participate; empty sets return zero.
    Closure boosts the physical seed system to rest, rescales its invariant
    mass to the parent mass, then boosts to the predicted parent frame.
    Computation uses float64 to protect invariant masses of boosted systems.
    """
    parent = parent_p4.double()
    coordinates = coordinates.double()
    if active_mask is None:
        active_mask = torch.ones_like(coordinates[..., 0], dtype=torch.bool)
    nonempty = active_mask.any(-1, keepdim=True)
    logits = coordinates[..., 0].masked_fill(~active_mask, -torch.inf)
    logits = torch.where(nonempty, logits, torch.zeros_like(logits))
    fractions = logits.softmax(dim=-1) * active_mask
    velocity_coordinates = coordinates[..., 1:].clamp(-100, 100)
    velocity = velocity_coordinates / (1 + velocity_coordinates.square().sum(-1, keepdim=True)).sqrt()
    seed = torch.cat((fractions.unsqueeze(-1) * velocity, fractions.unsqueeze(-1)), dim=-1)
    parent_mass = (parent[..., 3].square() - parent[..., :3].square().sum(-1)).clamp_min(1e-24).sqrt()
    if close:
        total = seed.sum(-2)
        rest = torch.zeros_like(total)
        rest[..., 3] = 1
        total = torch.where(nonempty, total, rest)
        seed = boost_p4(seed, total, inverse=True)
        seed_mass = (total[..., 3].square() - total[..., :3].square().sum(-1)).clamp_min(1e-24).sqrt()
        seed = seed / seed_mass[..., None, None]
    return boost_p4(seed * parent_mass[..., None, None], parent)


def decode_kinematics(
    kinematics: torch.Tensor,
    reference_pt: torch.Tensor,
    reference_eta: torch.Tensor,
    reference_phi: torch.Tensor,
    reference_energy: torch.Tensor,
    *,
    clamp_log_ratios: bool = False,
) -> torch.Tensor:
    """Decode ParTauDETR kinematics into Cartesian four-momenta.

    ``kinematics`` has final dimension
    ``[log_pt_ratio, delta_eta, sin_dphi, cos_dphi, log_mass_ratio]``.
    Reference tensors may omit trailing object dimensions, such as a ``[B]``
    jet reference used with ``[B, Q, 5]`` kinematics.
    """
    for reference in (reference_pt, reference_eta, reference_phi, reference_energy):
        if reference.device != kinematics.device:
            raise ValueError("Kinematics and reference tensors must share a device.")

    def expand_reference(reference: torch.Tensor) -> torch.Tensor:
        while reference.ndim < kinematics.ndim - 1:
            reference = reference.unsqueeze(-1)
        return reference

    reference_pt = expand_reference(reference_pt)
    reference_eta = expand_reference(reference_eta)
    reference_phi = expand_reference(reference_phi)
    reference_energy = expand_reference(reference_energy)

    reference_mass = torch.sqrt(
        torch.clamp(
            reference_energy**2 - (reference_pt * torch.cosh(reference_eta)) ** 2,
            min=1e-12,
        )
    )
    log_pt_ratio = kinematics[..., 0]
    log_mass_ratio = kinematics[..., 4]
    if clamp_log_ratios:
        log_pt_ratio = log_pt_ratio.clamp(-5.0, 5.0)
        log_mass_ratio = log_mass_ratio.clamp(-5.0, 5.0)

    pt = torch.exp(log_pt_ratio) * reference_pt
    eta = kinematics[..., 1] + reference_eta
    if clamp_log_ratios:
        max_abs_eta = math.acosh(math.sqrt(torch.finfo(kinematics.dtype).max))
        eta = eta.clamp(-max_abs_eta, max_abs_eta)
    phi = reference_phi + torch.atan2(kinematics[..., 2], kinematics[..., 3])
    mass = torch.exp(log_mass_ratio) * reference_mass

    return torch.stack(
        [
            pt * torch.cos(phi),
            pt * torch.sin(phi),
            pt * torch.sinh(eta),
            torch.sqrt((pt * torch.cosh(eta)) ** 2 + mass**2),
        ],
        dim=-1,
    )