import math

import torch


DECOMPOSITION_REASONS = (
    "valid", "empty", "invalid_parent", "invalid_daughters",
    "mass_budget", "zero_momenta", "numerical_failure", "not_tagged",
)


def encode_kinematics(p4: torch.Tensor, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode Cartesian [px, py, pz, energy] relative to the reconstructed jet."""
    values = {}
    for name in ("pt", "eta", "phi", "energy"):
        value = reference[name].to(p4)
        while value.ndim < p4.ndim - 1:
            value = value.unsqueeze(-1)
        values[name] = value
    px, py, pz, energy = p4.unbind(-1)
    pt = torch.hypot(px, py).clamp_min(1e-12)
    momentum = torch.linalg.vector_norm(p4[..., :3], dim=-1)
    mass = ((energy - momentum) * (energy + momentum)).clamp_min(1e-12).sqrt()
    reference_momentum = values["pt"] * values["eta"].cosh()
    reference_mass = (
        (values["energy"] - reference_momentum)
        * (values["energy"] + reference_momentum)
    ).clamp_min(1e-12).sqrt()
    delta_phi = torch.atan2(py, px) - values["phi"]
    return torch.stack((
        torch.log(pt / values["pt"].clamp_min(1e-12)),
        torch.asinh(pz / pt) - values["eta"],
        delta_phi.sin(), delta_phi.cos(), torch.log(mass / reference_mass),
    ), dim=-1)


@torch.no_grad()
def conserve_daughters(
    raw_p4: torch.Tensor,
    parent_p4: torch.Tensor,
    selected: torch.Tensor,
    *,
    iterations: int = 48,
    atol: float = 1e-8,
    rtol: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Conserve the parent by centering and scaling daughters in its rest frame.

    All p4 tensors use [px, py, pz, energy]. Invalid events retain raw daughters.
    Integer reasons index DECOMPOSITION_REASONS. This inference-only operation
    deliberately returns float64 four-vectors to preserve boosted mass shells.
    """
    if iterations < 1 or atol < 0 or rtol < 0:
        raise ValueError("Invalid conservation solver settings.")
    raw = raw_p4.double()
    parent = parent_p4.double()
    selected = selected.bool()
    count = selected.sum(-1)
    reason = torch.zeros_like(count)
    parent_norm = torch.linalg.vector_norm(parent[:, :3], dim=-1)
    mass_squared = (parent[:, 3] - parent_norm) * (parent[:, 3] + parent_norm)
    parent_ok = torch.isfinite(parent).all(-1) & (parent[:, 3] > 0) & (mass_squared > 0)
    parent_safe = torch.where(parent_ok[:, None], parent, parent.new_tensor([0, 0, 0, 1]))
    parent_mass = torch.where(parent_ok, mass_squared, torch.ones_like(mass_squared)).sqrt()
    daughter_norm = torch.linalg.vector_norm(raw[..., :3], dim=-1)
    daughter_mass_squared = (raw[..., 3] - daughter_norm) * (raw[..., 3] + daughter_norm)
    daughter_ok = torch.isfinite(raw).all(-1) & (raw[..., 3] >= 0) & (daughter_mass_squared >= 0)
    bad_daughters = (selected & ~daughter_ok).any(-1)
    safe = torch.where((selected & daughter_ok)[..., None], raw, torch.zeros_like(raw))
    masses = torch.where(selected & daughter_ok, daughter_mass_squared, 0).clamp_min(0).sqrt()
    parent_space = parent_safe[:, None, :3]
    boost = (
        (safe[..., :3] * parent_space).sum(-1)
        / (parent_mass * (parent_safe[:, 3] + parent_mass))[:, None]
        - safe[..., 3] / parent_mass[:, None]
    )
    rest = safe[..., :3] + boost[..., None] * parent_space
    rest = rest.masked_fill(~selected[..., None], 0)
    center = rest.sum(1) / count.clamp_min(1)[:, None]
    centered = (rest - center[:, None]).masked_fill(~selected[..., None], 0)
    norms = torch.linalg.vector_norm(centered, dim=-1)
    mass_sum = masses.sum(-1)
    budget_ok = mass_sum <= parent_mass
    at_threshold = mass_sum == parent_mass
    nonzero = norms.sum(-1) > 0
    lower = torch.zeros_like(parent_mass)
    upper = parent_mass / torch.where(nonzero, norms.sum(-1), 1)
    solvable = parent_ok & ~bad_daughters & (count > 1) & budget_ok & (nonzero | at_threshold)
    upper = torch.where(solvable & ~at_threshold, upper, 0)
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        energy = torch.hypot(masses, midpoint[:, None] * norms).sum(-1)
        below = energy < parent_mass
        lower = torch.where(below, midpoint, lower)
        upper = torch.where(below, upper, midpoint)
    scale = (lower + upper) / 2
    rest_space = scale[:, None, None] * centered
    rest_energy = torch.hypot(masses, scale[:, None] * norms)
    dot = (rest_space * parent_space).sum(-1)
    lab_space = rest_space + (
        dot / (parent_mass * (parent_safe[:, 3] + parent_mass))[:, None]
        + rest_energy / parent_mass[:, None]
    )[..., None] * parent_space
    lab_energy = (parent_safe[:, 3, None] * rest_energy + dot) / parent_mass[:, None]
    candidate = torch.cat((lab_space, lab_energy[..., None]), dim=-1)
    candidate = candidate.masked_fill(~selected[..., None], 0)
    closed = torch.isclose(candidate.sum(1), parent, atol=atol, rtol=rtol).all(-1)
    shell = candidate[..., 3].square() - candidate[..., :3].square().sum(-1)
    shell_ok = (torch.isclose(shell, masses.square(), atol=atol, rtol=rtol) | ~selected).all(-1)
    numerical_ok = closed & shell_ok & torch.isfinite(candidate).all(dim=(1, 2))
    reason = torch.where((count > 1) & ~numerical_ok, 6, reason)
    reason = torch.where((count > 1) & ~nonzero & ~at_threshold, 5, reason)
    reason = torch.where((count > 1) & ~budget_ok, 4, reason)
    reason = torch.where((count > 1) & bad_daughters, 3, reason)
    reason = torch.where(~parent_ok, 2, reason)
    reason = torch.where(count == 0, 1, reason)
    valid = reason == 0
    corrected = torch.where((solvable & numerical_ok)[:, None, None] & selected[..., None], candidate, raw)
    corrected = torch.where(((count == 1) & parent_ok)[:, None, None] & selected[..., None], parent[:, None], corrected)
    scale = torch.where((count == 1) & parent_ok, 1, scale)
    scale = torch.where(valid, scale, torch.full_like(scale, float("nan")))
    return {"p4": corrected, "raw_p4": raw, "parent_p4": parent,
            "valid": valid, "reason": reason, "scale": scale}


@torch.no_grad()
def reconstruct_daughters(outputs, reference, selected, **solver_options):
    """Decode raw network outputs and apply the common inference correction."""
    references = [reference[name].double() for name in ("pt", "eta", "phi", "energy")]
    raw = decode_kinematics(outputs["pred_kinematics"].double(), *references)
    parent = decode_kinematics(outputs["pred_parent_kinematics"].double(), *references)
    return conserve_daughters(raw, parent, selected, **solver_options)


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