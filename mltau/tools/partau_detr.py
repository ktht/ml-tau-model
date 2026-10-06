import math

import torch
import torch.nn.functional as F


def tetrahedral_basis(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    ) / math.sqrt(3.0)


def to_tetrahedral(momentum: torch.Tensor) -> torch.Tensor:
    projection = (momentum[..., None, :3] * tetrahedral_basis(momentum)).sum(-1)
    return (momentum[..., 3:4] + projection) / 4


def from_tetrahedral(components: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (3 * (components[..., :, None] * tetrahedral_basis(components)).sum(-2),
         components.sum(-1, keepdim=True)),
        dim=-1,
    )


def decode_fractions(
    logits: torch.Tensor, parent_p4: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """Decode selected daughters; an empty selection returns zero four-vectors."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        masked = logits.float().masked_fill(~selected[..., None], -torch.inf)
        masked = torch.where(selected.any(-1)[:, None, None], masked, torch.zeros_like(masked))
        fractions = masked.softmax(dim=1) * selected[..., None]
        components = to_tetrahedral(parent_p4.float().detach())
        return from_tetrahedral(fractions * components[:, None, :])


def parent_momentum(outputs: dict, reference: dict) -> torch.Tensor:
    with torch.autocast(device_type=outputs["tau_kinematics"].device.type, enabled=False):
        return decode_kinematics(
            outputs["tau_kinematics"].float(),
            *(reference[name].float() for name in ("pt", "eta", "phi", "energy")),
            clamp_log_ratios=True,
        )


def predicted_momenta(outputs: dict, reference: dict, threshold=None):
    parent = parent_momentum(outputs, reference)
    scores = outputs["pred_logits"].float().softmax(-1)[..., 0]
    selected = torch.ones_like(scores, dtype=torch.bool) if threshold is None else scores >= threshold
    return decode_fractions(outputs["pred_fraction_logits"], parent, selected), selected, parent


def momentum_coordinates(momentum: torch.Tensor) -> torch.Tensor:
    """Return log(pt), eta, phi, log(m), flooring only loss coordinates."""
    momentum = momentum.float()
    transverse_squared = momentum[..., :2].square().sum(-1)
    transverse = transverse_squared.clamp_min(1e-12).sqrt()
    nonzero = transverse_squared > 1e-12
    azimuth = torch.atan2(
        torch.where(nonzero, momentum[..., 1], torch.zeros_like(transverse)),
        torch.where(nonzero, momentum[..., 0], torch.ones_like(transverse)),
    )
    mass_squared = momentum[..., 3].square() - momentum[..., :3].square().sum(-1)
    return torch.stack(
        (transverse.log(), torch.asinh(momentum[..., 2] / transverse), azimuth,
         0.5 * mass_squared.clamp_min(1e-12).log()), dim=-1,
    )


def kinematic_residuals(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = momentum_coordinates(prediction) - momentum_coordinates(target)
    delta_phi = torch.atan2(residual[..., 2].sin(), residual[..., 2].cos())
    return torch.stack((residual[..., 0], residual[..., 1], delta_phi, residual[..., 3]), -1)


def momentum_loss(prediction, target, scales, weights):
    names = ("log_pt", "delta_eta", "delta_phi", "log_mass")
    residuals = kinematic_residuals(prediction, target)
    scale = residuals.new_tensor([scales[name] for name in names])
    weight = residuals.new_tensor([weights[name] for name in names])
    if bool((scale <= 0).any()) or bool((weight < 0).any()) or float(weight.sum()) <= 0:
        raise ValueError("Kinematic scales must be positive and weights nonnegative with positive sum.")
    values = F.huber_loss(residuals / scale, torch.zeros_like(residuals), reduction="none")
    means = values.mean(0) if values.shape[0] else residuals.sum(0)
    return (means * weight).sum() / weight.sum(), dict(zip(names, means.unbind()))


def record_momentum(record: dict) -> torch.Tensor:
    transverse, eta, phi, energy = (record[name].float() for name in ("pt", "eta", "phi", "energy"))
    return torch.stack((transverse * phi.cos(), transverse * phi.sin(), transverse * eta.sinh(), energy), -1)


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