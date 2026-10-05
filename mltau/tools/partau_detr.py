import math

import torch


@torch.autocast(device_type="cuda", enabled=False)
@torch.autocast(device_type="cpu", enabled=False)
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

    kinematics = kinematics.float()

    def expand_reference(reference: torch.Tensor) -> torch.Tensor:
        reference = reference.float()
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
    sin_dphi, cos_dphi = kinematics[..., 2], kinematics[..., 3]
    undefined_phi = (sin_dphi == 0) & (cos_dphi == 0)
    phi = reference_phi + torch.atan2(
        sin_dphi, torch.where(undefined_phi, torch.ones_like(cos_dphi), cos_dphi)
    )
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