import math

import torch

PION_MASS_GEV = 0.14


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
    ``[log_pt_ratio, delta_eta, sin_dphi, cos_dphi]``.
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

    log_pt_ratio = kinematics[..., 0]
    if clamp_log_ratios:
        log_pt_ratio = log_pt_ratio.clamp(-5.0, 5.0)

    pt = torch.exp(log_pt_ratio) * reference_pt
    eta = kinematics[..., 1] + reference_eta
    if clamp_log_ratios:
        max_abs_eta = math.acosh(math.sqrt(torch.finfo(kinematics.dtype).max))
        eta = eta.clamp(-max_abs_eta, max_abs_eta)
    phi = reference_phi + torch.atan2(kinematics[..., 2], kinematics[..., 3])
    return torch.stack(
        [
            pt * torch.cos(phi),
            pt * torch.sin(phi),
            pt * torch.sinh(eta),
            torch.sqrt((pt * torch.cosh(eta)) ** 2 + PION_MASS_GEV**2),
        ],
        dim=-1,
    )