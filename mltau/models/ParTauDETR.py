import contextlib
import math

import torch
from torch import nn

from mltau.models.ParticleTransformer import ParticleTransformer


def decode_kinematics(
    kinematics: torch.Tensor,
    reference_pt: torch.Tensor,
    reference_eta: torch.Tensor,
    reference_phi: torch.Tensor,
    reference_energy: torch.Tensor,
    *,
    clamp_log_ratios: bool = False,
    fixed_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode ParTauDETR kinematics into Cartesian four-momenta.

    ``kinematics`` has final dimension
    ``[log_pt_ratio, delta_theta, delta_phi, log_mass_ratio]``.
    Reference tensors may omit trailing object dimensions, such as a ``[B]``
    jet reference used with ``[B, Q, 4]`` kinematics.

    ``fixed_mass`` (same shape as the decoded pt) replaces the regressed
    log-mass-ratio channel entirely: a daughter's mass is set by its identity
    (a charged pion's 139.6 MeV, a neutral's 135 MeV), so when the mass channel
    is unsupervised (kinematics_weights.log_mass = 0) the physical mass comes
    from the predicted meson class instead of an untrained head output.
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
    log_mass_ratio = kinematics[..., 3]
    if clamp_log_ratios:
        log_pt_ratio = log_pt_ratio.clamp(-5.0, 5.0)
        log_mass_ratio = log_mass_ratio.clamp(-5.0, 5.0)

    pt = torch.exp(log_pt_ratio) * reference_pt
    reference_theta = 2.0 * torch.atan(torch.exp(-reference_eta))
    theta = reference_theta + kinematics[..., 1]
    theta_eps = torch.finfo(kinematics.dtype).eps
    theta = theta.clamp(theta_eps, math.pi - theta_eps)
    eta = -torch.log(torch.tan(theta / 2.0))
    phi = reference_phi + kinematics[..., 2]
    if fixed_mass is not None:
        mass = fixed_mass.to(dtype=kinematics.dtype, device=kinematics.device)
    else:
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


class ParTauDETR(ParticleTransformer):
    """
    Set-to-set model: ParticleTransformer encoder + DETR-style decoder.

    Encoder:
      - Keeps the original ParticleTransformer particle-token path.
      - Optionally appends a global token (cls_token -> cls_blocks -> norm)
        to decoder memory.

    Decoder heads per query:
      - pred_logits: object vs no-object
      - pred_kinematics: regression (4D kinematics target)
      - pred_charge_logits: charge classification logits
      - pred_meson_class_logits: configured meson classification logits

    Jet-level head:
      - is_tau: binary tau-tagging logits from the pooled global token.
    """

    def __init__(
        self,
        input_dim: int,
        num_meson_classes: int,
        num_queries: int = 8,
        num_charge_classes: int = 3,
        num_kinematics_components: int = 4,
        # decoder configuration
        decoder_num_layers: int = 4,
        decoder_num_heads: int | None = None,
        decoder_ffn_ratio: int = 4,
        decoder_dropout: float = 0.1,
        # jet-level readout head configuration
        head_dropout: float = 0.1,
        tau_id_head: bool = True,
        # encoder configuration (same as parent)
        pair_input_dim: int = 4,
        pair_extra_dim: int = 0,
        remove_self_pair: bool = False,
        use_pre_activation_pair: bool = True,
        embed_dims: list[int] = [256, 512, 256],
        pair_embed_dims: list[int] = [64, 64, 64],
        num_heads: int = 8,
        num_layers: int = 8,
        block_params=None,
        num_cls_layers: int = 2,
        cls_block_params: dict | None = {
            "dropout": 0,
            "attn_dropout": 0,
            "activation_dropout": 0,
        },
        activation: str = "gelu",
        # misc
        for_inference: bool = False,
        use_amp: bool = False,
        metric: str = "eta-phi",  # for ee should be theta
        verbosity: int = 0,
        append_global_token: bool = True,
        return_memory: bool = False,
        return_cls: bool = False,
        aux_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=1,
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=None,
            activation=activation,
            for_inference=for_inference,
            use_amp=use_amp,
            metric=metric,
            verbosity=verbosity,
            **kwargs,
        )

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.append_global_token = append_global_token
        self.return_memory = return_memory
        self.return_cls = return_cls
        self.tau_id_head = tau_id_head
        self.aux_loss = aux_loss

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        decoder_heads = (
            decoder_num_heads if decoder_num_heads is not None else num_heads
        )

        self.query_embed = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * decoder_ffn_ratio,
            dropout=decoder_dropout,
            activation=activation,
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # DETR-style heads
        self.objectness_head = nn.Linear(embed_dim, 2)  # [object, no-object]
        self.kinematics_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(embed_dim, num_kinematics_components),
        )
        self.charge_head = nn.Linear(embed_dim, num_charge_classes)
        self.meson_class_head = nn.Linear(embed_dim, num_meson_classes)

        # Jet-level tau-tagging head. The global (cls) token already pools the
        # whole jet, so a small MLP on top of it is all that is needed for the
        # binary signal-vs-background classification.
        if tau_id_head:
            head_hidden = embed_dim // 2
            self.tau_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(head_dropout / 2),
                nn.Linear(head_hidden, 2),
            )
        else:
            self.tau_head = None

    @staticmethod
    def _bound_kinematics_angles(kinematics: torch.Tensor) -> torch.Tensor:
        angle_limit = math.pi / 2.0
        bounded_angles = angle_limit * torch.tanh(kinematics[..., 1:3])
        return torch.cat(
            (kinematics[..., :1], bounded_angles, kinematics[..., 3:]), dim=-1
        )

    @torch.jit.ignore()
    def no_weight_decay(self):
        return {
            "cls_token",
            "query_embed.weight",
        }

    def encode_particles(
        self,
        cand_features: torch.Tensor,
        cand_kinematics_pxpypze: torch.Tensor | None,
        cand_mask: torch.Tensor,
    ):
        """
        Returns:
            particle_memory: (P, N, C)
            padding_mask: (N, P) with True on padded positions.
        """
        cand_mask = cand_mask.type(torch.bool)
        padding_mask = ~cand_mask.squeeze(1)
        num_particles = cand_features.size(-1)

        particle_memory = self.embed(cand_features).masked_fill(
            ~cand_mask.permute(2, 0, 1), 0
        )

        attn_mask = None
        if cand_kinematics_pxpypze is not None and self.pair_embed is not None:
            attn_mask = self.pair_embed(cand_kinematics_pxpypze).view(
                -1, num_particles, num_particles
            )

        for block in self.blocks:
            particle_memory = block(
                particle_memory,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
            )

        return particle_memory, padding_mask

    def compute_global_token(
        self,
        particle_memory: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool the whole jet into a single (cls) token.

        Returns:
            global_token: (1, N, C)
        """
        cls_tokens = self.cls_token.expand(1, particle_memory.size(1), -1)
        for block in self.cls_blocks:
            cls_tokens = block(
                particle_memory, x_cls=cls_tokens, padding_mask=padding_mask
            )
        return self.norm(cls_tokens)

    def build_decoder_memory(
        self,
        particle_memory: torch.Tensor,
        padding_mask: torch.Tensor,
        global_token: torch.Tensor | None,
    ):
        if not self.append_global_token or global_token is None:
            return particle_memory, padding_mask

        memory = torch.cat((particle_memory, global_token), dim=0)
        global_pad = torch.zeros_like(padding_mask[:, :1])
        memory_padding_mask = torch.cat((padding_mask, global_pad), dim=1)
        return memory, memory_padding_mask

    def forward(
        self,
        cand_features: torch.Tensor,
        cand_kinematics_pxpypze: torch.Tensor | None = None,
        cand_mask: torch.Tensor | None = None,
    ):
        if cand_mask is None:
            raise ValueError(
                "`cand_mask` is required for variable-length set encoding."
            )

        amp_ctx = torch.autocast("cuda") if self.use_amp else contextlib.nullcontext()

        with amp_ctx:
            particle_memory, padding_mask = self.encode_particles(
                cand_features=cand_features,
                cand_kinematics_pxpypze=cand_kinematics_pxpypze,
                cand_mask=cand_mask,
            )

            need_global_token = (
                self.append_global_token
                or self.tau_id_head
                or self.return_cls
            )
            global_token = (
                self.compute_global_token(particle_memory, padding_mask)
                if need_global_token
                else None
            )

            memory, memory_padding_mask = self.build_decoder_memory(
                particle_memory=particle_memory,
                padding_mask=padding_mask,
                global_token=global_token,
            )

            batch_size = memory.size(1)
            query_pos = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)

            tgt = torch.zeros_like(query_pos)
            if self.aux_loss and self.training:
                # Per-layer outputs for deep supervision: run the decoder stack
                # by hand (nn.TransformerDecoder only returns the last layer)
                # and normalise every intermediate with the SAME shared final
                # LayerNorm, as in DETR, so all layers feed identically scaled
                # activations to the shared heads.
                decoded = tgt + query_pos
                intermediate = []
                for layer in self.decoder.layers:
                    decoded = layer(
                        decoded,
                        memory,
                        memory_key_padding_mask=memory_padding_mask,
                    )
                    intermediate.append(self.decoder.norm(decoded))
                hs = intermediate[-1]
            else:
                intermediate = None
                hs = self.decoder(
                    tgt=tgt + query_pos,
                    memory=memory,
                    memory_key_padding_mask=memory_padding_mask,
                )

            hs = hs.transpose(0, 1).contiguous()  # (N, Q, C)

            output = {
                "pred_logits": self.objectness_head(hs),
                "pred_kinematics": self._bound_kinematics_angles(
                    self.kinematics_head(hs)
                ),
                "pred_charge_logits": self.charge_head(hs),
                "pred_meson_class_logits": self.meson_class_head(hs),
            }

            if intermediate is not None and len(intermediate) > 1:
                aux_outputs = []
                for layer_hs in intermediate[:-1]:
                    layer_hs = layer_hs.transpose(0, 1).contiguous()
                    aux_outputs.append(
                        {
                            "pred_logits": self.objectness_head(layer_hs),
                            "pred_kinematics": self._bound_kinematics_angles(
                                self.kinematics_head(layer_hs)
                            ),
                            "pred_charge_logits": self.charge_head(layer_hs),
                            "pred_meson_class_logits": self.meson_class_head(
                                layer_hs
                            ),
                        }
                    )
                output["aux_outputs"] = aux_outputs

            # Jet-level tau-tagging logits from the pooled global token.
            if self.tau_head is not None and global_token is not None:
                output["is_tau"] = self.tau_head(global_token.squeeze(0))

            if self.return_cls and global_token is not None:
                output["x_cls"] = global_token.squeeze(0)

            if self.return_memory:
                output["particle_memory"] = particle_memory.transpose(0, 1).contiguous()
                output["memory"] = memory.transpose(0, 1).contiguous()
                output["memory_padding_mask"] = memory_padding_mask

            return output
