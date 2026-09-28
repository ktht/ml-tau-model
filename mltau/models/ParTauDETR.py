import contextlib

import torch
from torch import nn

from mltau.models.ParticleTransformer import ParticleTransformer


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

    Jet-level head:
      - is_tau: binary tau-tagging logits from the pooled global token.
    """

    def __init__(
        self,
        input_dim: int,
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
                self.append_global_token or self.tau_id_head or self.return_cls
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
            hs = self.decoder(
                tgt=tgt + query_pos,
                memory=memory,
                memory_key_padding_mask=memory_padding_mask,
            )

            hs = hs.transpose(0, 1).contiguous()  # (N, Q, C)

            output = {
                "pred_logits": self.objectness_head(hs),
                "pred_kinematics": self.kinematics_head(hs),
                "pred_charge_logits": self.charge_head(hs),
            }

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
