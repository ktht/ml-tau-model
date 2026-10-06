import warnings
from itertools import combinations

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

from mltau.models.ParTauDETR import ParTauDETR
from mltau.tools.io.general import BatchInputs
from mltau.tools.logging import set_to_set as s2s
from mltau.tools.losses import TauLoss
from mltau.tools.meson_classes import get_meson_classes
from mltau.tools.partau_detr import (
    kinematic_residuals, momentum_coordinates, momentum_loss,
    predicted_momenta, record_momentum, decode_fractions,
)


CLASSIFICATION_COSTS = ("prob", "nll", "clipped_nll")


def _classification_cost_matrix(pred_logits, target_classes, ignore_index=-100,
                                kind="clipped_nll", clip=5.0):
    log_probability = pred_logits.float().log_softmax(-1)
    if kind == "prob":
        cost = 1 - log_probability.exp()
    elif kind == "nll":
        cost = -log_probability
    elif kind == "clipped_nll":
        cost = (-log_probability).clamp_max(clip)
    else:
        raise ValueError(f"Unknown classification cost: {kind}")
    indices = target_classes.clamp_min(0)[:, None].expand(-1, cost.size(1), -1)
    return cost.gather(2, indices) * (target_classes != ignore_index)[:, None]


class HungarianMatcher(nn.Module):
    """Minimize assignment cost over every target-sized query subset."""

    def __init__(self, cost_objectness=1.0, cost_kinematics_l1=2.0,
                 cost_charge_ce=1.0, cost_meson_class_ce=1.0,
                 object_class_index=0, ignore_index=-100,
                 kinematics_component_weights=None, classification_cost="clipped_nll",
                 classification_cost_clip=5.0):
        super().__init__()
        if classification_cost not in CLASSIFICATION_COSTS:
            raise ValueError(f"Unknown classification cost: {classification_cost}")
        self.cost_objectness = cost_objectness
        self.cost_kinematics_l1 = cost_kinematics_l1
        self.cost_charge_ce = cost_charge_ce
        self.cost_meson_class_ce = cost_meson_class_ce
        self.object_class_index = object_class_index
        self.ignore_index = ignore_index
        self.classification_cost = classification_cost
        self.classification_cost_clip = classification_cost_clip
        self.last_cost_terms = {}
        self.kinematics_component_weights: torch.Tensor
        self.register_buffer("kinematics_component_weights", torch.tensor(
            kinematics_component_weights if kinematics_component_weights is not None
            else [1.0, 5.0, 5.0, 0.2], dtype=torch.float32))
        if self.kinematics_component_weights.numel() != 4:
            raise ValueError("Matching requires four weights: log_pt, delta_eta, delta_phi, log_mass")

    @torch.no_grad()
    def forward(self, pred_logits, fraction_logits, parent_p4, pred_charge_logits,
                pred_meson_class_logits, target_p4, target_charge_cls,
                target_meson_class, target_mask):
        self.last_cost_terms = {}
        objectness = -pred_logits.float().log_softmax(-1)[..., self.object_class_index]
        charge = _classification_cost_matrix(
            pred_charge_logits, target_charge_cls, self.ignore_index,
            self.classification_cost, self.classification_cost_clip)
        meson = _classification_cost_matrix(
            pred_meson_class_logits, target_meson_class, self.ignore_index,
            self.classification_cost, self.classification_cost_clip)
        masks = target_mask.cpu().numpy()
        counts = masks.sum(-1)
        num_queries = pred_logits.size(1)
        if np.any(counts > num_queries):
            raise ValueError("Cannot match more target daughters than available queries")
        batches, queries, targets = [], [], []
        term_names = ("objectness", "kinematics", "charge", "meson_class")
        term_totals = np.zeros(len(term_names), dtype=np.float64)
        for target_count in range(1, num_queries + 1):
            group = np.flatnonzero(counts == target_count)
            if not group.size:
                continue
            subsets_np = np.asarray(list(combinations(range(num_queries), target_count)))
            subsets = torch.as_tensor(subsets_np, device=pred_logits.device)
            num_subsets = len(subsets_np)
            for start in range(0, len(group), 256):
                batch_np = group[start:start + 256]
                target_np = np.stack([np.flatnonzero(masks[index]) for index in batch_np])
                batch_indices = torch.as_tensor(batch_np, device=pred_logits.device)
                target_indices = torch.as_tensor(target_np, device=pred_logits.device)
                subset_logits = fraction_logits[batch_indices[:, None, None], subsets[None]]
                flat_logits = subset_logits.reshape(-1, target_count, 4)
                parents = parent_p4[batch_indices, None].expand(-1, num_subsets, -1).reshape(-1, 4)
                decoded = decode_fractions(
                    flat_logits, parents,
                    torch.ones(flat_logits.shape[:2], dtype=torch.bool, device=pred_logits.device),
                ).reshape(len(batch_np), num_subsets, target_count, 4)
                truth = target_p4[batch_indices[:, None], target_indices]
                residuals = kinematic_residuals(decoded[..., None, :], truth[:, None, None])
                kinematics = (residuals.abs() * self.kinematics_component_weights).sum(-1)
                pair_indices = (batch_indices[:, None, None, None],
                                subsets[None, :, :, None], target_indices[:, None, None, :])
                terms = torch.stack((
                    self.cost_objectness * objectness[batch_indices[:, None, None], subsets[None]][..., None].expand_as(kinematics),
                    self.cost_kinematics_l1 * kinematics,
                    self.cost_charge_ce * charge[pair_indices],
                    self.cost_meson_class_ce * meson[pair_indices],
                ), dim=-1).cpu().numpy()
                costs = np.nan_to_num(terms.sum(-1), nan=1e6, posinf=1e6, neginf=-1e6)
                for local_index, batch_index in enumerate(batch_np):
                    best_cost = float("inf")
                    best_assignment = None
                    for subset_index, cost in enumerate(costs[local_index]):
                        rows, columns = linear_sum_assignment(cost)
                        total = float(cost[rows, columns].sum())
                        if total < best_cost:
                            best_cost = total
                            best_assignment = subset_index, rows, columns
                    if best_assignment is None:
                        raise RuntimeError("No finite subset assignment found")
                    subset_index, rows, columns = best_assignment
                    batches.extend([int(batch_index)] * target_count)
                    queries.extend(subsets_np[subset_index, rows].tolist())
                    targets.extend(target_np[local_index, columns].tolist())
                    term_totals += terms[local_index, subset_index, rows, columns].sum(0)
        indices = tuple(torch.tensor(values, device=pred_logits.device, dtype=torch.long)
                        for values in (batches, queries, targets))
        self.last_cost_terms = {
            f"cost/{name}": objectness.new_tensor(total / max(len(batches), 1))
            for name, total in zip(term_names, term_totals)
        }
        return indices


class SetCriterion(nn.Module):
    """Direct parent regression and detached, constrained daughter reconstruction."""

    def __init__(self, matcher, tau_loss, meson_classes, cfg):
        super().__init__()
        self.matcher = matcher
        self.tau_loss = tau_loss
        self.weights = {name: float(cfg[f"weight_{name}"]) for name in (
            "tau_id", "tau_kinematics", "objectness", "daughter_kinematics",
            "daughter_physicality", "charge", "meson_class", "consistency")}
        self.parent_scales = dict(cfg.parent_kinematics_scales)
        self.daughter_scales = dict(cfg.daughter_kinematics_scales)
        self.kinematics_weights = dict(cfg.kinematics_weights)
        self.mass_squared_scale = float(cfg.physicality_mass_squared_scale)
        if self.mass_squared_scale <= 0:
            raise ValueError("physicality_mass_squared_scale must be positive")
        self.eos_coef = float(cfg.eos_coef)
        valid = torch.zeros(3, len(meson_classes))
        for class_index, meson_class in enumerate(meson_classes):
            for charge in meson_class.charges:
                valid[charge + 1, class_index] = 1
        self.register_buffer("charge_meson_class_valid", valid)

    def forward(self, outputs, targets, parent_truth, reference, jet_weights, threshold):
        signal = targets["is_tau"].bool()
        target_mask = targets["particles_mask"].bool() & signal[:, None]
        target_p4 = targets["particles_p4"].float()
        charge_labels = targets["particles_charge_ohe"].argmax(-1).masked_fill(~target_mask, -100)
        meson_labels = targets["particles_meson_class_ohe"].argmax(-1).masked_fill(~target_mask, -100)
        daughters, selected, parent = predicted_momenta(outputs, reference, threshold)
        with torch.no_grad():
            batch_indices, query_indices, target_indices = self.matcher(
                outputs["pred_logits"], outputs["pred_fraction_logits"], parent, outputs["pred_charge_logits"],
                outputs["pred_meson_class_logits"], target_p4, charge_labels,
                meson_labels, target_mask)
        matched = torch.zeros_like(selected)
        matched[batch_indices, query_indices] = True
        matched_daughters = decode_fractions(outputs["pred_fraction_logits"], parent, matched)
        object_labels = torch.ones_like(selected, dtype=torch.long)
        object_labels[batch_indices, query_indices] = 0
        object_loss = F.cross_entropy(
            outputs["pred_logits"].float().transpose(1, 2), object_labels,
            weight=parent.new_tensor([1.0, self.eos_coef]), reduction="none")
        losses = {
            "objectness": (object_loss * signal[:, None]).sum() / (signal.sum() * selected.size(1)).clamp_min(1),
            "tau_id": self.tau_loss.compute_tagging_loss(outputs["is_tau"], targets["is_tau"], jet_weights),
        }
        truth = record_momentum(parent_truth)
        losses["tau_kinematics"], parent_components = momentum_loss(
            parent[signal], truth[signal], self.parent_scales, self.kinematics_weights)
        selected_matches = selected[batch_indices, query_indices]
        losses["daughter_kinematics"], daughter_components = momentum_loss(
            matched_daughters[batch_indices, query_indices],
            target_p4[batch_indices, target_indices],
            self.daughter_scales, self.kinematics_weights)
        for name, logits, labels in (
            ("charge", outputs["pred_charge_logits"], charge_labels),
            ("meson_class", outputs["pred_meson_class_logits"], meson_labels),
        ):
            losses[name] = (F.cross_entropy(logits[batch_indices, query_indices].float(),
                           labels[batch_indices, target_indices]) if batch_indices.numel()
                           else logits.sum() * 0)
        mass_squared = matched_daughters[..., 3].square() - matched_daughters[..., :3].square().sum(-1)
        active = matched
        losses["daughter_physicality"] = (
            F.relu(-mass_squared) * active).sum() / active.sum().clamp_min(1) / self.mass_squared_scale
        joint = (outputs["pred_charge_logits"].float().softmax(-1)[..., :, None]
                 * outputs["pred_meson_class_logits"].float().softmax(-1)[..., None, :])
        invalid = (joint * (1 - self.charge_meson_class_valid)).sum((-2, -1))
        losses["consistency"] = (invalid * signal[:, None]).sum() / (signal.sum() * selected.size(1)).clamp_min(1)
        weighted = {name: self.weights[name] * value for name, value in losses.items()}
        diagnostics = {
            "counts/num_matched": parent.new_tensor(batch_indices.numel()),
            "counts/num_regressed": parent.new_tensor(batch_indices.numel()),
            "matching/matched_below_threshold": (~selected_matches).sum().float() / signal.sum().clamp_min(1),
            "matching/empty_selection_fraction": ((~selected.any(-1)) & signal).sum().float() / signal.sum().clamp_min(1),
            "momentum/matched_spacelike_fraction": ((mass_squared < 0) & active).sum().float() / active.sum().clamp_min(1),
        }
        matched_closure = (matched_daughters.sum(1) - parent.detach()).abs().amax(-1)
        matched_nonempty = matched.any(-1)
        diagnostics["momentum/matched_closure_max"] = (
            matched_closure[matched_nonempty].max() if bool(matched_nonempty.any()) else matched_closure.sum() * 0)
        inference_mass_squared = daughters[..., 3].square() - daughters[..., :3].square().sum(-1)
        inference_active = selected & signal[:, None]
        diagnostics["momentum/spacelike_fraction"] = (
            ((inference_mass_squared < 0) & inference_active).sum().float()
            / inference_active.sum().clamp_min(1))
        nonempty = signal & selected.any(-1)
        closure = (daughters.sum(1) - parent.detach()).abs().amax(-1)
        diagnostics["momentum/closure_max"] = closure[nonempty].max() if bool(nonempty.any()) else closure.sum() * 0
        truth_closure = ((target_p4 * target_mask[..., None]).sum(1) - truth).abs().amax(-1)
        diagnostics["momentum/truth_closure_max"] = (
            truth_closure[signal].max() if bool(signal.any()) else truth_closure.sum() * 0)
        for label, prediction in (("parent", parent), ("daughter_sum", daughters.sum(1))):
            response = prediction[..., :3].norm(dim=-1) / truth[..., :3].norm(dim=-1).clamp_min(1e-6)
            values = response[signal]
            diagnostics[f"momentum/{label}_response_mean"] = values.mean() if values.numel() else response.sum() * 0
            diagnostics[f"momentum/{label}_response_std"] = values.std(unbiased=False) if values.numel() else response.sum() * 0
        return {
            "loss": sum(weighted.values()),
            "task_tau_id": weighted["tau_id"],
            "task_tau_kinematics": weighted["tau_kinematics"],
            "task_daughters": sum(value for name, value in weighted.items() if name not in ("tau_id", "tau_kinematics")),
            **{f"loss_{name}": value for name, value in losses.items()},
            **{f"weighted/{name}": value for name, value in weighted.items()},
            **{f"tau_kinematics/{name}": value for name, value in parent_components.items()},
            **{f"daughter_kinematics/{name}": value for name, value in daughter_components.items()},
            **diagnostics, **self.matcher.last_cost_terms,
        }


def project_tagging_gradient(tagging, kinematics):
    """Asymmetric PCGrad on parameters reached by both parent tasks only."""
    shared = [index for index, (tag, kin) in enumerate(zip(tagging, kinematics))
              if tag is not None and kin is not None]
    if not shared:
        return tagging, None, None
    dot = sum((tagging[index] * kinematics[index]).sum() for index in shared)
    norm_squared = sum(kinematics[index].square().sum() for index in shared)
    coefficient = torch.minimum(dot, torch.zeros_like(dot)) / norm_squared.clamp_min(1e-20)
    projected = list(tagging)
    for index in shared:
        projected[index] = tagging[index] - coefficient * kinematics[index]
    return projected, dot, coefficient


class ParTauDETRModule(L.LightningModule):
    """Constrained DETR with PCGrad only between tau ID and parent kinematics."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.save_hyperparameters({"cfg": OmegaConf.masked_copy(
            cfg, [name for name in ("model", "dataset", "training", "output_dir") if name in cfg])})
        arch, detr = cfg.model, cfg.model.detr
        if int(arch.get("momentum_decoder_version", 0)) != 2:
            raise ValueError("This model requires momentum_decoder_version=2; legacy checkpoints use independent daughter regression.")
        if int(arch.num_queries) != int(cfg.dataset.max_tau_daughters):
            raise ValueError("model.num_queries must equal dataset.max_tau_daughters")
        if int(arch.num_charge_classes) != 3 or int(arch.num_kinematics_components) != 5:
            raise ValueError("Expected three charge classes and five parent kinematic outputs")
        if not detr.tau_id_head:
            raise ValueError("The constrained model requires the tau ID head")
        encoder = OmegaConf.to_container(arch.encoder, resolve=True)
        embed_dim = int(arch.encoder.embed_dims[-1])
        decoder_heads = detr.decoder_num_heads or arch.encoder.num_heads
        if embed_dim % int(decoder_heads) or embed_dim % int(arch.encoder.num_heads):
            raise ValueError("Attention head counts must divide the embedding dimension")
        meson_classes = get_meson_classes(cfg.dataset.tau_daughter_pdg_ids)
        self.num_meson_classes = len(meson_classes)
        self.num_kinematics_components = 5
        self.ParTauDETR = ParTauDETR(
            input_dim=int(cfg.dataset.num_features), num_queries=int(arch.num_queries),
            num_charge_classes=3, num_meson_classes=self.num_meson_classes,
            num_kinematics_components=5, **encoder,
            decoder_num_layers=int(detr.decoder_num_layers), decoder_num_heads=int(decoder_heads),
            decoder_ffn_ratio=int(detr.decoder_ffn_ratio), decoder_dropout=float(detr.decoder_dropout),
            append_global_token=bool(detr.append_global_token), tau_id_head=True,
            head_dropout=float(detr.head_dropout), use_amp=False)
        match = detr.matcher
        self.matcher = HungarianMatcher(
            cost_objectness=float(match.cost_objectness), cost_kinematics_l1=float(match.cost_kinematics_l1),
            cost_charge_ce=float(match.cost_charge), cost_meson_class_ce=float(match.cost_meson_class_ce),
            kinematics_component_weights=list(match.kinematics_component_weights),
            classification_cost=str(match.classification_cost), classification_cost_clip=float(match.classification_cost_clip))
        self.tau_loss = TauLoss(label_smoothing=float(arch.tau_loss.label_smoothing))
        self.criterion = SetCriterion(self.matcher, self.tau_loss, meson_classes, detr.loss)
        self.score_threshold = float(detr.inference.score_threshold)
        self.tau_id_threshold = float(detr.inference.tau_id_threshold)
        self.register_buffer("score_threshold_calibrated", torch.tensor(self.score_threshold))
        self.register_buffer("meson_class_repr_pdg", torch.tensor(
            [130 if set(meson.charges) == {0} else 211 for meson in meson_classes]), persistent=False)
        scan = cfg.training.threshold_scan
        self.threshold_scan_enabled = bool(scan.enabled)
        self.threshold_scan_objective = str(scan.objective)
        self.threshold_buffer_every = max(1, int(scan.buffer_every_n_steps))
        self.threshold_grid = np.linspace(float(scan.low), float(scan.high), int(scan.points))
        self.threshold_buffer = s2s.ThresholdCalibrationBuffer(max_jets=int(scan.jets))
        self.validation_threshold_buffer = s2s.ThresholdCalibrationBuffer(max_jets=int(scan.jets))
        self.val_jets = s2s.SetToSetAccumulator(max_jets=int(cfg.training.get("jet_level_eval_jets", 200000)))
        self.best_val_metrics = None
        self._best_scan_accuracy = 0.0
        self._non_finite_steps = 0
        self._consecutive_skips = 0
        optimizer_cfg = cfg.training.optimizer
        self.grad_clip_val = float(cfg.training.trainer.gradient_clip_val)
        self.grad_skip_norm = float(optimizer_cfg.grad_skip_norm)
        self.grad_skip_patience = int(optimizer_cfg.grad_skip_patience)
        self.grad_skip_warmup_steps = int(optimizer_cfg.grad_skip_warmup_steps)

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        outputs = self.ParTauDETR(inputs.cand_features, inputs.cand_kinematics_pxpypze, inputs.cand_mask)
        return outputs, inputs.target, inputs.weight, inputs.gen_jet_tau_p4s, inputs.reco_jet_p4s

    def _losses(self, batch):
        outputs, targets, weights, parent, reference = self.forward(batch)
        losses = self.criterion(outputs, targets, parent, reference, weights, self.score_threshold_calibrated)
        return outputs, targets, losses

    def _log_losses(self, losses, stage, batch_size):
        for name, value in losses.items():
            if name.startswith("task_"):
                continue
            if name == "loss":
                key = f"{stage}_losses/loss"
            elif name.startswith("loss_"):
                key = f"{stage}_losses/{name[5:]}"
            elif name.startswith(("tau_kinematics/", "daughter_kinematics/", "weighted/")):
                key = f"{stage}_losses/{name}"
            else:
                key = f"{stage}/{name}"
            self.log(key, value.detach(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def _task_gradients(self, loss, parameters):
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None else 1.0
        gradients = torch.autograd.grad(loss * scale, parameters, retain_graph=True, allow_unused=True)
        result = []
        for gradient in gradients:
            if gradient is None:
                result.append(None)
            else:
                value = gradient.detach().float() / scale
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(value)
                    value /= torch.distributed.get_world_size()
                result.append(value)
        return result

    def training_step(self, batch, _batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        outputs, targets, losses = self._losses(batch)
        finite = torch.isfinite(losses["loss"]).int()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(finite):
            self._non_finite_steps += 1
            self.log("train/non_finite_steps", float(self._non_finite_steps), on_step=True)
            if self._non_finite_steps >= 5:
                raise RuntimeError("Five consecutive non-finite training losses")
            return None
        self._non_finite_steps = 0
        parameters = [parameter for parameter in self.ParTauDETR.parameters() if parameter.requires_grad]
        tagging = self._task_gradients(losses["task_tau_id"], parameters)
        kinematics = self._task_gradients(losses["task_tau_kinematics"], parameters)
        projected, dot, coefficient = project_tagging_gradient(tagging, kinematics)
        self.manual_backward(losses["loss"])
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        gradient_scale = float(scaler.get_scale()) if scaler is not None else 1.0
        for parameter, original, corrected in zip(parameters, tagging, projected):
            if original is not None and corrected is not None and parameter.grad is not None:
                parameter.grad.add_(((corrected - original) * gradient_scale).to(parameter.grad.dtype))
        if dot is not None:
            self.log("pcgrad/conflict", (dot < 0).float(), on_step=True, on_epoch=True)
            self.log("pcgrad/dot", dot, on_step=True)
            self.log("pcgrad/projection_coefficient", coefficient, on_step=True)
        optimizer.step()
        self.lr_schedulers().step()
        self._log_losses(losses, "train", len(targets["is_tau"]))
        if self.threshold_scan_enabled and self.global_step % self.threshold_buffer_every == 0:
            self._buffer_for_threshold_scan(batch, outputs, targets, self.threshold_buffer)
        return losses["loss"].detach()

    def on_before_optimizer_step(self, optimizer):
        gradients = [parameter.grad for parameter in self.parameters() if parameter.grad is not None]
        if not gradients:
            return
        total = torch.stack([gradient.float().norm() for gradient in gradients]).norm()
        self.log("grad/total_norm", total, on_step=True)
        self.log("grad/clipped", (total > self.grad_clip_val).float(), on_step=True, on_epoch=True)
        skip = not bool(torch.isfinite(total)) or (
            self.grad_skip_norm > 0 and self.global_step >= self.grad_skip_warmup_steps
            and float(total) > self.grad_skip_norm)
        if skip:
            for parameter in self.parameters():
                parameter.grad = None
            self._consecutive_skips += 1
            if self._consecutive_skips > self.grad_skip_patience:
                raise RuntimeError("Repeated non-finite or outlier gradients")
        else:
            self._consecutive_skips = 0
            if self.grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)
        self.log("grad/skipped", float(skip), on_step=True, on_epoch=True)
        self.log("optim/beta1", float(optimizer.param_groups[0]["betas"][0]), on_step=True)
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.log("grad/amp_scale", float(scaler.get_scale()), on_step=True)

    @torch.no_grad()
    def _buffer_for_threshold_scan(self, batch, outputs, targets, buffer):
        signal = targets["is_tau"].bool()
        if not bool(signal.any()):
            return
        provisional, _, parent = predicted_momenta(outputs, batch[6])
        predicted = momentum_coordinates(provisional)
        truth = momentum_coordinates(targets["particles_p4"])
        charged_class = self.meson_class_repr_pdg == 211
        buffer.add(
            outputs["pred_logits"].float().softmax(-1)[signal, :, 0],
            predicted[signal, :, 1], predicted[signal, :, 2],
            charged_class[outputs["pred_meson_class_logits"].argmax(-1)][signal],
            truth[signal, :, 1], truth[signal, :, 2], targets["particles_mask"][signal],
            charged_class[targets["particles_meson_class_ohe"].argmax(-1)][signal],
            fraction_logits=outputs["pred_fraction_logits"][signal], parent_p4=parent[signal])

    def on_validation_start(self):
        if not self.threshold_scan_enabled or self.trainer.sanity_checking:
            return
        best, values = self.threshold_buffer.scan(self.threshold_grid, objective=self.threshold_scan_objective)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            results = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(results, (best, values))
            best, values = next((result for result in results if result[0] is not None), (None, {}))
        if best is None:
            return
        accuracy = values[best]["decay_mode_accuracy"]
        if accuracy >= 0.5 * self._best_scan_accuracy:
            self._best_scan_accuracy = max(accuracy, self._best_scan_accuracy)
            self.score_threshold_calibrated.fill_(best)
        else:
            warnings.warn("Threshold objective collapsed; retaining the previous threshold")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.score_threshold_calibrated, src=0)
        self.log("threshold/score_threshold", self.score_threshold_calibrated, sync_dist=True)
        self.log("threshold/scan_argmax", float(best), sync_dist=True)
        self.log("threshold/best_decay_mode_accuracy", accuracy, sync_dist=True)
        self.log("threshold/best_f1", values[best]["f1"], sync_dist=True)

    @torch.no_grad()
    def _accumulate_jet_level(self, batch, outputs, targets):
        daughters, selected, _ = predicted_momenta(outputs, batch[6], self.score_threshold_calibrated)
        charges = daughters.new_tensor([-1, 0, 1], dtype=torch.long)
        predicted = s2s.daughters_to_jet_level(
            kin=None, p4=daughters, charge=charges[outputs["pred_charge_logits"].argmax(-1)],
            pdg=self.meson_class_repr_pdg[outputs["pred_meson_class_logits"].argmax(-1)],
            valid=selected, reco_jet=batch[6])
        truth = s2s.daughters_to_jet_level(
            kin=None, p4=targets["particles_p4"], charge=charges[targets["particles_charge_ohe"].argmax(-1)],
            pdg=self.meson_class_repr_pdg[targets["particles_meson_class_ohe"].argmax(-1)],
            valid=targets["particles_mask"], reco_jet=batch[6])
        self.val_jets.update(predicted, truth, targets["is_tau"], outputs["is_tau"].float().softmax(-1)[:, 1],
            {"gen_jet_tau_p4s": batch[5], "reco_jet_p4s": batch[6], "gen_jet_p4s": batch[7]})

    def validation_step(self, batch, _batch_idx):
        outputs, targets, losses = self._losses(batch)
        self._log_losses(losses, "val", len(targets["is_tau"]))
        if not self.trainer.sanity_checking:
            self._accumulate_jet_level(batch, outputs, targets)
            if self.threshold_scan_enabled:
                self._buffer_for_threshold_scan(batch, outputs, targets, self.validation_threshold_buffer)
        return losses["loss"]

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            tensorboard = next((logger.experiment for logger in self.trainer.loggers
                                if hasattr(logger.experiment, "add_figure")), None)
            try:
                metrics = s2s.log_set_to_set_metrics(self.val_jets, tensorboard, self.cfg, self.current_epoch, dataset="val")
                for name, value in metrics.items():
                    self.log(f"val_jet/{name}", value, sync_dist=True)
            except Exception as error:
                warnings.warn(f"Jet-level validation logging failed: {error}")
            if self.threshold_scan_enabled:
                best, values = self.validation_threshold_buffer.scan(self.threshold_grid, objective=self.threshold_scan_objective)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    results = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(results, (best, values))
                    best, values = next((result for result in results if result[0] is not None), (None, {}))
                if best is not None:
                    self.log("threshold/validation_argmax", best, sync_dist=True)
                    for name, value in values[best].items():
                        self.log(f"threshold/validation_best_{name}", value, sync_dist=True)
        self.val_jets.reset()
        self.validation_threshold_buffer.reset()

    def on_validation_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = {name: float(value) for name, value in self.trainer.callback_metrics.items()
                   if name.startswith("val_losses/")}
        current = metrics.get("val_losses/loss")
        if current is not None and (self.best_val_metrics is None or current < self.best_val_metrics["val_losses/loss"]):
            self.best_val_metrics = {**metrics, "epoch": self.current_epoch, "step": self.global_step}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["momentum_decoder_version"] = 2
        checkpoint["best_val_metrics"] = self.best_val_metrics
        checkpoint["best_scan_accuracy"] = self._best_scan_accuracy

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("momentum_decoder_version") != 2:
            raise ValueError("Cannot resume an independent-daughter checkpoint with the constrained decoder")
        self.best_val_metrics = checkpoint.get("best_val_metrics")
        self._best_scan_accuracy = checkpoint.get("best_scan_accuracy", 0.0)

    def predict_step(self, batch, _batch_idx):
        outputs, _, _, _, reference = self.forward(batch)
        daughters, selected, parent = predicted_momenta(outputs, reference, self.score_threshold_calibrated)
        tau_probability = outputs["is_tau"].float().softmax(-1)[:, 1]
        charges = daughters.new_tensor([-1, 0, 1], dtype=torch.long)
        return {
            **outputs, "is_tau_logits": outputs["is_tau"], "is_tau": tau_probability,
            "pred_p4": daughters, "tau_p4": parent,
            "pred_scores": outputs["pred_logits"].float().softmax(-1)[..., 0],
            "pred_mask_objectness": selected,
            "pred_mask": selected & (tau_probability >= self.tau_id_threshold)[:, None],
            "pred_charge": charges[outputs["pred_charge_logits"].argmax(-1)],
            "pred_meson_class": outputs["pred_meson_class_logits"].argmax(-1),
        }

    def test_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)

    def configure_optimizers(self):
        cfg = self.cfg.training
        skip = self.ParTauDETR.no_weight_decay()
        decay, no_decay = [], []
        for name, parameter in self.ParTauDETR.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim <= 1 or name in skip or name.endswith(".bias") else decay).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": float(cfg.optimizer.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0}], lr=float(cfg.lr))
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 0:
            raise ValueError("A positive optimizer-step budget is required")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(cfg.lr), total_steps=total_steps,
            pct_start=float(cfg.optimizer.pct_start), anneal_strategy="cos",
            cycle_momentum=bool(cfg.optimizer.cycle_momentum))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}