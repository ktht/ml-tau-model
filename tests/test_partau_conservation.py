"""Synthetic checks for parent-first DETR; no dataset files are required."""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf

from mltau.models.ParTauDETR import ParTauDETR
from mltau.models.ParTauDETR_module import HungarianMatcher, ParTauDETRModule, SetCriterion
from mltau.tools.losses import TauLoss
from mltau.tools.meson_classes import MesonClass
from mltau.tools.partau_detr import (
    DECOMPOSITION_REASONS, conserve_daughters, decode_kinematics, encode_kinematics,
)


class ConservationTests(unittest.TestCase):
    def test_boosted_closure_and_mass_preservation(self):
        spatial = torch.tensor([[[0.6, 0.2, 0.1], [-0.3, 0.1, 0.2], [0.1, -0.2, 0.1]]], dtype=torch.float64)
        masses = torch.tensor([[0.14, 0.5, 0.135]], dtype=torch.float64)
        raw = torch.cat((spatial, torch.sqrt(spatial.square().sum(-1) + masses.square())[..., None]), -1).requires_grad_()
        parent = torch.tensor([[3.0, 4.0, 12.0, math.sqrt(169 + 4)]], dtype=torch.float64)
        for count in (2, 3):
            with self.subTest(count=count):
                selected = torch.arange(3)[None] < count
                result = conserve_daughters(raw, parent, selected)
                self.assertTrue(result["valid"].all())
                corrected = result["p4"]
                torch.testing.assert_close(corrected.masked_fill(~selected[..., None], 0).sum(1), parent)
                invariant = corrected[..., 3].square() - corrected[..., :3].square().sum(-1)
                torch.testing.assert_close(invariant, masses.square(), atol=1e-9, rtol=1e-7)
                self.assertFalse(corrected.requires_grad)

    def test_single_daughter_ignores_regression(self):
        raw = torch.full((1, 2, 4), float("nan"))
        parent = torch.tensor([[1.0, 0.0, 0.0, 2.0]])
        result = conserve_daughters(raw, parent, torch.tensor([[True, False]]))
        self.assertTrue(result["valid"].item())
        torch.testing.assert_close(result["p4"][:, 0], parent.double())

    def test_invalid_reasons_and_fallback(self):
        raw = torch.tensor([[[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]]], dtype=torch.float64)
        cases = ((0.8, [True, True], "mass_budget"),
                 (2.0, [True, True], "zero_momenta"),
                 (2.0, [False, False], "empty"))
        for energy, mask, reason in cases:
            with self.subTest(reason=reason):
                parent = torch.tensor([[0.0, 0.0, 0.0, energy]], dtype=torch.float64)
                result = conserve_daughters(raw, parent, torch.tensor([mask]))
                self.assertFalse(result["valid"].item())
                self.assertEqual(DECOMPOSITION_REASONS[result["reason"].item()], reason)
                torch.testing.assert_close(result["p4"], raw)

    def test_exact_threshold(self):
        raw = torch.tensor([[[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]]], dtype=torch.float64)
        parent = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        result = conserve_daughters(raw, parent, torch.ones((1, 2), dtype=torch.bool))
        self.assertTrue(result["valid"].item())
        self.assertEqual(result["scale"].item(), 0.0)
        torch.testing.assert_close(result["p4"].sum(1), parent)

    def test_invalid_parent(self):
        raw = torch.tensor([[[0.0, 0.0, 0.0, 0.5]]], dtype=torch.float64)
        result = conserve_daughters(raw, torch.tensor([[2.0, 0.0, 0.0, 1.0]]), torch.tensor([[True]]))
        self.assertFalse(result["valid"].item())
        self.assertEqual(DECOMPOSITION_REASONS[result["reason"].item()], "invalid_parent")

    def test_kinematics_round_trip(self):
        reference = {"pt": torch.tensor([3.0], dtype=torch.float64),
                     "eta": torch.tensor([0.2], dtype=torch.float64),
                     "phi": torch.tensor([0.3], dtype=torch.float64),
                     "energy": torch.tensor([4.0], dtype=torch.float64)}
        kinematics = torch.tensor([[[0.2, -0.1, math.sin(0.4), math.cos(0.4), -1.0]]], dtype=torch.float64)
        decoded = decode_kinematics(kinematics, *[reference[name] for name in ("pt", "eta", "phi", "energy")])
        torch.testing.assert_close(encode_kinematics(decoded, reference), kinematics)


class CriterionTests(unittest.TestCase):
    def test_yaml_pcgrad_toggle(self):
        config_dir = Path(__file__).resolve().parents[1] / "mltau" / "config"
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = OmegaConf.merge(
                    OmegaConf.load(config_dir / "training.yaml"),
                    OmegaConf.load(config_dir / "dataset_ParTauDETR.yaml"),
                    OmegaConf.load(config_dir / "main_ParTauDETR.yaml"),
                )
                config.model.detr.pcgrad = enabled
                config.model.encoder.embed_dims = [8]
                config.model.encoder.pair_embed_dims = [8]
                config.model.encoder.num_layers = 1
                config.model.encoder.num_cls_layers = 1
                config.model.detr.decoder_num_layers = 1
                module = ParTauDETRModule(config)
                self.assertEqual(module.pcgrad_enabled, enabled)
                self.assertEqual(module.automatic_optimization, not enabled)
                if enabled:
                    _, outputs, arguments = self.make_case()
                    reference = arguments["kinematics_reference_p4"]
                    with torch.no_grad():
                        outputs["pred_logits"].copy_(torch.tensor([[[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]]]))
                        outputs["pred_kinematics"].fill_(float("nan"))
                        outputs["is_tau"].copy_(torch.tensor([[0.0, 5.0]]))
                    module.forward = Mock(return_value=(outputs, {}, None, reference, reference))
                    prediction = module.predict_step(None, 0)
                    self.assertTrue(prediction["decomposition_valid"].item())
                    self.assertEqual(prediction["pred_mask"].sum().item(), 1)
                    torch.testing.assert_close(prediction["pred_p4"][:, 0], prediction["pred_parent_p4"])

    def test_pcgrad_preserves_parent_direction(self):
        network = torch.nn.Linear(1, 1, bias=False)
        parent = network.weight.sum()
        tagging = -2 * parent
        daughters = -3 * parent
        optimizer = Mock()
        optimizer.zero_grad.side_effect = lambda **kwargs: network.zero_grad(**kwargs)
        scheduler = Mock()
        holder = SimpleNamespace(
            ParTauDETR=network,
            criterion=SimpleNamespace(loss_parent_p4_weight=1.0,
                                      loss_mass_feasibility_parent_weight=0.0,
                                      loss_tau_id_weight=1.0),
            optimizers=lambda: optimizer, lr_schedulers=lambda: scheduler,
            manual_backward=lambda loss, **kwargs: loss.backward(**kwargs),
        )
        ParTauDETRModule._pcgrad_step(holder, {
            "loss_parent_p4": parent, "loss_mass_feasibility_parent": parent * 0,
            "loss_tau_id": tagging, "loss": parent + tagging + daughters,
        })
        torch.testing.assert_close(network.weight.grad, torch.ones_like(network.weight))
        optimizer.step.assert_called_once()
        scheduler.step.assert_called_once()

    def make_case(self, selection="matched", count=2):
        criterion = SetCriterion(
            HungarianMatcher(cost_objectness=0, cost_charge_ce=0, cost_meson_class_ce=0),
            TauLoss(label_smoothing=0),
            (MesonClass("charged", (211, 321), (-1, 1)), MesonClass("neutral", (111,), (0,))),
            regression_query_selection=selection,
        )
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 5.0], [5.0, 0.0], [5.0, 0.0]]], requires_grad=True),
            "pred_kinematics": torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0],
                                               [1.0, 0.0, 0.0, 1.0, 0.0],
                                               [2.0, 0.0, 0.0, 1.0, 0.0]]], requires_grad=True),
            "pred_parent_kinematics": torch.tensor([[0.0, 0.0, 0.0, 1.0, -1.0]], requires_grad=True),
            "pred_charge_logits": torch.zeros((1, 3, 3), requires_grad=True),
            "pred_meson_class_logits": torch.zeros((1, 3, 2), requires_grad=True),
            "is_tau": torch.zeros((1, 2), requires_grad=True),
        }
        reference = {"pt": torch.tensor([1.0]), "eta": torch.tensor([0.0]),
                     "phi": torch.tensor([0.0]), "energy": torch.tensor([2.0])}
        arguments = dict(
            outputs=outputs,
            target_kinematics=torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]]]),
            target_charge_cls=torch.tensor([[0, 2]]), target_meson_class=torch.tensor([[0, 0]]),
            target_mask=torch.tensor([[True, count == 2]]),
            target_parent_charge=torch.tensor([1]), target_parent_decay_mode=torch.tensor([0]),
            target_parent_p4=reference, kinematics_reference_p4=reference,
            target_is_tau=torch.tensor([1]), objectness_threshold=0.5,
        )
        return criterion, outputs, arguments

    def test_threshold_regression_rematches_selected_queries(self):
        criterion, outputs, arguments = self.make_case("threshold")
        losses = criterion(**arguments)
        self.assertEqual(losses["num_kinematics_supervised"].item(), 2)
        gradient = torch.autograd.grad(losses["loss_kinematics"], outputs["pred_kinematics"])[0]
        self.assertEqual(gradient[0, 0].abs().sum().item(), 0)
        self.assertGreater(gradient[0, 2].abs().sum().item(), 0)

    def test_matched_regression_uses_unselected_match(self):
        criterion, outputs, arguments = self.make_case("matched")
        losses = criterion(**arguments)
        gradient = torch.autograd.grad(losses["loss_kinematics"], outputs["pred_kinematics"])[0]
        self.assertEqual(losses["num_kinematics_supervised"].item(), 2)
        self.assertEqual(gradient[0, 2].abs().sum().item(), 0)

    def test_single_daughter_keeps_charge_but_not_regression(self):
        criterion, outputs, arguments = self.make_case(count=1)
        losses = criterion(**arguments)
        self.assertEqual(losses["num_kinematics_supervised"].item(), 0)
        self.assertEqual(losses["loss_kinematics"].item(), 0)
        self.assertEqual(losses["loss_mass_feasibility_daughters"].item(), 0)
        self.assertEqual(losses["num_charge_supervised"].item(), 1)
        gradient = torch.autograd.grad(losses["loss_charge"], outputs["pred_charge_logits"])[0]
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_feasibility_gradient_routing(self):
        criterion, outputs, arguments = self.make_case()
        losses = criterion(**arguments)
        daughter_gradient, parent_gradient = torch.autograd.grad(
            losses["loss_mass_feasibility_daughters"],
            (outputs["pred_kinematics"], outputs["pred_parent_kinematics"]),
            allow_unused=True, retain_graph=True,
        )
        self.assertGreater(daughter_gradient[..., 4].sum().item(), 0)
        self.assertIsNone(parent_gradient)
        daughter_gradient, parent_gradient = torch.autograd.grad(
            losses["loss_mass_feasibility_parent"],
            (outputs["pred_kinematics"], outputs["pred_parent_kinematics"]), allow_unused=True,
        )
        self.assertIsNone(daughter_gradient)
        self.assertLess(parent_gradient[..., 4].item(), 0)

    def test_empty_threshold_selection_still_supervises_objectness(self):
        criterion, outputs, arguments = self.make_case("threshold")
        arguments["objectness_threshold"] = 1.0
        losses = criterion(**arguments)
        self.assertEqual(losses["num_kinematics_supervised"].item(), 0)
        gradient = torch.autograd.grad(losses["loss_objectness"], outputs["pred_logits"])[0]
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_parent_head_shape(self):
        model = ParTauDETR(
            input_dim=3, num_meson_classes=2, num_queries=3,
            embed_dims=[8], pair_embed_dims=[], pair_input_dim=0,
            num_heads=2, num_layers=1, num_cls_layers=1,
            decoder_num_layers=1, decoder_num_heads=2,
        ).eval()
        with torch.no_grad():
            result = model(torch.ones((2, 3, 4)), cand_mask=torch.ones((2, 1, 4), dtype=torch.bool))
        self.assertEqual(result["pred_parent_kinematics"].shape, (2, 5))
        self.assertEqual(result["pred_kinematics"].shape, (2, 3, 5))


if __name__ == "__main__":
    unittest.main()