import unittest
from itertools import permutations

import torch

from mltau.tools.partau_detr import boost_p4, decode_kinematics, encode_p4, fraction_daughters


class PhysicalFractionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.reference = {
            "pt": torch.tensor([20.0, 80.0], dtype=torch.float64),
            "eta": torch.tensor([0.2, -1.2], dtype=torch.float64),
            "phi": torch.tensor([3.13, -3.13], dtype=torch.float64),
        }
        self.reference["energy"] = ((self.reference["pt"] * self.reference["eta"].cosh()).square() + 4).sqrt()
        self.parent = decode_kinematics(
            torch.tensor([[0., 0., 0., 1., 0.]] * 2, dtype=torch.float64),
            *(self.reference[name] for name in ("pt", "eta", "phi", "energy")),
        ).requires_grad_()

    def test_physicality_closure_and_gradients(self):
        for count in (1, 5, 8):
            for close in (False, True):
                coordinates = torch.randn(2, count, 4, dtype=torch.float64, requires_grad=True)
                daughters = fraction_daughters(self.parent, coordinates, close=close)
                self.assertTrue((daughters[..., 3] > 0).all())
                mass_squared = daughters[..., 3].square() - daughters[..., :3].square().sum(-1)
                self.assertTrue((mass_squared >= -1e-9).all())
                if close:
                    torch.testing.assert_close(daughters.sum(1), self.parent, atol=1e-9, rtol=1e-9)
                loss = encode_p4(daughters, self.reference).square().mean()
                gradients = torch.autograd.grad(loss, (coordinates, self.parent), retain_graph=True)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_permutation_and_boost_inverse(self):
        coordinates = torch.randn(2, 5, 4, dtype=torch.float64)
        daughters = fraction_daughters(self.parent, coordinates, close=True)
        permutation = torch.tensor([3, 1, 4, 0, 2])
        permuted = fraction_daughters(self.parent, coordinates[:, permutation], close=True)
        torch.testing.assert_close(permuted, daughters[:, permutation])
        rest = boost_p4(daughters, self.parent, inverse=True)
        torch.testing.assert_close(boost_p4(rest, self.parent), daughters)

    def test_nonclosure_and_selected_subset(self):
        coordinates = torch.ones(2, 5, 4, dtype=torch.float64)
        loose = fraction_daughters(self.parent, coordinates, close=False)
        self.assertFalse(torch.allclose(loose.sum(1), self.parent))
        closed = fraction_daughters(self.parent, coordinates, close=True)
        self.assertTrue((closed[:, :2, 3].sum(1) < self.parent[:, 3]).all())

    def test_kinematic_roundtrip(self):
        encoded = encode_p4(self.parent, self.reference)
        decoded = decode_kinematics(encoded, *(self.reference[name] for name in ("pt", "eta", "phi", "energy")))
        torch.testing.assert_close(decoded, self.parent)

    def test_selected_subset_closure_and_empty_gradients(self):
        coordinates = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        active = torch.tensor([[True, False, True, False, False], [False] * 5])
        daughters = fraction_daughters(self.parent, coordinates, close=True, active_mask=active)
        torch.testing.assert_close(daughters[0].sum(0), self.parent[0])
        self.assertTrue((daughters[~active] == 0).all())
        gradients = torch.autograd.grad(daughters.square().sum(), (coordinates, self.parent))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue((gradients[0][~active] == 0).all())


class CriterionExperimentTest(unittest.TestCase):
    def test_closed_matching_exhausts_subsets_and_assignments(self):
        torch.manual_seed(23)
        criterion = self.make_criterion(experiment={"mode": "closure", "diagnostics": True})
        reference = {"pt": torch.tensor([10., 10., 10.]), "eta": torch.zeros(3),
                     "phi": torch.zeros(3), "energy": torch.tensor([11., 11., 11.])}
        parent = torch.tensor([[10., 0., 0., 11.]] * 3, dtype=torch.float64, requires_grad=True)
        coordinates = torch.randn(3, 3, 4, requires_grad=True)
        outputs = {"pred_logits": torch.randn(3, 3, 2), "pred_parent_p4": parent,
                   "pred_fraction_coordinates": coordinates,
                   "pred_charge_logits": torch.randn(3, 3, 3),
                   "pred_meson_class_logits": torch.randn(3, 3, 2)}
        mask = torch.tensor([[True, False, True], [False, True, False], [False] * 3])
        targets = torch.randn(3, 3, 5)
        charge = torch.tensor([[0, -100, 2], [-100, 1, -100], [-100] * 3])
        meson = torch.tensor([[0, -100, 1], [-100, 1, -100], [-100] * 3])
        pairs, selected = criterion.matcher.match_closed_subsets(outputs, targets, charge, meson, mask, reference)
        torch.testing.assert_close(selected.sum(-1), mask.sum(-1))
        closed = fraction_daughters(parent, coordinates, close=True, active_mask=selected)
        torch.testing.assert_close(closed.sum(1)[:2], parent[:2])
        self.assertTrue((closed[~selected] == 0).all())
        batch_indices, query_indices, target_indices = pairs
        for event in (0, 1):
            slots = mask[event].nonzero(as_tuple=True)[0]
            event_reference = {name: value[event:event + 1] for name, value in reference.items()}
            def assignment_cost(query_order):
                query_order = torch.as_tensor(query_order)
                p4 = fraction_daughters(parent[event:event + 1], coordinates[event:event + 1, query_order], close=True)
                costs, _ = criterion.matcher.cost_matrix(
                    outputs["pred_logits"][event:event + 1, query_order], encode_p4(p4, event_reference).float(),
                    outputs["pred_charge_logits"][event:event + 1, query_order], outputs["pred_meson_class_logits"][event:event + 1, query_order],
                    targets[event:event + 1, slots], charge[event:event + 1, slots], meson[event:event + 1, slots],
                )
                return costs[0].diagonal().sum()
            event_pairs = batch_indices == event
            ordered_queries = query_indices[event_pairs][target_indices[event_pairs].argsort()]
            actual = assignment_cost(ordered_queries)
            brute_force = torch.stack([assignment_cost(order) for order in permutations(range(3), slots.numel())]).min()
            torch.testing.assert_close(actual, brute_force)
        outputs["pred_kinematics"] = encode_p4(closed, reference).float()
        before = outputs["pred_kinematics"].clone()
        losses = criterion(
            outputs, targets, charge, meson, mask, torch.zeros(3), torch.zeros(3, dtype=torch.long),
            reference, reference, target_is_tau=torch.tensor([1, 1, 0]),
        )
        torch.testing.assert_close(outputs["pred_kinematics"], before)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(coordinates.grad).all())
        self.assertTrue(torch.isfinite(parent.grad).all())

    def make_criterion(self, detach=False, experiment=None):
        from mltau.models.ParTauDETR_module import HungarianMatcher, SetCriterion
        from mltau.tools.losses import TauLoss
        from mltau.tools.meson_classes import MesonClass
        return SetCriterion(
            HungarianMatcher(), TauLoss(), (MesonClass("charged", (211,), (-1, 1)), MesonClass("neutral", (111,), (0,))),
            loss_objectness_weight=0, loss_tau_id_weight=0,
            loss_kinematics_weight=0, loss_charge_weight=0, loss_meson_class_weight=0,
            loss_soft_parent_kinematics_weight=1,
            detach_momentum_gate=detach, parent_experiment=experiment,
        )

    def test_gate_detachment(self):
        criterion = self.make_criterion()
        logits = torch.zeros(2, 3, 2, requires_grad=True)
        momentum = torch.ones(2, 3, 4, requires_grad=True)
        gate = criterion._soft_objectness_gate(logits, 0.5)
        normal = (momentum * gate.unsqueeze(-1)).sum()
        detached = (momentum * gate.detach().unsqueeze(-1)).sum()
        torch.testing.assert_close(normal, detached)
        normal_grad = torch.autograd.grad(normal, logits, retain_graph=True)[0]
        detached_grad = torch.autograd.grad(detached, (logits, momentum), allow_unused=True)
        self.assertGreater(normal_grad.abs().sum(), 0)
        self.assertIsNone(detached_grad[0])
        self.assertTrue(torch.isfinite(detached_grad[1]).all())

    def test_criterion_ablation_and_diagnostics(self):
        reference = {"pt": torch.tensor([10.]), "eta": torch.tensor([0.]),
                     "phi": torch.tensor([0.]), "energy": torch.tensor([11.])}
        logits = torch.zeros(1, 3, 2, requires_grad=True)
        kinematics = torch.tensor([[[0., .1, .1, 1., 0.]] * 3], requires_grad=True)
        outputs = {"pred_logits": logits, "pred_kinematics": kinematics,
                   "pred_charge_logits": torch.zeros(1, 3, 3, requires_grad=True),
                   "pred_meson_class_logits": torch.zeros(1, 3, 2, requires_grad=True)}
        arguments = dict(
            outputs=outputs, target_kinematics=kinematics[:, :1].detach(),
            target_charge_cls=torch.ones(1, 1, dtype=torch.long),
            target_meson_class=torch.ones(1, 1, dtype=torch.long),
            target_mask=torch.ones(1, 1, dtype=torch.bool),
            target_parent_charge=torch.zeros(1), target_parent_decay_mode=torch.zeros(1, dtype=torch.long),
            target_parent_p4=reference, kinematics_reference_p4=reference,
        )
        normal = self.make_criterion()(**arguments)["loss"]
        detached = self.make_criterion(detach=True)(**arguments)["loss"]
        torch.testing.assert_close(normal, detached)
        gradient = torch.autograd.grad(normal, logits, retain_graph=True)[0]
        self.assertGreater(gradient.abs().sum(), 0)
        gradient = torch.autograd.grad(detached, logits, retain_graph=True, allow_unused=True)[0]
        self.assertTrue(gradient is None or (gradient == 0).all())
        diagnostic_criterion = self.make_criterion(experiment={"diagnostics": True})
        diagnostics = diagnostic_criterion(**arguments)
        torch.testing.assert_close(normal, diagnostics["loss"])
        self.assertIn("experiment/oracle_count_matched/huber", diagnostics)
        events = diagnostic_criterion.last_parent_diagnostics
        self.assertEqual(events["hard_iou"].shape, (1,))
        self.assertTrue((events["oracle_huber"] <= events["hungarian_huber"] + 1e-6).all())
        no_matching = self.make_criterion(experiment={"hungarian_losses": False})(**arguments)
        self.assertTrue(torch.isfinite(no_matching["loss"]))
        arguments["target_mask"] = torch.zeros(1, 1, dtype=torch.bool)
        arguments["target_is_tau"] = torch.zeros(1, dtype=torch.long)
        empty = diagnostic_criterion(**arguments)
        self.assertTrue(all(torch.isfinite(value).all() for value in empty.values()))


class ParentHeadTest(unittest.TestCase):
    def test_all_modes_and_baseline_compatibility(self):
        from mltau.models.ParTauDETR import ParTauDETR
        options = dict(
            input_dim=3, num_meson_classes=2, num_queries=3,
            embed_dims=[8], pair_embed_dims=[8], num_heads=2,
            num_layers=1, num_cls_layers=1, decoder_num_layers=1,
            decoder_num_heads=2, decoder_dropout=0.,
        )
        features = torch.randn(2, 3, 4)
        mask = torch.ones(2, 1, 4, dtype=torch.bool)
        reference = {"pt": torch.tensor([10., 20.]), "eta": torch.tensor([0., .2]),
                     "phi": torch.tensor([0., .1]), "energy": torch.tensor([11., 22.])}
        baseline = ParTauDETR(**options).eval()
        explicit = ParTauDETR(**options, parent_mode="baseline").eval()
        explicit.load_state_dict(baseline.state_dict(), strict=True)
        for name, value in baseline(features, cand_mask=mask).items():
            torch.testing.assert_close(value, explicit(features, cand_mask=mask)[name])
        for mode in ("direct", "fractions", "closure"):
            model = ParTauDETR(**options, parent_mode=mode).eval()
            outputs = model(features, cand_mask=mask, kinematics_reference_p4=reference)
            self.assertEqual(outputs["pred_parent_p4"].shape, (2, 4))
            if mode == "closure":
                nonempty = outputs["pred_active_mask"].any(-1)
                torch.testing.assert_close(outputs["pred_daughter_p4"].sum(1)[nonempty], outputs["pred_parent_p4"][nonempty], atol=1e-7, rtol=1e-7)
                self.assertTrue((outputs["pred_daughter_p4"][~outputs["pred_active_mask"]] == 0).all())
            loss = outputs["pred_parent_p4"].square().mean() + outputs["pred_kinematics"].square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()