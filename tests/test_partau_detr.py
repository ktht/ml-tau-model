import math
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mltau.models.ParTauDETR_module import (
    HungarianMatcher, ParTauDETRModule, project_tagging_gradient,
)
from mltau.tools.logging.set_to_set import ThresholdCalibrationBuffer, scan_threshold
from mltau.tools.partau_detr import (
    decode_fractions, from_tetrahedral, kinematic_residuals,
    momentum_coordinates, momentum_loss, predicted_momenta, to_tetrahedral,
)


def make_config():
    config_dir = Path(__file__).resolve().parents[1] / "mltau" / "config"
    config = OmegaConf.merge(*(OmegaConf.load(config_dir / name) for name in (
        "training.yaml", "dataset_ParTauDETR.yaml", "main_ParTauDETR.yaml")))
    config.model.encoder.embed_dims = [16]
    config.model.encoder.pair_embed_dims = [8]
    config.model.encoder.num_heads = 2
    config.model.encoder.num_layers = 1
    config.model.encoder.num_cls_layers = 1
    config.model.detr.decoder_num_heads = 2
    config.model.detr.decoder_num_layers = 1
    config.model.detr.decoder_dropout = 0.0
    config.training.threshold_scan.enabled = False
    config.training.optimizer.grad_skip_norm = 0.0
    return config


def make_batch(config):
    batch_size, particles = 2, 4
    slots = int(config.dataset.max_tau_daughters)
    features = torch.randn(batch_size, int(config.dataset.num_features), particles)
    candidates = torch.zeros(batch_size, 4, particles)
    candidates[:, 0] = torch.arange(1, particles + 1)
    candidates[:, 1] = 0.1
    candidates[:, 2] = 0.2
    candidates[:, 3] = (candidates[:, :3].square().sum(1) + 1).sqrt()
    mask = torch.ones(batch_size, 1, particles, dtype=torch.bool)
    reference = {"pt": torch.full((batch_size,), 10.0), "eta": torch.zeros(batch_size),
                 "phi": torch.zeros(batch_size), "energy": torch.full((batch_size,), 12.0)}
    daughter_p4 = torch.zeros(batch_size, slots, 4)
    daughter_p4[0, 0] = torch.tensor([4.0, 0.2, 0.1, 5.0])
    daughter_p4[0, 1] = torch.tensor([6.0, -0.2, -0.1, 7.0])
    target_mask = torch.zeros(batch_size, slots, dtype=torch.bool)
    target_mask[0, :2] = True
    charge = torch.zeros(batch_size, slots, 3)
    charge[0, 0, 2] = 1
    charge[0, 1, 1] = 1
    meson = torch.zeros(batch_size, slots, 2)
    meson[0, 0, 0] = 1
    meson[0, 1, 1] = 1
    targets = {"particles_p4": daughter_p4, "particles_mask": target_mask,
               "particles_charge_ohe": charge, "particles_meson_class_ohe": meson,
               "particles_kinematics": torch.zeros(batch_size, slots, 5),
               "is_tau": torch.tensor([1, 0])}
    return features, candidates, targets, mask, torch.ones(batch_size), reference, reference, reference


class MomentumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_transform_roundtrip(self):
        momentum = torch.randn(12, 4, dtype=torch.float64)
        torch.testing.assert_close(from_tetrahedral(to_tetrahedral(momentum)), momentum)

    def test_closure_empty_single_and_multiple(self):
        parent = torch.tensor([[1., 2., 3., 5.]]).expand(3, -1).clone().requires_grad_()
        logits = torch.randn(3, 4, 4, requires_grad=True)
        selected = torch.tensor([[False, False, False, False],
                                 [False, True, False, False], [True, False, True, True]])
        daughters = decode_fractions(logits, parent, selected)
        torch.testing.assert_close(daughters[0], torch.zeros_like(daughters[0]))
        torch.testing.assert_close(daughters[1:].sum(1), parent[1:], atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(daughters[~selected], torch.zeros_like(daughters[~selected])))
        daughters.square().sum().backward()
        self.assertIsNone(parent.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[2].abs().sum()), 0)
        self.assertEqual(float(logits.grad[~selected].abs().sum()), 0)

    def test_permutation_and_autocast(self):
        logits = torch.randn(1, 4, 4)
        parent = torch.tensor([[3., 4., 2., 6.]])
        selected = torch.tensor([[True, False, True, True]])
        order = torch.tensor([2, 0, 3, 1])
        expected = decode_fractions(logits, parent, selected)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = decode_fractions(logits[:, order], parent, selected[:, order])
        torch.testing.assert_close(actual, expected[:, order])
        torch.testing.assert_close(actual.sum(1), parent, atol=2e-6, rtol=2e-6)

    def test_wrapped_angle_and_spacelike_loss(self):
        prediction = torch.tensor([[-1., 0.01, 0., 2.]], requires_grad=True)
        truth = torch.tensor([[-1., -0.01, 0., 2.]])
        residual = kinematic_residuals(prediction, truth)
        self.assertAlmostEqual(float(residual[0, 2]), -2 * math.atan(0.01), places=6)
        spacelike = torch.tensor([[2., 0., 0., 1.]], requires_grad=True)
        settings = {name: 1.0 for name in ("log_pt", "delta_eta", "delta_phi", "log_mass")}
        loss, _ = momentum_loss(spacelike, truth, settings, settings)
        penalty = torch.relu(spacelike[:, :3].square().sum(-1) - spacelike[:, 3].square()).mean()
        self.assertGreater(float(penalty), 0)
        (loss + penalty).backward()
        self.assertTrue(torch.isfinite(spacelike.grad).all())

    def test_pcgrad_keeps_private_and_daughter_gradients(self):
        tagging = [torch.tensor([-2., 1.]), torch.tensor([3.]), None]
        kinematics = [torch.tensor([1., 0.]), None, torch.tensor([4.])]
        projected, dot, _ = project_tagging_gradient(tagging, kinematics)
        self.assertLess(float(dot), 0)
        self.assertAlmostEqual(float((projected[0] * kinematics[0]).sum()), 0)
        self.assertIs(projected[1], tagging[1])
        self.assertIsNone(projected[2])
        daughter_gradient = torch.tensor([-5., 2.])
        original = tagging[0] + kinematics[0] + daughter_gradient
        merged = original + projected[0] - tagging[0]
        torch.testing.assert_close(merged, projected[0] + kinematics[0] + daughter_gradient)
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = merged.clone()
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer.step()
        torch.testing.assert_close(parameter, -0.1 * merged)


class SubsetMatcherTests(unittest.TestCase):
    def test_matches_exhaustive_ordered_assignments(self):
        torch.manual_seed(31)
        num_queries = 4
        counts = [0, 1, 2, 4]
        fraction_logits = torch.randn(len(counts), num_queries, 4)
        parent = torch.tensor([[2., 1., 0.5, 8.]]).expand(len(counts), -1)
        object_logits = torch.randn(len(counts), num_queries, 2)
        charge_logits = torch.randn(len(counts), num_queries, 3)
        meson_logits = torch.randn(len(counts), num_queries, 2)
        target_mask = torch.tensor([[False, False, False, False],
                                    [False, False, True, False],
                                    [False, True, False, True],
                                    [True, True, True, True]])
        target_p4 = torch.zeros(len(counts), num_queries, 4)
        charge_labels = torch.full((len(counts), num_queries), -100, dtype=torch.long)
        meson_labels = torch.full_like(charge_labels, -100)
        for batch_index, count in enumerate(counts):
            if count:
                active = target_mask[batch_index]
                target_logits = torch.randn(1, count, 4) * 0.05
                target_p4[batch_index, active] = decode_fractions(
                    target_logits, parent[batch_index:batch_index + 1],
                    torch.ones(1, count, dtype=torch.bool))[0]
                charge_labels[batch_index, active] = torch.arange(count) % 3
                meson_labels[batch_index, active] = torch.arange(count) % 2
        matcher = HungarianMatcher()
        batch_indices, query_indices, target_indices = matcher(
            object_logits, fraction_logits, parent, charge_logits, meson_logits,
            target_p4, charge_labels, meson_labels, target_mask)
        self.assertEqual(len(batch_indices), sum(counts))
        expected_terms = torch.zeros(4)
        for batch_index, count in enumerate(counts):
            pairs = batch_indices == batch_index
            self.assertEqual(int(pairs.sum()), count)
            if not count:
                continue
            valid_targets = torch.where(target_mask[batch_index])[0]
            best_cost = float("inf")
            best_order = None
            best_terms = None
            for ordering in permutations(range(num_queries), count):
                ordered_queries = torch.tensor(ordering)
                decoded = decode_fractions(
                    fraction_logits[batch_index, ordered_queries][None],
                    parent[batch_index:batch_index + 1],
                    torch.ones(1, count, dtype=torch.bool))[0]
                components = torch.stack((
                    -object_logits[batch_index].log_softmax(-1)[ordered_queries, 0],
                    2 * (kinematic_residuals(decoded, target_p4[batch_index, valid_targets]).abs()
                         * torch.tensor([1., 5., 5., 0.2])).sum(-1),
                    (-charge_logits[batch_index].log_softmax(-1)[
                        ordered_queries, charge_labels[batch_index, valid_targets]]).clamp_max(5),
                    (-meson_logits[batch_index].log_softmax(-1)[
                        ordered_queries, meson_labels[batch_index, valid_targets]]).clamp_max(5),
                ), dim=-1).sum(0)
                cost = float(components.sum())
                if cost < best_cost:
                    best_cost, best_order, best_terms = cost, ordered_queries, components
            actual_order = query_indices[pairs][target_indices[pairs].argsort()]
            torch.testing.assert_close(actual_order, best_order)
            expected_terms += best_terms
        for name, expected in zip(("objectness", "kinematics", "charge", "meson_class"),
                                  expected_terms / sum(counts)):
            torch.testing.assert_close(matcher.last_cost_terms[f"cost/{name}"], expected)

    def test_empty_and_excess_targets(self):
        matcher = HungarianMatcher()
        arguments = (
            torch.zeros(1, 2, 2), torch.zeros(1, 2, 4), torch.tensor([[1., 0., 0., 2.]]),
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 2), torch.ones(1, 3, 4),
            torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long),
        )
        indices = matcher(*arguments, torch.zeros(1, 3, dtype=torch.bool))
        self.assertTrue(all(index.numel() == 0 for index in indices))
        self.assertTrue(all(float(value) == 0 for value in matcher.last_cost_terms.values()))
        with self.assertRaisesRegex(ValueError, "more target daughters"):
            matcher(*arguments, torch.ones(1, 3, dtype=torch.bool))


class CriterionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = make_config()
        self.model = ParTauDETRModule(self.config)
        self.batch = make_batch(self.config)

    def test_detach_and_subset_assignment(self):
        outputs, targets, weights, parent, reference = self.model(self.batch)
        outputs["pred_logits"] = torch.full_like(outputs["pred_logits"], -2.0)
        outputs["pred_logits"][..., 1] = 2.0
        outputs["pred_logits"][:, :2, 0] = 4.0
        outputs["pred_logits"].requires_grad_()
        losses = self.model.criterion(outputs, targets, parent, reference, weights, 0.5)
        self.assertEqual(int(losses["counts/num_matched"]), 2)
        self.assertEqual(int(losses["counts/num_regressed"]), 2)
        self.assertLess(float(losses["momentum/matched_closure_max"]), 1e-5)
        daughter_gradients = torch.autograd.grad(
            losses["task_daughters"], (outputs["tau_kinematics"], outputs["pred_fraction_logits"]),
            retain_graph=True, allow_unused=True)
        self.assertIsNone(daughter_gradients[0])
        self.assertIsNotNone(daughter_gradients[1])
        _, winning_queries, _ = self.model.matcher(
            outputs["pred_logits"], outputs["pred_fraction_logits"],
            predicted_momenta(outputs, reference)[2], outputs["pred_charge_logits"],
            outputs["pred_meson_class_logits"], targets["particles_p4"],
            targets["particles_charge_ohe"].argmax(-1),
            targets["particles_meson_class_ohe"].argmax(-1), targets["particles_mask"])
        excluded = torch.ones_like(targets["particles_mask"])
        excluded[0, winning_queries] = False
        self.assertEqual(float(daughter_gradients[1][excluded].abs().sum()), 0)
        losses["loss"].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters() if parameter.grad is not None))
        outputs, targets, weights, parent, reference = self.model(self.batch)
        empty = self.model.criterion(outputs, targets, parent, reference, weights, 1.0)
        full = self.model.criterion(outputs, targets, parent, reference, weights, 0.0)
        self.assertEqual(int(empty["counts/num_matched"]), 2)
        self.assertEqual(int(empty["counts/num_regressed"]), 2)
        self.assertGreater(float(empty["loss_daughter_kinematics"]), 0)
        for name in ("loss", "loss_daughter_kinematics", "loss_daughter_physicality", "loss_objectness"):
            torch.testing.assert_close(empty[name], full[name])
        self.assertTrue(torch.isfinite(empty["loss"]))
        empty["loss"].backward()

    def test_background_only(self):
        self.batch[2]["is_tau"].zero_()
        outputs, targets, weights, parent, reference = self.model(self.batch)
        losses = self.model.criterion(outputs, targets, parent, reference, weights, 0.5)
        self.assertEqual(float(losses["loss_tau_kinematics"]), 0)
        self.assertEqual(float(losses["task_daughters"]), 0)
        self.assertIn("cost/kinematics", losses)
        losses["loss"].backward()

    def test_threshold_scan_redecodes(self):
        outputs, targets, _, _, reference = self.model(self.batch)
        with torch.no_grad():
            provisional, _, parent = predicted_momenta(outputs, reference)
            coordinates = momentum_coordinates(provisional)
            truth = momentum_coordinates(targets["particles_p4"])
            scores = outputs["pred_logits"].softmax(-1)[..., 0]
            charged = outputs["pred_meson_class_logits"].argmax(-1) == 0
            true_charged = targets["particles_meson_class_ohe"].argmax(-1) == 0
            buffer = ThresholdCalibrationBuffer()
            buffer.add(scores, coordinates[..., 1], coordinates[..., 2], charged,
                       truth[..., 1], truth[..., 2], targets["particles_mask"], true_charged,
                       fraction_logits=outputs["pred_fraction_logits"], parent_p4=parent)
            _, values = buffer.scan([0.3, 0.6, 0.9], objective="f1")
            for threshold in values:
                daughters, _, _ = predicted_momenta(outputs, reference, threshold)
                current = momentum_coordinates(daughters)
                _, expected = scan_threshold(
                    scores.numpy(), current[..., 1].numpy(), current[..., 2].numpy(), charged.numpy(),
                    truth[..., 1].numpy(), truth[..., 2].numpy(), targets["particles_mask"].numpy(),
                    true_charged.numpy(), thresholds=[threshold], objective="f1")
                self.assertEqual(values[threshold], expected[threshold])

    def test_training_checkpoint_resume(self):
        class StopAfterFirstEpoch(L.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                trainer.should_stop = True

        loader = DataLoader([self.batch] * 5, batch_size=None)
        with tempfile.TemporaryDirectory() as directory:
            trainer = L.Trainer(accelerator="cpu", devices=1, precision="bf16-mixed",
                                max_steps=5, max_epochs=-1, limit_train_batches=2,
                                callbacks=[StopAfterFirstEpoch()],
                                logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, num_sanity_val_steps=0,
                                limit_val_batches=0, default_root_dir=directory)
            trainer.fit(self.model, train_dataloaders=loader)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(self.model.lr_schedulers().total_steps, 5)
            self.assertEqual(self.model.lr_schedulers().last_epoch, 2)
            checkpoint = str(Path(directory) / "test.ckpt")
            trainer.save_checkpoint(checkpoint)
            restored = ParTauDETRModule.load_from_checkpoint(checkpoint, weights_only=False)
            self.model.eval()
            restored.eval()
            with torch.no_grad():
                expected = self.model.predict_step(self.batch, 0)
                actual = restored.predict_step(self.batch, 0)
            torch.testing.assert_close(actual["tau_p4"], expected["tau_p4"])
            torch.testing.assert_close(actual["pred_p4"], expected["pred_p4"])
            restored.train()
            resumed = L.Trainer(accelerator="cpu", devices=1, precision="bf16-mixed",
                                max_steps=5, max_epochs=-1, limit_train_batches=2, logger=False,
                                enable_checkpointing=False, enable_progress_bar=False,
                                num_sanity_val_steps=0, limit_val_batches=0,
                                default_root_dir=directory)
            resumed.fit(restored, train_dataloaders=loader, ckpt_path=checkpoint)
            self.assertEqual(resumed.global_step, 5)
            self.assertEqual(restored.lr_schedulers().total_steps, 5)
            self.assertEqual(restored.lr_schedulers().last_epoch, 5)


if __name__ == "__main__":
    unittest.main()