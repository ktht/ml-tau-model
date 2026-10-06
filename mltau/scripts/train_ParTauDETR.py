import os

# comet_ml installs its framework monkey-patches at IMPORT time, gated on this
# variable, so it has to be set before the import below -- not after. Lightning's
# CometLogger sets the same variable in its constructor, which is too late once
# comet_ml has been imported at module scope, as it is here. Without this, the
# pytorch patch wraps Tensor.backward() and logs a bare "loss" metric on its own
# step counter alongside the curves this repository logs deliberately.
os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")

# comet_ml monkey-patches the frameworks it auto-instruments, so it has to be
# imported before torch/lightning to log correctly. Guarded so that this script
# still runs on an environment without comet installed.
try:
    import comet_ml  # noqa: F401

    _COMET_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    comet_ml = None
    _COMET_IMPORT_ERROR = exc

import faulthandler
import inspect
import json
import signal
import time
import warnings

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from mltau.models import ParTauDETR_module
from mltau.tools.io import ParTauDETR_dataloader as dl


# Every experiment from this repository belongs to one Comet project. This is a
# constant rather than a config key on purpose: a stale or overridden value
# silently creates a second project (Comet slugifies names to lowercase, so
# "ml-tau-ParTauDETR" showed up as "ml-tau-partaudetr").
COMET_PROJECT_NAME = "ml-tau"

# Lightning >= 2.5 injects ExperimentConfig(disabled=True) whenever `online`
# is the False singleton, which produces an OfflineExperiment that records
# nothing and writes no uploadable archive. A falsy value that is not that
# singleton (0 is not False) keeps comet_ml's real offline path while dodging
# the coercion. Verified against lightning 2.5.4 / comet_ml 3.52.
COMET_OFFLINE = 0


def build_comet_logger(cfg: DictConfig, save_dir: str):
    """
    Build a CometLogger from cfg.logging.comet, or return None if disabled.

    Lightning reworked CometLogger in 2.5 to wrap `comet_ml.start()`, which moved
    the project, name and offline-directory arguments around. Both spellings are
    supported, selected by inspecting the installed signature.
    """
    comet_cfg = cfg.logging.comet
    if not comet_cfg.enabled:
        return None

    if comet_ml is None:
        raise RuntimeError(
            "logging.comet.enabled is true but comet_ml is not installed "
            f"({_COMET_IMPORT_ERROR}). Install it with `pip install comet_ml`, "
            "or set logging.comet.enabled=false."
        )

    from lightning.pytorch.loggers import CometLogger

    accepted = inspect.signature(CometLogger.__init__).parameters
    online = bool(comet_cfg.online)
    experiment_name = comet_cfg.experiment_name
    experiment_name = None if experiment_name is None else str(experiment_name)
    experiment_key = comet_cfg.experiment_key
    experiment_key = None if experiment_key is None else str(experiment_key)
    workspace = None if comet_cfg.workspace is None else str(comet_cfg.workspace)
    # Forwarded into comet_ml.ExperimentConfig; drives the System Metrics tab.
    # Lightning sets COMET_DISABLE_AUTO_LOGGING=1, but that only gates framework
    # auto-instrumentation, not the system/environment monitor.
    env_kwargs = {
        "log_env_details": bool(comet_cfg.log_env_details),
        "log_env_gpu": bool(comet_cfg.log_env_gpu),
        "log_env_cpu": bool(comet_cfg.log_env_cpu),
        # Belt and braces with COMET_DISABLE_AUTO_LOGGING above: that stops the
        # patches being installed, this stops them logging if they were.
        "auto_metric_logging": bool(comet_cfg.get("auto_metric_logging", False)),
    }

    if "online" in accepted:
        # Lightning >= 2.5. Two traps here:
        #  - `mode` is comet_ml's get/create/get_or_create selector, NOT
        #    online-vs-offline. That is `online`.
        #  - every unrecognised kwarg is forwarded into comet_ml.ExperimentConfig,
        #    so the name and the offline directory are passed flat, not as a
        #    prebuilt ExperimentConfig object.
        # Leaving `name` unset lets Comet generate one itself.
        kwargs = {
            "project": COMET_PROJECT_NAME,
            "online": True if online else COMET_OFFLINE,
            "offline_directory": save_dir,
            "log_code": bool(comet_cfg.log_code),
            "log_graph": bool(comet_cfg.log_graph),
            **env_kwargs,
        }
        if experiment_name is not None:
            kwargs["name"] = experiment_name
        if workspace is not None:
            kwargs["workspace"] = workspace
        if experiment_key is not None:
            # Resume the run carrying this key, creating it if it is new.
            kwargs["experiment_key"] = experiment_key
            kwargs["mode"] = "get_or_create"
        return _apply_tags(CometLogger(**kwargs), comet_cfg)

    # Lightning < 2.5
    kwargs = {
        "project_name": COMET_PROJECT_NAME,
        "save_dir": save_dir,
        "offline": not online,
    }
    if workspace is not None:
        kwargs["workspace"] = workspace
    if experiment_name is not None:
        kwargs["experiment_name"] = experiment_name
    if experiment_key is not None:
        kwargs["experiment_key"] = experiment_key
    # Older CometLogger forwards unknown kwargs to the comet Experiment, so the
    # env_* flags reach it even though they are not named in the signature.
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return _apply_tags(CometLogger(**kwargs, **env_kwargs), comet_cfg)


def _apply_tags(logger, comet_cfg):
    """
    Attach cfg.logging.comet.tags to the experiment.

    Done through experiment.add_tags rather than a constructor argument because
    only the >=2.5 CometLogger forwards unknown kwargs to ExperimentConfig; this
    path works on both.
    """
    tags = [str(t) for t in (comet_cfg.get("tags", None) or [])]
    if tags:
        try:
            logger.experiment.add_tags(tags)
        except Exception as exc:  # pragma: no cover - never fail a run over a label
            warnings.warn(f"Could not set Comet tags {tags}: {exc}")
    return logger


def _optional_float(value: float | None) -> float | None:
    """`null` in the config disables a threshold; anything else is a float."""
    return None if value is None else float(value)


def _optional_trainer_kwargs(cfg: DictConfig) -> dict:
    """
    Trainer arguments that are only passed when set, because Lightning's own
    defaults (validate once per epoch, use every val batch) are not expressible
    as an explicit value we would want to hardcode here.
    """
    kwargs = {}
    max_steps = cfg.training.trainer.get("max_steps", None)
    if max_steps is not None:
        # A step budget and an epoch budget together mean whichever binds first
        # wins, which for small datasets is max_epochs -- defeating the point.
        kwargs["max_steps"] = int(max_steps)
        kwargs["max_epochs"] = -1
    else:
        kwargs["max_epochs"] = int(cfg.training.trainer.max_epochs)
    interval = cfg.training.trainer.get("val_check_interval", None)
    if interval is not None:
        kwargs["val_check_interval"] = int(interval)
    limit = cfg.training.trainer.get("limit_val_batches", None)
    if limit is not None:
        kwargs["limit_val_batches"] = int(limit)
    return kwargs


def _cuda_visible_devices_hint() -> str:
    """
    Detect a CUDA_VISIBLE_DEVICES that indexes past the devices actually present.

    Slurm hands out node-global GPU indices while its cgroup renumbers the
    job's view from 0, so an allocation of the node's second GPU arrives as
    CUDA_VISIBLE_DEVICES=1 against a single visible device. CUDA then finds
    nothing. pynvml is used because it reports the physical devices regardless
    of the masking variable, and unlike torch it does not need a working
    CUDA context.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return ""
    indices = [part for part in visible.split(",") if part.strip().isdigit()]
    if not indices:
        return ""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception:
        return ""
    if count and max(int(i) for i in indices) >= count:
        return (
            f"CUDA_VISIBLE_DEVICES={visible} indexes past the {count} device(s) "
            "this process can actually see.\n"
            "Slurm sets the node-global GPU index while its cgroup renumbers the\n"
            "job's view from 0, so the variable is stale. Renumber it densely:\n"
            f"  export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(count))}\n"
            "train-gpu-DETR.sh now does this automatically before launching.\n\n"
        )
    return ""


def check_accelerator(cfg: DictConfig) -> str:
    """
    Validate the requested accelerator before any data is touched.

    Lightning's `accelerator="auto"` quietly selects CPU when CUDA is not
    visible, so a misconfigured job trains ~100x too slowly instead of failing.
    """
    requested = str(cfg.training.trainer.accelerator)
    if requested in ("gpu", "cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            _cuda_visible_devices_hint()
            + "training.trainer.accelerator is "
            f"{requested!r} but no CUDA device is visible.\n"
            f"  torch.cuda.is_available() : False\n"
            f"  torch.version.cuda        : {torch.version.cuda}\n"
            "  CUDA_VISIBLE_DEVICES      : "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "  SLURM_JOB_GPUS            : "
            f"{os.environ.get('SLURM_JOB_GPUS', '<unset>')}\n"
            "  SLURM_JOB_ID              : "
            f"{os.environ.get('SLURM_JOB_ID', '<unset>')}\n"
            "Check that (a) the job actually requested a GPU -- note that #SBATCH\n"
            "lines only count before the first command in the submit script -- and\n"
            "(b) run.sh passes --nv to apptainer. To run without a GPU on purpose,\n"
            "set training.trainer.accelerator=cpu training.trainer.precision=bf16-mixed."
        )
    if requested in ("gpu", "cuda"):
        print(
            f"[ParTauDETR] CUDA devices: {torch.cuda.device_count()} "
            f"({torch.cuda.get_device_name(0)}), "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GiB"
        )
    return requested


def build_loggers(cfg: DictConfig, tb_log_dir: str) -> list:
    loggers = []
    if cfg.logging.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTauDETR_experiment",
                log_graph=False,
                default_hp_metric=False,
            )
        )
    comet_logger = build_comet_logger(
        cfg, save_dir=os.path.join(cfg.output_dir, "comet")
    )
    if comet_logger is not None:
        loggers.append(comet_logger)
    if not loggers:
        warnings.warn("No logger is enabled under cfg.logging; metrics are dropped.")
    return loggers


def measure_inference_time(model, datamodule, cfg, device, max_batches: int = 20,
                           warmup_batches: int = 3) -> dict | None:
    """
    Time a forward-only pass over the validation loader.

    Two numbers, because they answer different questions. `forward_*` excludes
    everything but the model, which is what scales with model size and is
    comparable across a study whose runs differ in dataset size. `end_to_end_*`
    includes waiting for the dataloader, so a big gap between them says the run
    was input-bound rather than compute-bound -- worth knowing before reading
    any throughput number as a property of the network.

    The first batches are discarded: on ROCm and CUDA alike the first forward
    pays for kernel compilation and allocator warm-up, which at these batch
    counts would otherwise dominate the mean.

    `device` is passed in rather than read off the model: Lightning moves the
    module back to the CPU when fit() tears down, so taking the device from the
    parameters timed the CPU and wrote that into metrics.json as the model's
    throughput (2.6 ms/jet against ~45 us/jet on the L40S it trained on).
    """
    loader = datamodule.val_dataloader()
    if loader is None:
        return None

    device = torch.device(device)
    model.to(device)
    model.eval()
    # A short run has fewer validation batches than the requested count; time
    # what exists rather than looping past the end.
    try:
        available = len(loader) - warmup_batches
        if available >= 1:
            max_batches = min(max_batches, available)
    except TypeError:  # loader without a length
        pass

    forward_seconds = 0.0
    loop_start = None
    jets = 0
    batches = 0

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index == warmup_batches:
                    # Start the end-to-end clock only once warm, so both numbers
                    # cover the same batches.
                    _sync()
                    loop_start = time.perf_counter()
                batch = _move_batch(batch, device)
                _sync()
                started = time.perf_counter()
                model.forward(batch)
                _sync()
                elapsed = time.perf_counter() - started
                if index >= warmup_batches:
                    forward_seconds += elapsed
                    jets += int(batch[0].shape[0])
                    batches += 1
                if batches >= max_batches:
                    break
    except Exception as exc:  # pragma: no cover - timing must not fail a run
        warnings.warn(f"inference timing failed: {exc}")
        return None
    finally:
        model.train()

    if batches == 0 or jets == 0:
        return None

    end_to_end = time.perf_counter() - loop_start if loop_start else forward_seconds
    return {
        "device": str(device),
        "batches_timed": batches,
        "jets_timed": jets,
        "batch_size": jets // batches,
        "forward_seconds_total": round(forward_seconds, 4),
        "forward_ms_per_batch": round(forward_seconds / batches * 1e3, 3),
        "forward_us_per_jet": round(forward_seconds / jets * 1e6, 3),
        "forward_jets_per_second": round(jets / forward_seconds, 1),
        "end_to_end_seconds_total": round(end_to_end, 4),
        "end_to_end_jets_per_second": round(jets / end_to_end, 1),
        # < 1 means the loop spends most of its time waiting for data.
        "forward_fraction_of_wall": round(forward_seconds / end_to_end, 3),
    }


def _move_batch(batch, device):
    """Batches are nested tuples/dicts of tensors; move the tensors only."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch(value, device) for value in batch)
    return batch


def write_run_metrics(cfg, trainer, datamodule, loggers, wall_seconds, path,
                      inference=None):
    """
    Summarise the run into one JSON file next to the checkpoints.

    The best validation loss is otherwise recorded nowhere machine-readable:
    the best-model checkpoint is weights-only, so it carries no `callbacks`
    block (and no best_model_score), the filename carries no metric template,
    and the value survives only in the TensorBoard event files and in Comet.
    Aggregating a scaling study then means parsing protobufs or hitting the
    network.
    """
    summary = {
        "wall_seconds": round(wall_seconds, 1),
        "global_step": int(trainer.global_step),
        "epochs_completed": int(trainer.current_epoch),
        # global_step is fixed by max_steps; epochs are NOT, they scale with
        # 1/dataset_size, so this says how many passes over the data were made.
        "max_steps_requested": (
            None
            if cfg.training.trainer.get("max_steps", None) is None
            else int(cfg.training.trainer.max_steps)
        ),
        "seed": int(cfg.training.get("seed", 42)),
        "selection_seed": int(cfg.dataset.get("selection_seed", 42)),
        "max_jets_per_sample": OmegaConf.to_container(
            cfg.dataset.get("max_jets_per_sample", {}) or {}, resolve=True
        ),
    }

    # Jets and batches actually used, which differ from the requested limits:
    # selection rounds up to whole row groups.
    for split in ("train", "val"):
        dataset = getattr(datamodule, f"{split}_dataset", None)
        if dataset is not None:
            summary[f"n_{split}_jets"] = int(dataset.num_rows)
            summary[f"n_{split}_batches"] = int(len(dataset))

    # Best scores, per monitored checkpoint.
    for callback in trainer.checkpoint_callbacks:
        monitor = callback.monitor
        if monitor is None:
            continue
        key = "best_val_loss" if monitor.startswith("val") else "best_train_loss"
        score = callback.best_model_score
        summary[key] = None if score is None else float(score)
        summary[f"{key}_checkpoint"] = callback.best_model_path or None
        # The epoch/step the best checkpoint was taken at lives inside the file;
        # a large gap to global_step means it stopped improving early.
        if callback.best_model_path and os.path.exists(callback.best_model_path):
            try:
                ckpt = torch.load(
                    callback.best_model_path, map_location="cpu", weights_only=False
                )
                summary[f"{key}_epoch"] = int(ckpt.get("epoch", -1))
                summary[f"{key}_step"] = int(ckpt.get("global_step", -1))
                del ckpt
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Could not read {callback.best_model_path}: {exc}")

    # A missing best_val_loss means validation never produced a score. The most
    # likely cause with a step budget is max_steps < batches-per-epoch, so the
    # run ends before the first end-of-epoch validation. Recorded explicitly:
    # 36 silently null runs would be discovered only at aggregation time.
    if summary.get("best_val_loss") is None:
        n_train_batches = summary.get("n_train_batches")
        detail = (
            f"max_steps={summary['max_steps_requested']} vs "
            f"{n_train_batches} training batches per epoch"
            if summary.get("max_steps_requested") and n_train_batches
            else "check val_check_interval / limit_val_batches"
        )
        note = (
            "no best validation score was recorded: validation never ran or "
            f"logged nothing ({detail})"
        )
        summary["warning"] = note
        warnings.warn(note)

    # The per-head breakdown BELONGING TO the best validation loss, captured by
    # ParTauDETRModule.on_validation_end. final_metrics below is the last
    # validation pass, which is a different epoch whenever the run stopped
    # improving early, so the two must not be confused.
    best_metrics = getattr(getattr(trainer, "lightning_module", None),
                           "best_val_metrics", None)
    summary["best_val_metrics"] = best_metrics

    # Forward-only throughput of the trained model.
    summary["inference"] = inference

    # Every metric from the final validation pass, so the per-head breakdown is
    # available without opening TensorBoard.
    summary["final_metrics"] = {
        name: float(value)
        for name, value in trainer.callback_metrics.items()
        if hasattr(value, "item") or isinstance(value, (int, float))
    }

    # Link back to the Comet run.
    for logger in loggers:
        experiment = getattr(logger, "_experiment", None)
        if experiment is None:
            continue
        for attr, out in (("get_key", "comet_experiment_key"), ("url", "comet_url")):
            try:
                value = getattr(experiment, attr)
                summary[out] = value() if callable(value) else value
            except Exception:
                pass

    with open(path, "w") as out_file:
        json.dump(summary, out_file, indent=2, sort_keys=True)
        out_file.write("\n")
    print(f"[ParTauDETR] wrote {path}")

    # Also to stdout, so the slurm log of a scaling task carries the headline
    # numbers without anyone opening the JSON.
    if best_metrics:
        print(
            f"[ParTauDETR] best val loss {best_metrics['val_losses/loss']:.5f} "
            f"at epoch {best_metrics['epoch']} (step {best_metrics['step']})",
            flush=True,
        )
        for name, value in sorted(best_metrics.items()):
            if name.startswith("val_losses/") and name != "val_losses/loss":
                print(f"    {name:38s} {value:.5f}", flush=True)
    if inference:
        print(
            f"[ParTauDETR] inference {inference['forward_us_per_jet']} us/jet "
            f"({inference['forward_jets_per_second']:,.0f} jets/s) on "
            f"{inference['device']}, {inference['batches_timed']} batches of "
            f"{inference['batch_size']}",
            flush=True,
        )
    return summary


def check_output_dir(cfg: DictConfig) -> str:
    """
    Fail early and legibly if output_dir is not writable.

    The configured default is an absolute path on one particular machine, so on
    any other it surfaces as a bare `OSError: Read-only file system: '/home/...'`
    from os.makedirs, with nothing connecting it to a config key or to a missing
    command-line override.
    """
    output_dir = os.path.expanduser(os.path.expandvars(str(cfg.output_dir)))
    probe = output_dir
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if not os.access(probe, os.W_OK):
        raise PermissionError(
            f"output_dir is not writable: {output_dir}\n"
            f"  nearest existing parent : {probe}\n"
            f"  writable                : False\n"
            "\n"
            "output_dir defaults to an absolute path baked into "
            "main_ParTauDETR.yaml, which is only correct on the machine it was "
            "written for. Pass one explicitly, e.g.\n"
            "  sbatch train-gpu-lumi.sh output_dir=/scratch/<project>/<run_name>\n"
            "Note that `sbatch ... train-gpu-lumi.sh` with no trailing arguments "
            "passes no Hydra overrides at all; the log line "
            "'overrides: []' is the giveaway."
        )
    return output_dir


@hydra.main(config_path="../config", config_name="main_ParTauDETR", version_base=None)
def train(cfg: DictConfig):
    # `kill -USR1 <pid>` dumps the Python stack of every thread to stderr without
    # killing the process. A hung job otherwise gives no way to tell whether it
    # is blocked in the Parquet scan, waiting on a dataloader worker, or inside
    # a HIP call.
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    # Seed before anything builds a module or a dataloader. workers=True gives
    # each dataloader worker a distinct, derived seed.
    L.seed_everything(int(cfg.training.get("seed", 42)), workers=True)
    # TF32 matmuls on Ampere and later GPUs (L40S, A100). Lightning prints this
    # hint on every run; it is free throughput and independent of AMP.
    torch.set_float32_matmul_precision("high")

    # Ensure the datamodule follows the signal-only path by default.
    cfg.training.model.name = "ParTauDETR"
    cfg.training.model.task = "set2set"

    # Safety belt: after Hydra composes configs, `cfg.dataset` must be a DictConfig,
    # not an overridden string.
    if not isinstance(cfg.dataset, DictConfig):
        raise TypeError(
            f"'cfg.dataset' is expected to be a DictConfig but got {type(cfg.dataset)}. "
            f"Check that Hydra composed configs correctly (main_ParTauDETR.yaml defaults)."
        )

    datamodule = dl.ParTauDETRDataModule(cfg=cfg, debug_run=cfg.training.debug_run)


    model = ParTauDETR_module.ParTauDETRModule(cfg=cfg)

    check_output_dir(cfg)

    models_dir = os.path.join(cfg.output_dir, "models")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    comet_dir = os.path.join(cfg.output_dir, "comet")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(comet_dir, exist_ok=True)

    print(f"[ParTauDETR] output_dir: {cfg.output_dir}")
    print(f"[ParTauDETR] checkpoints dir: {models_dir}")

    loggers = build_loggers(cfg, tb_log_dir=tb_log_dir)
    for logger in loggers:
        url = getattr(getattr(logger, "experiment", None), "url", None)
        print(f"[ParTauDETR] logger: {type(logger).__name__}{f' -> {url}' if url else ''}")

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        # Log the LR next to the losses. OneCycleLR steps per batch, so an
        # unexplained change in the losses can only be attributed to (or cleared
        # of) the schedule if the LR curve is visible alongside them.
        LearningRateMonitor(logging_interval="step"),
        # Best by validation loss. Evaluated at the end of a validation pass,
        # which is the only time val_losses/* exist in callback_metrics.
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_losses/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            filename="ParTauDETR-model_best",
            save_on_train_epoch_end=False,
        ),
        # last.ckpt with optimizer and scheduler state, so a run that is
        # pre-empted or hits its wall clock can resume with `ckpt_path`. Kept
        # apart from the best-model callback because that one is weights-only
        # (a ModelCheckpoint has a single save_weights_only flag for both its
        # top-k and its last file). Costs one full checkpoint (~90 MB).
        ModelCheckpoint(
            dirpath=models_dir,
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
            save_on_train_epoch_end=False,
        ),
        # Stop on a non-finite OR a diverging validation loss. Nothing else
        # did: the 250-epoch run of 2026-09-16 went NaN at epoch 91 and ran 19
        # more hours on NaN weights. patience is effectively infinite, so this
        # never stops a run for plateauing -- it is purely the finiteness and
        # divergence check. training_step has the per-step counterpart for the
        # training loss, and ParTauDETR_module.on_before_optimizer_step has the
        # per-step gradient counterpart.
        #
        # check_finite alone was not enough for the 2026-09-20 run: the
        # gradients overflowed at step 62 949 but the val loss only went
        # 0.75 -> 5.34 -> 7.67 and stayed finite, so the callback never fired
        # and 11 600 dead steps followed. divergence_threshold catches that --
        # with mode="min" Lightning stops once the monitor rises ABOVE it, and
        # a healthy plateau here is ~0.73, so 3.0 leaves a 4x margin over
        # anything the run does while it is still learning.
        EarlyStopping(
            monitor="val_losses/loss",
            mode="min",
            patience=10**9,
            check_finite=True,
            divergence_threshold=_optional_float(
                cfg.model.detr.loss.get("val_loss_divergence_threshold", None)
            ),
            verbose=True,
        ),
        # Fallback: best by train loss. train_losses/* are logged with
        # on_epoch=True, so they only appear once the training epoch has been
        # reduced -- this must run at train epoch end, not at validation end.
        # Lightning otherwise infers this flag from val_check_interval and would
        # point both checkpoints at the same hook, where one of the two metrics
        # is always missing.
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="train_losses/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="ParTauDETR-model_best_train",
            save_on_train_epoch_end=True,
        ),
    ]

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator=check_accelerator(cfg),
        devices=cfg.training.trainer.devices,
        precision=str(cfg.training.trainer.precision),
        gradient_clip_val=None,
        num_sanity_val_steps=cfg.training.trainer.num_sanity_val_steps,
        enable_progress_bar=True,
        **_optional_trainer_kwargs(cfg),
    )

    started = time.perf_counter()
    inference = None
    try:
        trainer.fit(model=model, datamodule=datamodule)
        # After fit, so the timing reflects the trained model on the device it
        # trained on. Guarded inside the helper: a timing failure must not lose
        # the run's metrics.
        bench_cfg = cfg.training.get("inference_benchmark", None)
        if bench_cfg is None or bool(bench_cfg.get("enabled", True)):
            inference = measure_inference_time(
                model,
                datamodule,
                cfg,
                device=trainer.strategy.root_device,
                max_batches=int(bench_cfg.get("batches", 20)) if bench_cfg else 20,
                warmup_batches=int(bench_cfg.get("warmup", 3)) if bench_cfg else 3,
            )
            if inference is not None:
                print(
                    f"[ParTauDETR] inference: {inference['forward_us_per_jet']} us/jet, "
                    f"{inference['forward_jets_per_second']:,.0f} jets/s forward "
                    f"({inference['forward_fraction_of_wall']:.0%} of wall)",
                    flush=True,
                )
    finally:
        # Written in `finally` so a run killed by the wall clock still records
        # the best score it reached, which is the quantity the scaling study
        # needs. Must precede experiment.end(), which clears _experiment.
        try:
            write_run_metrics(
                cfg,
                trainer,
                datamodule,
                loggers,
                time.perf_counter() - started,
                os.path.join(cfg.output_dir, "metrics.json"),
                inference=inference,
            )
        except Exception as exc:  # pragma: no cover - never mask a training error
            warnings.warn(f"Could not write metrics.json: {exc}")
        # CometLogger.finalize() only flushes; an OfflineExperiment writes its
        # uploadable .zip on end(). Without this an offline run leaves nothing
        # behind, so end the experiment even if training raised.
        for logger in loggers:
            experiment = getattr(logger, "_experiment", None)
            if experiment is not None and hasattr(experiment, "end"):
                experiment.end()


if __name__ == "__main__":
    train()
