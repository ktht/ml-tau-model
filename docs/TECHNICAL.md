# ParTau: Technical Documentation

## Model Architecture

### Base: ParticleTransformer
The core backbone is a faithful implementation of the [ParT architecture](https://arxiv.org/abs/2202.03772):

- **`Embed`**: Particle feature embedding via BatchNorm → (LayerNorm → Linear → GELU) × N
- **`PairEmbed`**: Per-pair Lorentz-invariant interaction features ($\ln k_T$, $\ln z$, $\ln \Delta R$, $\ln m^2$) embedded via Conv1d layers and injected as attention bias
- **`Block`**: Standard Transformer blocks (multi-head self-attention + FFN) with learned residual scaling
- CLS token aggregation (BERT-style) produces a per-jet representation

Default configuration: `embed_dims=[256, 512, 256]`, `num_layers=2`, `num_heads=8`.

### Task Heads (ParTau)
Four task-specific CLS tokens independently attend to the shared backbone output. Each token then passes through a dedicated CLS block and normalization layer before reaching its respective readout head:

| Head | Output | Activation | Architecture |
|------|--------|------------|--------------|
| `tau_id_head` | Background vs Signal logits (2 classes) | Softmax | 1-layer FFN |
| `tau_charge_head` | Charge logit (positive score) | Sigmoid | 1-layer FFN |
| `classification_head` | Decay mode probabilities (6 classes) | Softmax | 1-layer FFN |
| `regression_head` | $[\log(p_T^\text{vis}/p_T^\text{jet}),\ \Delta\eta,\ \sin\Delta\phi,\ \cos\Delta\phi,\ \log(m_\text{vis}/m_\text{jet})]$ | None | 1-layer FFN |

### Training (PyTorch Lightning)
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-2`)
- **LR Schedule**: OneCycleLR (max learning rate `1e-3`)
- **Loss functions**:
  - Tau ID: `CrossEntropyLoss` (2-class) with label smoothing
  - Charge: `BCEWithLogitsLoss`
  - Decay mode: `CrossEntropyLoss`
  - Kinematics: Combined loss using `HuberLoss(δ=1.0)` for $p_T$, $\eta$, and $m$, plus a chord loss (L2 distance) for the $(\sin\phi, \cos\phi)$ vector. The mass component is weighted by $\lambda_m = 0.2$.
- **Conditional gating**: Auxiliary losses (charge, decay mode, kinematics) are multiplied by the truth tau label, so only signal jets contribute to those tasks.

## Input Features

17 features per candidate particle:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `cand_deta` | $\eta_\text{cand} - \eta_\text{jet}$ |
| 2 | `cand_dphi` | $\phi_\text{cand} - \phi_\text{jet}$ |
| 3 | `cand_logpt` | $\log(p_T)$ |
| 4 | `cand_loge` | $\log(E)$ |
| 5 | `cand_logptrel` | $\log(p_T / p_{T,\text{jet}})$ |
| 6 | `cand_logerel` | $\log(E / E_\text{jet})$ |
| 7 | `cand_deltaR` | $\Delta R(\text{cand}, \text{jet})$ |
| 8 | `cand_charge` | Particle charge |
| 9 | `isElectron` | \|PDG\| = 11 |
| 10 | `isMuon` | \|PDG\| = 13 |
| 11 | `isPhoton` | \|PDG\| = 22 |
| 12 | `isChargedHadron` | \|PDG\| = 211 (π±) |
| 13 | `isNeutralHadron` | \|PDG\| = 130 (K⁰L) |
| 14 | `cand_dz` | Longitudinal impact parameter $d_z$ |
| 15 | `cand_dz_error` | Significance of $d_z$ ($d_z / \sigma_{d_z}$) |
| 16 | `cand_dxy` | Transverse impact parameter $d_{xy}$ |
| 17 | `cand_dxy_error` | Significance of $d_{xy}$ ($d_{xy} / \sigma_{d_{xy}}$) |

A maximum of 20 candidates per jet are used (padded/clipped).

## (Tau) Decay Mode Mapping

| Class | HPS DM | Description |
|-------|--------|-------------|
| 0 | 0 | 1-prong, 0 neutrals |
| 1 | 1 | 1-prong, 1 π⁰ |
| 2 | 2, 3, 4 | 1-prong, ≥2 π⁰ |
| 3 | 5, 10 | 3-prong, 0 π⁰ |
| 4 | 6–9, 11–14 | 3-prong, ≥1 π⁰ |
| 5 | 15, 16, -1 | Rare / Leptonic / Other |

Rare decay modes (15), leptonic decays (16), and background (-1) are all mapped to class 5. Background samples are masked during training so only signal taus contribute to this head. Background is also tagged in a separate `tau_id_head`.

### ParTauDETR Daughter-Derived Modes

DETR derives its decay mode from the predicted daughter set, rather than the
six-class head above. The default daughter classes, in configuration order, are
charged hadrons, neutral hadrons, electrons, and muons. Their representative
PDG IDs are 211, 130, 11, and 13; signed input PDGs are classified by absolute
value. Older two-class configurations remain supported.

The shared decoder in `mltau/tools/evaluation/set_to_set_models.py` preserves
the upstream hadronic convention and extended rare ID 30, and adds:

| Derived DM | Category |
|------------|----------|
| -2 | Electronic tau decay |
| -3 | Muonic tau decay |

A leptonic mode requires exactly one electron or muon and no other visible
species, ignoring photons and neutrinos. Mixed hadron/lepton sets and multiple
leptons are classified as rare. The parquet column `gen_jet_tau_decaymode`
continues to use 16 for both leptonic modes; derived truth instead uses daughter
PDGs. Compare `tau_decaymode` with `gen_jet_tau_decaymode_exp`, not directly with
the stored parent label. Training metrics, threshold calibration, and the CLI
evaluator use the separate derived categories.

Both matched and soft parent DM losses exclude stored label 16 from their
targets and normalization because these losses model only hadronic counts.
Leptonic jets still supervise daughter identity, objectness, charge, kinematics,
tau ID, and any enabled parent charge/kinematics constraints.

Two optional topology penalties use the same soft objectness gate as the parent
constraints. With gate $g_q$, define $\ell_q = g_q[P_q(e)+P_q(\mu)]$ and
$h_q = g_q[P_q(\text{charged hadron})+P_q(\text{neutral hadron})]$:

- `model.detr.loss.weight_excess_leptons`: weights
  $[\max(0,\sum_q\ell_q-1)]^2$, counting electrons and muons together.
- `model.detr.loss.weight_mixed_lepton_hadron`: weights
  $\sum_{q\ne r}\ell_q h_r$, excluding same-query ambiguity.

Both default to `0.0`, so neither changes the optimization objective unless
enabled. Their unweighted values are still calculated and logged every epoch
as `train_losses/excess_leptons`, `train_losses/mixed_lepton_hadron`,
`val_losses/excess_leptons`, and `val_losses/mixed_lepton_hadron`. Each penalty
is averaged over genuine tau jets only; a background-only batch contributes
zero. Configurations without lepton classes also give zero. These are soft
penalties, not hard inference constraints.

Class-based mass decoding uses pion masses for the two hadron groups and
electron/muon masses for the lepton groups. `create_predictions` and the CLI
evaluator pass the configured class ordering automatically. Direct calls to
`get_predicted_particles(..., mass_from_class=True)` must also pass
`tau_daughter_pdg_ids=cfg.dataset.tau_daughter_pdg_ids` for the expanded head.
Without that argument, only the legacy two-class mass lookup is supported.

The four-class daughter head requires retraining. Existing checkpoints remain
usable with their original class configuration; loading a two-class checkpoint
into a four-class model requires explicit handling of incompatible head weights.

## Data

- **Format**: Apache Parquet files, streamed with `awkward-array` using row-group chunking
- **Dataset**: CLD detector simulation (key4hep framework), $e^+e^-$ collision events
- **Split**: Dataset files are provided in a 90/10 Train/Test split.
- **Batch size**: 12288

### Expected Parquet Fields

| Field | Description |
|-------|-------------|
| `reco_cand_p4s` | Jet constituent 4-momenta (px, py, pz, E) |
| `reco_cand_charges` | Candidate charges |
| `reco_cand_pdgs` | Candidate PDG IDs |
| `reco_jet_p4` | Reconstructed jet 4-momenta |
| `gen_jet_p4` | Generator-level jet 4-momenta |
| `gen_jet_tau_p4` | Generator-level visible tau 4-momenta |
| `gen_jet_tau_decaymode` | HPS decay mode index (−1 = background) |
| `gen_jet_tau_charge` | True tau charge |
| `cls_weight` | Per-jet event weight (optional) |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

| Package | Role |
|---------|------|
| `torch`, `torchvision` | Core deep learning |
| `pytorch-lightning` | Training loop management |
| `hydra-core`, `omegaconf` | Hierarchical configuration |
| `awkward` | Jagged particle physics arrays |
| `vector` | 4-vector kinematics |
| `numpy` | Numerical operations |
| `scikit-learn` | ML utilities |
| `tensorboard` | Training monitoring |
| `matplotlib`, `mplhep` | CMS-style physics plots |
| `boost-histogram` | Fast histogram filling |
| `onnx`, `onnxruntime-gpu` | Static model export and CPU/GPU inference benchmarking |

## ONNX Inference Runtime Benchmark

`mltau/scripts/benchmark_onnx.py` exports a fixed-shape fp32 ONNX graph and
benchmarks it with ONNX Runtime. The supported targets are:

| Command model | Python model | Output |
|---------------|--------------|--------|
| `singlepartau` | `ParTau` from `mltau/models/SingleParTau.py` | Selected task head |
| `multipartau` | `ParTau` from `mltau/models/MultiParTau.py` | All four task heads |
| `mixer` | `MixerTau` from `mltau/models/MixerTau.py` | Selected task head |
| `all` | All three models above | One result per model |

`partau` remains an alias for `singlepartau`. The default graph uses a batch
size of 1, 17 input features, and 16 particles per jet. All dimensions are
static in the exported graph.

### ONNX Runtime installation

Install the project requirements:

```bash
pip install -r requirements.txt
```

The requirements use `onnxruntime-gpu`, not the CPU-only `onnxruntime`
package. `onnxruntime-gpu` includes both `CUDAExecutionProvider` and
`CPUExecutionProvider`, so the same installation runs both runtime targets.
Do not install `onnxruntime` and `onnxruntime-gpu` in the same environment.

### Benchmark all models on CPU and GPU

Use `all` to export and benchmark SingleParTau, MultiParTau, and Mixer in one
run. By default, both CPU and GPU are benchmarked:

```bash
PYTHONPATH=. python3 mltau/scripts/benchmark_onnx.py all \
  --iterations 500 \
  --num-particles 32
```

This writes `singlepartau_static_fp32.onnx`,
`multipartau_static_fp32.onnx`, and `mixer_static_fp32.onnx`. The `all`
target cannot be combined with `--output` or `--checkpoint`; benchmark an
individual model when either option is needed.

The CPU session is explicitly restricted to sequential execution with one
intra-op thread and one inter-op thread. The GPU session uses
`CUDAExecutionProvider` and ONNX Runtime I/O binding. Its reported latency
covers inference with inputs and outputs resident on the GPU; host-to-device
and device-to-host transfer time is excluded.

## Training

### Local / Interactive

```bash
./run.sh python3 mltau/scripts/train.py
```

`run.sh` wraps the command in an Apptainer (Singularity) container with the pinned PyTorch environment.

### HPC Cluster (SLURM)

```bash
sbatch train-gpu.sh
```

Requests a GPU node (RTX, 40 GB) and logs to `logs/slurm-{name}-{jobid}-{node}.out`.

### Hydra Config Overrides

Training parameters can be overridden from the command line via Hydra:

```bash
./run.sh python3 mltau/scripts/train.py training.lr=5e-4 training.max_epochs=200
```

Configuration files live under `mltau/config/`:

| File | Contents |
|------|----------|
| `main.yaml` | Top-level config, composes dataset + training + metrics |
| `dataset.yaml` | `max_cands`, `data_dir`, train/val/test split ratios |
| `training.yaml` | `lr`, `max_epochs`, `batch_size`, `num_workers` |
| `metrics/` | Plot styles, axis settings, and working points for all tasks |


### Outputs

```
{output_dir}/
  models/       # ModelCheckpoint saves (monitors val_losses/loss)
  logs/         # Training logs
  tensorboard/  # TensorBoard event files
```

## Evaluation & Metrics

Metrics are computed and logged to TensorBoard every epoch:

### Tau Tagging
- ROC curve
- Tau efficiency vs. $p_T$, $\eta$, $\theta$ at three working points (loose / medium / tight)
- Jet fake rate vs. $p_T$, $\eta$, $\theta$
- Scalar metrics at the medium WP: accuracy, precision, recall, F1, TPR, TNR, FPR, FNR

### Kinematics
- Response (median) and resolution (IQR/median) vs. $p_T$, $\theta$, $\phi$, $\eta$, $m_\text{vis}$
- 2D resolution plots per variable

### Decay Mode
- Confusion matrix
- General classification metrics

### Charge ID
- Performance metrics vs. $p_T$, $\eta$, $\theta$
- Baseline comparison with jet charge Q*κ method
- Confusion matrix analysis with 95% average efficiency working point

## Huggingface

The latest model weights are available from huggingface at https://huggingface.co/HEP-KBFI/fcc-tau/tree/main/cld/qq_vs_z_91gev/0612.
They can be upoaded with the following script, after authenticating
```
uv run python3 mltau/scripts/upload_model_hf.py --path-prefix cld/qq_vs_z_91gev/0612 outputs/0612_multipartau_full_b8483f6
```

## Project Structure

```
mltau/
  config/           # Hydra configuration files
  models/
    ParticleTransformer.py   # Base ParT implementation
    MultiParTau.py           # Multi-task extension with 4 output heads
    MultiParTau_module.py    # Lightning module for multi-task ParTau
    SingleParTau.py          # Single-task ParTau model
    SingleParTau_module.py   # Lightning module for single-task ParTau
    HPS.py                   # Baseline HPS model definition
  scripts/
    train.py                 # Main training entry point
    run_inference.py         # Main inference entry point
    upload_model_hf.py       # Upload to HuggingFace Hub
    HPS/                     # Baseline HPS processing scripts
  tools/
    features.py              # Math and kinematic utilities
    general.py               # 4-vector normalization and DM mapping
    losses.py                # Unified TauLoss and FocalLoss implementations
    evaluation/              # Per-task evaluation logic
    io/                      # Data loading and preprocessing logic
    logging/                 # TensorBoard metric loggers per task
    optimizers/              # Optimizer wrappers (e.g., Lookahead)
    logging/                 # TensorBoard metric loggers per task
    optimizers/              # Optimizer wrappers (e.g., Lookahead)
```
