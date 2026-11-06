# Installation and verified environments

## Install into an existing environment

An existing Conda environment can be used; creating another environment or
reinstalling PyTorch is unnecessary when compatible dependencies already work.

```bash
conda activate <your-environment>
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -m pip install .

# Optional components:
python -m pip install ".[dev]"          # pytest, build, linting tools
python -m pip install ".[data]"         # optional Kaggle SDK; never installed by downloader
python -m pip install ".[demo,wandb]"   # Gradio and W&B
```

Use a compatible torch/torchvision pair for your platform and CUDA driver, from
the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
For an offline install with dependencies already present, use
`python -m pip install --no-deps --no-build-isolation .`; this assumes the declared
build tools (`setuptools>=68` and `wheel`) are already available. This command
does not verify or install missing runtime dependencies.

`requirements.txt` installs dependencies only. Installing `.` also installs the
Python packages, the bundled configuration, and these console commands:

```bash
train-age --help
eval-age --help
export-age --help
```

Installing the core package does not require Kaggle, Gradio or W&B. See
[download guardrails](DOWNLOADING.md) for read-only checks and optional SDK setup. Dataset downloading
and training are separate operations; installation does not fetch datasets or
model weights. See [checkpoint loading and legacy profiles](CHECKPOINTS.md) for
the shared loader, restricted deserialization and explicit historical weight support.

## Configuration and relative paths

Configurations now use a typed, resolved, read-only schema. Partial YAML files and
joint CLI overrides are supported; see [configuration contracts](CONFIGURATION.md)
for model/loss compatibility, preparation-only settings and legacy-key migration.
Use `train-age --print-config` to inspect settings without starting training.
See [loss definitions and numerical policy](LOSSES.md) for the default L1+variance
variant, raw regression training and modern inference bounds.
See [data protocol and reproducibility](DATA_PROTOCOL.md) before preparing a fresh
split directory; tracked historical CSVs are preserved.

- The packaged default configuration is `src/configs/default.yaml` and is loaded
  through `importlib.resources`; an installed wheel needs no repository checkout.
- `configs/default.yaml` remains the editable source-checkout compatibility copy.
  Source execution uses it by default. Contributors changing defaults must update
  both copies; a regression test enforces equality of the committed defaults.
- `--config /path/to/custom.yaml` always loads the explicitly selected file.
- Training and evaluation resolve relative dataset/CSV paths against the checkout
  for source execution, and against the current working directory for an installed
  package. Absolute dataset paths also work. The configuration file's directory
  is not automatically used as the dataset root.
- Checkpoint-backed commands restore their preprocessing/decoder from the artifact.
  Their `--config` accepts runtime overrides under the policy in
  [CHECKPOINTS.md](CHECKPOINTS.md); training config behavior is described separately.

For example, after installing the package, create a working directory with your
data and CSVs and run:

```bash
train-age --config /absolute/path/to/experiment.yaml
```

## API minimum versus tested environments

The declared API minimum is PyTorch 2.3 / torchvision 0.18. The trainer uses
`torch.amp.GradScaler("cuda", ...)`, which is present in the
[official PyTorch 2.3 source](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/amp/__init__.py).
The [official previous-version instructions](https://pytorch.org/get-started/previous-versions/)
list compatible torch/torchvision pairs. Dependencies using only lower bounds
are not a promise that every possible combination is tested.

Local verification on 2026-11-06 used existing Conda environments, without changing
them or downloading dependencies:

| Check | Python | torch | torchvision | timm | Result |
|---|---|---|---|---|---|
| Full suite, artifact build, temporary installed wheel, CLI smoke | 3.13.9 | 2.9.0+cu128 | 0.24.0+cu128 | 1.0.28 | CPU tests; see the P1.1 verification record below |
| Independent runtime import smoke | 3.11.0 | 2.11.0+cu128 | 0.26.0+cu128 | 1.0.7 | Trainer, torch.amp.GradScaler import and disabled scaler construction succeeded |

The full-suite environment also had OmegaConf 2.3.0, TensorBoard 2.20.0,
setuptools 80.9.0, wheel 0.45.1, build 1.4.0, pytest 9.0.1, and Ruff 0.16.3.
The CUDA wheel suffix alone does not prove GPU testing. In the P1.5 full-suite run,
all **224 tests passed**, including two CUDA autocast/GradScaler tests on an NVIDIA
GeForce RTX 5070 Laptop GPU, plus multiworker/spawn data tests. This required access
to system shared memory/device resources outside the restricted sandbox; earlier
restricted runs could not see CUDA and skipped those tests. A test-only controlled
initial scale avoids confusing FP16 startup overflow with objective instability.
This is not an accuracy benchmark or full GPU training/repeatability experiment.
P1.6 expanded the suite to **268 passing tests**, including 44 fake-API/ZIP download
cases and distribution credential canaries; no real Kaggle download/authentication
was used. See the [Phase 1 report](../audit/PHASE_1_IMPLEMENTATION_REPORT.md). The exact PyTorch 2.3 minimum
was verified against the upstream API source, but not executed locally: no existing
2.3 environment was found. A minimum-version runtime CI job remains P3.2 work.
The current CI matrix covers Python 3.10, 3.11, and 3.12; its remote result must be
checked on the pushed commit rather than inferred from local tests.

## Verify a checkout

```bash
python -m pytest tests/ -v
ruff check src/ scripts/ demo/ tests/
```

Packaging tests require the development build tools. They build a wheel and an
sdist in a temporary source tree, rebuild the wheel from the sdist, install it
with `--no-index --no-deps` into a temporary venv, and run the installed CLI from
a directory outside the checkout. The venv inherits existing runtime dependencies;
this proves installation of this project, not a fresh download/resolution of all
third-party dependencies. No user Conda environment is modified.

The existing `Readme.md` filename and license table are retained for compatibility.
Some setuptools versions emit a conventional-README filename warning and an SPDX
license-metadata deprecation warning; the artifacts still build and include the
declared README and MIT license. These warnings are not missing runtime resources.

To inspect build artifacts manually:

```bash
python -m build --no-isolation --outdir /tmp/facial-age-dist
```

The sdist and wheel must include the bundled YAML and CLI packages. Checkpoints,
datasets, TensorBoard logs, the audit directory, and Kaggle credentials must not
be included. Distribution tests check metadata alignment with `requirements.txt`.

## P1.1 verification record

This packaging step does not report model accuracy, throughput, or GPU training
results. Verification consists of the existing unit suite plus packaging regression
tests, metadata validation, artifact content inspection, and Ruff. See the P1.1
through P1.6 completion notes in [the implementation roadmap](../audit/IMPLEMENTATION_ROADMAP.md)
for historical and current test counts and remaining scope.
