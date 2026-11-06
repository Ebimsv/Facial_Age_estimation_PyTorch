# Validated experiment configuration

`load_config()` now merges packaged defaults, a selected YAML file, and CLI
overrides into an OmegaConf structured schema. Unknown keys, invalid types,
nonfinite values and incompatible model/loss settings fail before training seeds,
data loading or model construction. Values are resolved once and the resulting
configuration is read-only. This is configuration validation, not a guarantee of
model accuracy or exact experiment reproducibility.

## Defaults, partial YAML and overrides

The packaged default is `src/configs/default.yaml`. Without `--config`, source
execution uses the editable compatibility copy `configs/default.yaml` over the
packaged defaults. Installed wheels use their bundled YAML. Explicit `--config`
selects that file instead of the checkout copy; partial YAML files inherit missing
settings from the packaged defaults. Both committed copies of the defaults must
remain identical.

For example, save this partial file as `experiment.yaml`:

```yaml
model:
  name: efficientnet_b0
  pretrained: false
training:
  epochs: 10
  batch_size: 16
device: cpu
```

Inspect the final configuration without opening images, downloading weights or
starting training:

```bash
python scripts/train.py --config experiment.yaml --print-config
train-age --config experiment.yaml --print-config training.batch_size=8
```

CLI overrides are validated together, after merging. Regression needs both mode
settings, not only one:

```bash
train-age --print-config model.output_dim=1 loss.type=l1
```

`training.epohs=10` is rejected as an unknown key. `model.output_dim=1` alone is
rejected because the default distribution decoder expects 101 logits. Explicit
`???` values, broken interpolations and invalid `key=value` syntax are rejected.
Numeric/string conversions follow OmegaConf's structured type rules; compatible
interpolations such as `optimizer.head_lr=${optimizer.backbone_lr}` are supported.

Python callers can use:

```python
from src.configs import load_config
from src.utils.config import merge_config, to_plain_config

cfg = load_config("experiment.yaml", overrides=["training.epochs=20"])
cfg = merge_config(cfg, {"training": {"batch_size": 8}})
plain = to_plain_config(cfg)  # resolved nested dict/lists, suitable for JSON
```

Do not mutate `cfg` in place after validation. Checkpoints and W&B receive the
resolved plain configuration rather than a shallow dictionary of DictConfig nodes.
Checkpoint schema v1 and shared inference restoration are now implemented; see
[checkpoint contracts](CHECKPOINTS.md) for the separate runtime override policy.

## Model and age contracts

| Setting | Contract |
|---|---|
| Backbone | `mobilenet_v3`, `efficientnet_b0`, `efficientnet_b3`, `resnet50`, `vit_small` |
| Distribution mode | `loss.type=mean_variance`; output width equals `end_age-start_age+1`; at least two classes |
| Regression mode | `loss.type=l1`; exactly one output; current supported bounds are fixed at `[0,100]` |
| Age support | Inclusive, nonnegative integer `start_age <= end_age` |
| Loss weights | Finite and nonnegative; at least one positive weight in distribution mode |
| Input size | Integer >= 32; ViT-S/14 additionally requires a multiple of 14 |
| Dropout | Finite, `0 <= dropout < 1` |
| Normalization | Three finite RGB means and three finite, strictly positive standard deviations |

ViT uses the configured resolution instead of a hardcoded 224. For example,
`model.name=vit_small data.img_size=280` is accepted and
`model.name=vit_small data.img_size=192` fails before backbone creation. Tests use
`pretrained=false` and perform real forwards at 224 and 280 without downloads.

For a shifted distribution support, update all related settings together:

```bash
train-age --print-config loss.start_age=10 loss.end_age=20 \
  model.output_dim=11 data.max_age=21
```

Losses check logit width and target batch shape. A one-column regression tensor
cannot silently broadcast across the distribution's age classes. Targets must be
finite, match the batch size, and lie inside the inclusive support. Training and
evaluation also validate CSV ages before image I/O. Regression forward clamping
and the mean-variance variant's mathematics are unchanged in P1.2; P1.4 addresses
those separately. Demo plots now use the checkpoint's saved support, and all
checkpoint-backed commands restore the same preprocessing/decoder.

## Training and loader checks

Epochs and both batch sizes must be positive integers; workers are nonnegative.
The seed is in `[0, 2**32-1]`. Learning rates are finite and positive; weight decay
and gradient clipping are finite and nonnegative. `scheduler.pct_start` is strictly
between 0 and 1. The device is null for auto selection, `cpu`, or `cuda[:index]`.
CUDA requests during training fail clearly if CUDA is unavailable.

Empty train/validation/test splits are rejected. Training currently uses
`drop_last=True`, so a training set smaller than its batch size is rejected before
model construction. Once loader length is known, the zero-length OneCycleLR
warmup boundary (`epochs * batches_per_epoch * pct_start == 1`) is also rejected.
No invalid run should be reported as a zero-loss successful epoch.

## Preparation settings do not resplit training data

`data.max_age`, `data.train_ratio` and `data.val_ratio` are preparation-only
parameters. Changing them in a training command does not rebuild or filter the
existing split CSVs. The preparer now consumes the same validated config:

```bash
python scripts/prepare_data.py --config experiment.yaml --print-config
python scripts/prepare_data.py --config experiment.yaml \
  --dataset-dir /path/to/utkcropped --csv-dir /path/to/new-csvs \
  --train-ratio 0.8 --val-ratio 0.1 --seed 42
```

Explicit preparation flags override YAML values. Both ratios must be positive,
their sum must be < 1, and the remaining test fraction must stay positive under
the preparer's existing four-decimal rounding. Actual sample counts may differ
from nominal ratios because scikit-learn rounds the two split stages to integers.
The existing two-stage split algorithm has not been replaced.

Preparation filters ages to `[loss.start_age, data.max_age)`; the exclusive
`data.max_age` must be greater than start_age and at most end_age+1. Successful
preparation writes `preparation_manifest.json` with effective parameters, resolved
config and actual sample counts. This parameter record does not yet contain
dataset-content hashes, identity annotations or a complete provenance protocol;
those are P1.5 work. Existing tracked CSVs have not been regenerated.

## Migration of previously ignored settings

- `logging.save_best_only=true` is removed with a visible FutureWarning because
  current training always saves only the best validation model. Setting it false
  is rejected; saving every epoch is not implemented by this change.
- A legacy `inference` block containing only the known null defaults is removed
  with a FutureWarning. Non-null values or unknown keys are rejected. Use the
  inference CLI's `--checkpoint`, `--image`/`--image-dir`, and `--output` flags.

Legacy weights require the historical head/profile, now supported explicitly by
`--legacy-resnet50`; changing output_dim alone is insufficient. See
[checkpoint migration](CHECKPOINTS.md), [the audit](../audit/REPOSITORY_AUDIT.md)
and [the roadmap](../audit/IMPLEMENTATION_ROADMAP.md).
