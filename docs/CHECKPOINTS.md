# Checkpoints, inference contracts and legacy weights

Evaluation, image/folder inference, the Gradio demo and export use
`src.utils.checkpoint.load_checkpoint`. The shared bundle restores the model,
deterministic transform, age decoder, runtime config and device. Models are built
with `pretrained=False` and weights are loaded strictly; there are no downloads
and no `strict=False` compatibility shortcuts.

## New artifacts: schema version 1

New Trainer checkpoints are created through `create_checkpoint`:

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1`; unknown, null and boolean versions are rejected |
| `model_spec` | Architecture kind, backbone, output width, dropout and pretraining setting |
| `preprocessing` | RGB, square size, bilinear resize, antialias policy, mean and std |
| `predictor` | Loss/decoder mode, inclusive age support and inference postprocessing |
| `cfg` | Recursively resolved plain config; no nested OmegaConf nodes |
| `model_state_dict` | Weights matching the declared architecture exactly |
| `epoch`, `val_mae` | Optional training metadata; not inferred from filenames |
| `optimizer_state_dict` | Optional AdamW state; restored only with complete P2.1 training state |
| `training_state` | Optional training-state version 1: boundary counters, scheduler/scaler/RNG/loaders/contract/history; required for resume |
| `experiment_metadata` | Optional data/manifest hashes, source declarations, runtime/Git/device and randomness policy from new Trainer runs; old conversion leaves provenance unknown |
| `training_objective` | Optional declared objective name/revision/dtype, recorded by new Trainer runs; absent metadata is unknown |

The explicit contract must agree with the saved config. Missing inference metadata,
unsupported architecture/version, or incompatible state shapes fail clearly.
P2.1 Trainer outputs also contain a versioned `training_state` with complete
optimizer/scheduler/scaler/RNG/loader state. Earlier schema-v1 inference artifacts
remain valid for inference but are not automatically resumable. See
[TRAINING.md](TRAINING.md) for validation, best/last pairing and epoch-boundary scope.

## Override precedence

1. Bundled defaults provide noncontract settings.
2. Saved config and its validated contract determine architecture, decoder and
   preprocessing. Local edits to `configs/default.yaml` do not change loading.
3. Explicit runtime YAML/CLI settings override only these fields:
   `device`, `data.root_dir`, `data.csv_dir`, `data.num_workers`,
   `training.eval_batch_size`, `training.use_amp`.

`--config` now means runtime overrides when loading a checkpoint. It may repeat a
saved field unchanged, but attempting to change a protected setting, including
image size, normalization, output width or loss mode, is rejected. Use a partial
runtime YAML rather than an unrelated full experiment config. Interpolations are
resolved against the saved config. An explicitly supplied CLI device wins; when
none is selected, inference auto-detects CPU/CUDA rather than reusing the training
machine's GPU index. Export currently runs on CPU.

For compatibility with old callers, explicitly passing an unmodified copy of the
bundled default YAML is treated as fallback defaults: its model/loss/preprocessing
values do not replace the saved contract. This exception requires exact content
equivalence after YAML parsing, not a particular filename; customized contract
conflicts still fail.

For example, `runtime.yaml`:

```yaml
device: cpu
data:
  root_dir: /path/to/utkcropped
  csv_dir: /path/to/splits
  num_workers: 0
training:
  eval_batch_size: 32
```

```bash
python scripts/inference.py --checkpoint /path/to/model.pt --image img_test/30_1_2.jpg --device cpu
eval-age --checkpoint /path/to/model.pt --config runtime.yaml --bias
eval-age --checkpoint /path/to/model.pt --device cpu \
  --root-dir /path/to/utkcropped --csv-dir /path/to/splits --num-workers 0
python demo/app.py --checkpoint /path/to/model.pt --device cpu
```

The saved train/val/test CSV protocol is not rebuilt by any of these commands.
Evaluation still uses the shared loader builder for all three CSVs. Its training
split still shuffles/drops the final batch; full split-independent evaluation is
P2.2 work. All three evaluation transforms now use the artifact preprocessing.

## Historical raw ResNet50 state_dict

The repository's tracked `checkpoints/epoch-16-loss_valid-4.73.pt` predates the new
head and wrapper format. Select its historical profile explicitly:

```bash
python scripts/inference.py \
  --checkpoint checkpoints/epoch-16-loss_valid-4.73.pt \
  --legacy-resnet50 --device cpu --image img_test/30_1_2.jpg

python demo/app.py --checkpoint checkpoints/epoch-16-loss_valid-4.73.pt \
  --legacy-resnet50 --device cpu
```

The profile is recovered from repository commit `432569f`, which introduced that
weight: ResNet50, `Dropout(0.2) → Linear(2048,256) → Linear(256,1)` with no hidden
activation, resize 128×128 and ImageNet mean/std. The historical model and decoder
return raw scalar predictions, including values outside [0,100]. Its distinct head and raw decoder are preserved. Modern regression bounding is
applied by the decoder, outside the model training graph.

This is an explicit repository-history compatibility profile, not an embedded
training manifest. Do not apply it to arbitrary shape-compatible weights with
unknown preprocessing. No age, epoch or benchmark result is parsed from a filename;
`4.73` is historical validation metadata, not a verified test MAE.

To repackage an already verified legacy file without changing weights or behavior:

```python
import torch
from src.utils.checkpoint import load_checkpoint, create_checkpoint

bundle = load_checkpoint(
    "checkpoints/epoch-16-loss_valid-4.73.pt", legacy_resnet50=True, device="cpu"
)
torch.save(create_checkpoint(bundle.model, bundle.cfg), "/tmp/legacy_resnet50_v1.pt")
# The v1 artifact carries the historical architecture/profile and needs no raw-state flag.
restored = load_checkpoint("/tmp/legacy_resnet50_v1.pt", device="cpu")
```

The source checkpoint is never overwritten automatically. A repackaged artifact
does not establish new training provenance or benchmark performance.

## Unversioned wrappers and pickle policy

An unversioned wrapper with a complete saved inference config can be adapted with
a visible warning. Plain wrappers saved after P1.2 load with the restricted
deserializer. Wrappers with missing config are rejected rather than guessed from
defaults. Optional old `model_name`/`output_dim` headers must agree with the config.

All ordinary loads explicitly use `torch.load(..., weights_only=True)`. Failure
never triggers an automatic unsafe retry. Some older wrappers contain pickled
OmegaConf objects; only for an artifact whose origin you trust, opt into
`--trust-legacy-pickle`. This permits arbitrary pickle code to execute while
loading. The flag is separate from `--legacy-resnet50`: the tracked raw tensor
state_dict uses the restricted loader and does not require unsafe pickle.

After inspecting a trusted old wrapper, `create_checkpoint` can resave it in the
plain v1 format. Keep the original file until the new artifact's predictions have
been checked. Restricted loading is not a universal sandbox for arbitrary files;
see the [official serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## Export contract

```bash
export-age --checkpoint /path/to/model.pt --output /tmp/model.jit
```

Exports contain model logits. `/tmp/model.jit.json` records `model_spec`,
`preprocessing`, `predictor` and `output=logits`; preprocessing and age decoding are
not silently folded into the graph. `--img-size` may repeat the saved resolution
but cannot replace it. TorchScript parity is tested locally on a small CNN;
ONNX/runtime verification and a wider deployment matrix remain P2.4 work.

## Verification scope

Tests cover regression/distribution round trips, shifted age support, nondefault
resize/normalization, strict weight loading, unsupported metadata, runtime override
policy, no unsafe fallback, trusted old OmegaConf wrappers, synthetic historical
weights, Trainer output, CLI/evaluation/demo agreement and TorchScript metadata/parity.
The Gradio comparison is an optional-extra test and skips when Gradio is absent.
The real tracked legacy weight was also loaded locally on CPU and used for one
image prediction. This does not measure test-set accuracy, fairness, speed or GPU
behavior. No new pretrained weights or datasets were downloaded.

## P1.4 loss and export update

New regression models return raw logits; the modern decoder still applies
`clamp_0_100`, so old schema-v1 artifacts retain their final-age policy. Distribution
moments and modern decoding now use float32. Both `l1_mean_variance` and its old
`mean_variance` alias are supported without rewriting old artifact labels.
Exports contain raw regression logits and must be decoded using their sidecar.
New training runs record objective revision 2; generic old-weight repackaging does
not guess a training revision. See [LOSSES.md](LOSSES.md).


## P1.5 experiment provenance

The training entry point verifies split CSVs and a v2 data manifest before model
construction, then passes provenance to Trainer. `experiment_metadata` is optional
training metadata and does not alter the schema-v1 inference contract. Loading or
converting old checkpoints does not infer source/version, split seed or image hashes.
P2.1 now restores completed-epoch RNG/generator positions; older artifacts retain unknown resume state. See
[DATA_PROTOCOL.md](DATA_PROTOCOL.md) for the recorded fields and verification scope.


## P2.1 training artifacts

Fresh Trainer runs write individually atomic `best.pt` and `last.pt` under a
unique run directory. They preserve the schema-v1 inference contract and are still
restricted-loadable by the common prediction/evaluation/export pipeline. Training
resume uses a separate strict reader; it does not change inference runtime override
policy. NumPy RNG keys are stored as tensors, not pickled NumPy objects. Copy both
best and last to relocate a resumable run. See [TRAINING.md](TRAINING.md).
