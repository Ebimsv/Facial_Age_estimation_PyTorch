# Data validation and split provenance (P1.5)

## Prepare a new protocol

For the default pinned Hugging Face download, the optional Kaggle alternative,
or a read-only local dataset check, see
[DOWNLOADING.md](DOWNLOADING.md). That check is a sanity check, separate from this
preparation/split protocol; it does not regenerate CSVs.

Keep the committed `csv_dataset/*.csv` as the historical split. Their sizes are
15,822 train / 2,793 validation / 4,654 test, with 23,269 master rows. The default
80/10/10 settings do not describe that historical protocol. Preparing new data
never overwrites a nonempty output directory; choose a fresh directory each time.

```bash
python scripts/prepare_data.py \
  --dataset-dir /absolute/path/to/utkcropped \
  --csv-dir csv_dataset/protocol_v2 \
  --source '<dataset identifier or URL>' --dataset-version '<release or unknown>' \
  --train-ratio 0.8 --val-ratio 0.1 --seed 42 --check-decode --plot

python scripts/train.py \
  data.root_dir=/absolute/path/to/utkcropped data.csv_dir=csv_dataset/protocol_v2
```

The source and version are user declarations, not proof of origin or permission.
They default to `unknown`; no version is inferred from a folder name. Preparation
paths resolve against the working directory. Training paths follow the existing
[source/installed path rules](INSTALLATION.md). For paths containing spaces, quote
the complete `key=value` training override. No dataset is downloaded by preparation.

## Validation and rejection policy

Preparation scans a single folder in sorted filename order. It accepts regular
JPG/JPEG/PNG files, including UTKFace `.jpg.chip.jpg` names, with at least four
underscore-separated fields. Age, gender and ethnicity come from the first three
integer fields. The fourth field must be nonempty; timestamps are not identities.
Gender codes must be 0/1, ethnicity codes 0–4, and age nonnegative. Accepted age
is filtered to `[loss.start_age, data.max_age)` (upper bound exclusive).
Invalid labels are rejected rather than mapped to a demographic group.

Byte SHA-256 hashing is enabled by default. **Every member of an exact-byte
duplicate group is excluded**, including the first member. This avoids arbitrarily
choosing a label when copies have conflicting annotations. Inspect `duplicate_groups`
and the rejection CSV before adopting a protocol. This policy can change sample
counts and distributions compared with historical CSVs; old metrics do not transfer.
`--skip-content-hash` explicitly disables this filtering and records content checking
as unavailable. Exact bytes do not detect differently encoded or near-duplicate faces.

`--check-decode` additionally opens and fully converts candidates to RGB with Pillow;
it is independent of hashing. Without it, a corrupt file with a valid name can
enter the protocol and fail later on dataset access. `UTKDataset.__getitem__` closes
image handles and reports the failing image when decoding fails. Decode checking
is not a face detector, quality annotation review or a guarantee about future edits.

CSV loading checks required columns, finite in-range ages, known demographic
strings, duplicate names, canonical relative paths, allowed suffixes and file
existence. Absolute paths, traversal and symlinks escaping the image root are
rejected. CSV ages may be fractional within support; training reads CSV labels
and does not silently reconstruct them from filenames. Extra CSV columns are
ignored by the training dataset. Direct dataset use without an age support still
rejects negative/nonfinite ages; the training/evaluation entry points pass support.

## Split algorithm and failure behavior

The new protocol is `utkface_age_stratified_v2`. It sorts by image name, holds out
test, then divides the remainder into training/validation with scikit-learn age
stratification and the same seed. Stage sizes use the ceiling of exact decimal
fractions, passed as integer counts, avoiding a four-decimal cutoff or a binary
rounding artifact that adds an image. Validation's relative share is
`val_ratio / (train_ratio + val_ratio)`; integer allocation means actual ratios can
differ slightly. Output rows are sorted by name.

Each observed age must remain represented in both outputs of each stage. Sparse
classes and insufficient held-out sizes fail with counts and stage information;
there is no automatic random split, coarser age binning or sample duplication.
Choose a documented age filter/ratio or collect more samples, and use a fresh
output path. Split feasibility is checked before any output is created. A failed
scan reports rejection reasons in the exception; successful runs persist the report.
A write/plot failure can leave a partial output directory: inspect it and use a new
path; this step does not implement transactional directory publication.

## Generated artifacts and training checks

| Artifact | Purpose |
|---|---|
| `utkface_dataset.csv` | All accepted records |
| `train_set.csv`, `valid_set.csv`, `test_set.csv` | Nonempty, disjoint partitions whose rows/labels equal master |
| `rejected_samples.csv` | Excluded entry names and reasons, including age filters and duplicates |
| `split_manifest.json` | Schema 1; protocol, source/version, seed, filters/ratios, counts, CSV byte hashes, image hashes, duplicate groups, validation scope, environment and Git revision |
| `preparation_manifest.json` | Retained parameter/config summary for earlier consumers; the split manifest is the provenance contract |
| `split_distributions.png` | Optional histogram for this generated protocol |

The split manifest omits wall-clock timestamps and output path so repeat runs on
the same input/environment/code have the same semantic manifest. Source root is
recorded; moving the input folder changes this metadata. Git revision and whether
tracked files are modified are recorded when a checkout is available; installed
wheels may report unknown Git information. Pin the environment when comparing
splits across machines or scikit-learn versions. Save both manifests with a run.

`build_dataloaders` checks overlap by name/resolved path, and union/label equality
when a master CSV is present. With a v2 manifest it verifies current CSV hashes/
counts and rehashes image bytes before constructing loaders. A changed image or
CSV fails before model construction in the training entry point. Hash verification
adds startup I/O; it does not happen per batch. Files must stay unchanged during
the run. Hashes detect accidental changes relative to the manifest; they do not
authenticate the manifest author.

Unversioned CSVs remain usable after schema/path/disjointness checks. Their current
CSV hashes are recorded as `unversioned_csv`, with content and identity checks
unknown; metadata does not invent their original seed, source or training history.
No manifest is silently written beside legacy data. Direct `UTKDataset` validates
one CSV only; cross-split/manifest checks belong to `build_dataloaders`.

The training entry point passes loader provenance to `Trainer`. New checkpoints
and W&B metadata include `experiment_metadata`: data/manifest hashes and declared
source, Python/dependency versions, Git state, device and actual determinism flags.
Generic checkpoint conversion does not invent it; a Trainer constructed without
loader provenance records data as unknown. Schema-v1 inference remains compatible.

## Randomness scope

Loaders have separate PyTorch generators seeded with `seed`, `seed+1`, `seed+2`.
Validation iteration does not consume training's generator. A top-level,
spawn-compatible worker initializer seeds global Python and NumPy RNGs from
PyTorch's worker seed. Independently constructed NumPy `Generator` objects or
custom libraries still need explicit seeds. With zero workers, augmentation uses
the main process RNGs initialized by `set_seed`.

```bash
# Optional strict PyTorch determinism; unsupported operations can raise.
PYTHONHASHSEED=42 python scripts/train.py training.deterministic=true \
  data.csv_dir=csv_dataset/protocol_v2 data.root_dir=/absolute/path/to/utkcropped
```

The default remains strict algorithms disabled, cuDNN deterministic enabled and
cuDNN benchmarking disabled. Strict mode requests deterministic PyTorch algorithms
and supplies a default `CUBLAS_WORKSPACE_CONFIG=:4096:8` before normal training CUDA
initialization (an explicit existing value is preserved). Set CUDA requirements
before initializing CUDA in custom callers. `PYTHONHASHSEED` must be set before
starting Python; runtime assignment cannot change its existing hash seed.

These settings aim for repeatability within a fixed runtime, hardware, worker count
and execution sequence; they do not promise identical results across devices or
versions. Generator/RNG checkpoint restore and exact resume remain P2.1. See
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [scikit-learn split API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

Tests use synthetic images for parsing/rejections, duplicate groups, path/symlink
checks, sparse ages, manifest repeatability, tampering, checkpoint round trips and
multiple RNG streams with zero/two workers and spawn. The committed legacy CSV
partition is checked without rewriting it or requiring its images. Identity leakage,
near-duplicate leakage, annotation quality, real-data preparation and GPU repeatability
remain unverified; no new MAE or pretrained artifact is reported by this step.


P1.5 local verification on 2026-11-06: 224 tests passed in the existing Python
3.13.9 / torch 2.9.0+cu128 environment, including both CUDA loss tests on an RTX
5070 Laptop GPU. Reproducibility tests exercise CPU workers/spawn; they do not
establish deterministic full GPU training. No dependencies or dataset were downloaded.


## Real pinned-mirror preparation (Hugging Face extension)

A complete local download of `py97/UTKFace-Cropped` at revision
`e80f21fb21fa3631380c3cb634b79be3ac2f4cba` passed the configured archive size/SHA-256
checks. Preparation in `dataset/protocol_hf_v1` used the existing policy: seed 42,
age <85, full decode for in-range candidates, and exclusion of every member of
exact-byte duplicate groups. Counts were 23,705 valid-name images at download,
436 age exclusions, 750 duplicate exclusions in 368 groups, and 22,519 accepted
samples (18,015 / 2,252 / 2,252). These outputs stay local under ignored `dataset/`.
The common loader reverified all accepted image hashes and CSV/partition contracts.

This newly available image evidence confirms **174 exact-byte duplicate groups
span the committed historical CSV splits**: train/test 90, train/validation 66,
validation/test 18. All historical CSV image paths exist in this mirror, and the
CSV bytes remain unchanged. Their label/filename disjointness alone did not rule
out image-content leakage. Use a fresh deduplicated protocol for new experiments;
near-duplicate, identity, annotation accuracy and model performance are still
unverified. The earlier P1.5 note about unverified real-data preparation is a
historical record, superseded by this local preparation evidence only.
