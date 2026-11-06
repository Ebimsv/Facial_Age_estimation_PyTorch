# Download UTKFace without Kaggle

## Default: a pinned public Hugging Face mirror

```bash
# No Hugging Face/Kaggle SDK, account, token or dependency installation required.
python scripts/download_data.py
# Explicit equivalent:
python scripts/download_data.py --source huggingface --dest dataset

# Inspect settings and the resolved URL without network or output writes:
python scripts/download_data.py --print-config
# Read-only local image sanity check (no provider import/auth/network):
python scripts/download_data.py --check-only
```

Source settings live in [`configs/download.yaml`](../configs/download.yaml), with
an identical packaged copy at `src/configs/download.yaml`. This is a **separate
schema** from `configs/default.yaml`: downloader settings never change saved
training configs or checkpoint contracts. The configuration pins:

| Field | Value |
|---|---|
| Repository | `py97/UTKFace-Cropped` |
| Revision | `e80f21fb21fa3631380c3cb634b79be3ac2f4cba` |
| File | `UTKFace.tar.gz` |
| Size | 106,634,631 bytes |
| SHA-256 | `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7` |
| Selected image directory | `UTKFace` |

The public URL is derived from repo/revision/filename rather than duplicated in
code. [Pinned upstream commit and file metadata](https://huggingface.co/datasets/py97/UTKFace-Cropped/commit/e80f21fb21fa3631380c3cb634b79be3ac2f4cba)
supply the size and digest. Changing source content requires intentionally updating
revision, size and digest together. Mutable `main` revisions are rejected.

This is an unofficial mirror. Its card's metadata does not establish new rights
in the original images; consult the [original UTKFace page](https://susanqq.github.io/UTKFace/),
which describes noncommercial research use. Archive hashing verifies bytes against
the pinned declaration; it does not independently authenticate labels, identities
or the mirror's provenance.

## Configuration and override precedence

```bash
python scripts/download_data.py --config /path/to/download.yaml --print-config
python scripts/download_data.py --config /path/to/download.yaml --dest /path/to/data
```

Packaged defaults, checkout/custom YAML, then explicit CLI values are merged and
validated. Partial YAML is supported; typos, bad source, mutable revision, malformed
digest, unsafe subdirectory and invalid sizes/timeouts/retry counts fail before
network/output I/O. Both committed YAML copies must stay identical; tests enforce
this and verify the installed wheel contains the download resource.

`--dest` is the **parent**; images go to `<dest>/utkcropped`. An explicit relative
CLI destination resolves against the working directory, preserving the earlier
CLI behavior. Without `--dest`, the configured relative path uses the checkout
root in source execution or the working directory in installed mode. Existing
valid images are reused without a download; their origin/revision is not inferred
from the currently selected config. Existing invalid targets are preserved.

`timeout_seconds` defaults to 30 per blocking socket operation, not a whole-transfer
deadline. `max_retries=2` means at most three attempts for network interruption,
HTTP 429 or 5xx. A retry restarts the partial file inside this run's staging area;
there is no cross-run resume/cache. HTTP 403/404, size and digest mismatches fail
without retry. No failure automatically switches providers or authenticates to Kaggle.

## Verification, extraction and cleanup

1. A unique `.utkface-download-*` directory is created inside the destination parent.
   Existing archives/sibling files are never selected or removed.
2. Hugging Face is streamed with Python's standard library. Only a complete HTTP
   200 response is accepted; declared size, actual byte count and SHA-256 must all
   match before extraction. Oversized data is stopped at the pinned byte limit.
3. TAR metadata is checked before image copying: unsafe/traversal paths, duplicate
   names, symlinks, hardlinks, devices, FIFO/sparse files and excessive sizes/counts
   are rejected. No `extractall` or Python-version-dependent extraction filter is
   used. Only regular valid-name images in the selected directory are copied;
   archive ownership/mode information is never applied.
4. TAR members keep archive order during extraction. Sorting names would repeatedly
   rewind/decompress a gzip stream and can make a large dataset impractically slow.
5. The image folder must contain at least 18,000 valid UTK names, and up to the first
   five sorted images must fully decode. This sanity check is separate from the
   full decode/content/duplicate checks in preparation.
6. The final target is reserved by exclusive creation. File copies are exclusive;
   an existing/concurrently created target is never overwritten. Failure removes
   only files created by that publication; another process's unexpected files stay.
   This is not atomic directory publication: wait for success before preparing data.
7. This run's temporary archive/extraction are removed. The final images remain
   locally for future preparation/training; data is ignored by Git and excluded
   from distributions. A failed run can leave an empty destination parent.

Both archive types retain limits of 200,000 members, 8 GiB total declared
uncompressed size and 32 MiB per candidate image. These are acceptance limits,
not a complete defense against all malicious archives. `--check-only` does not
establish dataset identity, full decode success or leakage absence.

## Optional alternative: Kaggle

```bash
# Only for an explicit Kaggle download:
python -m pip install ".[data]"
python scripts/download_data.py --source kaggle --dest dataset
# VERSION must be a verified positive Kaggle version number:
python scripts/download_data.py --source kaggle --dataset-version VERSION
```

Kaggle remains available via the optional `data` extra. The script never installs
packages. Authentication is delegated to its SDK, following the
[official guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
No project code parses credential contents; SDK import/auth/download stdout/stderr
and exception detail are suppressed. The Kaggle path was tested with fake APIs;
its live archive layout/version availability remain unverified.

Only `utkface-new.zip` inside this run's staging is accepted. ZIP paths/types,
traversal, duplicate members, encryption and size limits are checked. Root or a
unique nested `UTKFace`/`utkcropped` folder is discovered; ambiguous/nonstandard
layouts require `--image-subdir`. Without a Kaggle version, the source is latest,
not pinned. Hugging Face is now the default: older commands expecting Kaggle must
explicitly pass `--source kaggle`, or select it in download YAML.

`kaggle*.json`, `access_token` and `.kaggle/` remain ignored/excluded. This protects
untracked files and distributions, not previously committed history. No credential
was read, moved or rotated during this work; configure third-party debug logging
responsibly. Current SDK releases can require newer Python than the core project;
see [Kaggle on PyPI](https://pypi.org/project/kaggle/).

## Prepare separately, then continue the roadmap

A fresh Hugging Face download prints the pinned source/revision for preparation:

```bash
python scripts/prepare_data.py --dataset-dir dataset/utkcropped \
  --csv-dir csv_dataset/protocol_hf_v1 --check-decode \
  --source huggingface:py97/UTKFace-Cropped \
  --dataset-version e80f21fb21fa3631380c3cb634b79be3ac2f4cba
```

Existing CSVs are never overwritten; use another fresh output directory on repeats.
Preparation checks feasible strata and writes hashes/rejection reports as described
in [DATA_PROTOCOL.md](DATA_PROTOCOL.md). A cached folder without verified provenance
must not automatically be labeled as the configured mirror revision.

Training will use `data.root_dir` and `data.csv_dir`; source/version options belong
to download/preparation, not the model. Atomic checkpoint and completed-epoch resume are implemented in P2.1;
see [TRAINING.md](TRAINING.md) before starting or continuing a run. This download addition neither
trains a model nor creates an accuracy benchmark.

Basic CI uses synthetic HTTP/TAR/ZIP fixtures only, including partial transfers,
checksum mismatch, retries, unsafe members, cleanup and installed configuration.
Real-download verification, when performed locally, is recorded separately in
[the roadmap](../audit/IMPLEMENTATION_ROADMAP.md).


## Verified local run (2026-11-06)

The default pinned file was downloaded in full with this implementation. Its
106,634,631 bytes and SHA-256 matched before TAR extraction. **23,705 valid-name
images** were published to `dataset/utkcropped`; the temporary archive was removed.
All image paths referenced by the four committed historical CSVs were present.

A real new protocol was then prepared in the ignored local directory
`dataset/protocol_hf_v1`, using full decode and byte hashing inside the default age
filter (<85), seed 42 and the pinned source/revision. There were 436 age-filter
rejections and 750 rejected copies across 368 exact-byte duplicate groups, leaving
22,519 samples: **18,015 train / 2,252 validation / 2,252 test**. The common loader
reverified CSV/image hashes and master/partition consistency before any model was
constructed. CSVs/images from this local run are not committed or distributed.

For that existing local protocol, future training selects
`data.csv_dir=dataset/protocol_hf_v1`; the standard `data.root_dir=dataset/utkcropped`
already points to the images. No training was started. The new local manifest
SHA-256 is `9d2d6f8dfd7b6127124ba88b2da3f5dc5b4183b1c2389ee8887cd25a01d3ce46`;
its environment/Git fields make it a record of this run, not a universal digest
for all future preparations.

Byte duplicates also appeared across historical splits: 174 groups total (90
train/test, 66 train/validation, 18 validation/test). The historical files were
preserved, but should not supply a new independent benchmark. The new protocol's
accepted hashes are unique; near-duplicate and identity leakage remain unknown.
No new accuracy or speed result is claimed. The complete offline suite passed
**319 tests**, including 50 Hugging Face cases and installed-resource checks.
