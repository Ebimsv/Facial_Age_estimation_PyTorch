# Dataset download guardrails (P1.6)

## Local checks and optional installation

```bash
# Local read-only inspection: no Kaggle import, authentication, download or install.
python scripts/download_data.py --check-only
python scripts/download_data.py --dest /path/to/dataset-parent --check-only

# Only if you choose to download through Kaggle:
python -m pip install ".[data]"
python scripts/download_data.py --dest dataset
```

`--dest` is the **parent** directory; normalized images go to `<dest>/utkcropped`.
The source-checkout default remains `<repository>/dataset`. The core package does
not require Kaggle. A missing or broken import produces a manual `.[data]` install
hint; the script never invokes pip. The optional extra declares `kaggle>=1.6.17`.
Existing Kaggle 2.2.4 source was inspected locally without importing/authenticating
it; a full SDK/version/network matrix is not claimed. Current Kaggle releases may
require a newer Python than the core project: see [Kaggle on PyPI](https://pypi.org/project/kaggle/).

Authentication belongs to the SDK. Follow the [official authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
Depending on the SDK version, this can use OAuth, `KAGGLE_API_TOKEN`, the token file
`~/.kaggle/access_token`, or legacy `~/.kaggle/kaggle.json`. Keep credential files
outside the checkout. The project script does not parse them, copy them or print
values; SDK stdout/stderr and exception detail are suppressed during import,
authentication and download. Configure third-party debug logging responsibly.

`kaggle*.json`, `access_token` and `.kaggle/` are ignored by Git, and credential
canaries are excluded from built distributions. Ignore rules protect untracked
files; they do not erase previously tracked content or establish that history is
clean. No credential was read or rotated during this implementation.

## Version and layout selection

The fixed dataset identifier is `jangedoo/utkface-new`. Without a version the SDK
requests latest; this is not a pinned reproducibility protocol. To request an
explicit known version, supply its positive integer:

```bash
# Example syntax; replace VERSION with a version you have verified exists.
python scripts/download_data.py --dest dataset --dataset-version VERSION

# If the archive contains multiple crops or has a nonstandard directory name:
python scripts/download_data.py --dest dataset --image-subdir UTKFace
```

The download method/expected `<slug>.zip` filename was checked against the
[official SDK source](https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py)
and installed 2.2.4 source. The live dataset file listing could not be retrieved
on 2026-11-06, and no real image archive was downloaded. **Its current layout and
version availability remain unverified.** Selection therefore inspects the actual
archive at runtime rather than assuming a fixed extracted path.

Automatic selection requires exactly one qualifying flat image directory: archive
root, or a nested directory named `UTKFace`/`utkcropped` (case insensitive). A
qualifying directory has at least 18,000 valid UTKFace filenames. Other names need
`--image-subdir`; ambiguity fails and reports up to eight qualifying paths. Use
`.` for archive root. Unknown layouts are rejected, not guessed or flattened across
different groups. The chosen image directory is normalized to `utkcropped`.

## Staging, validation and cleanup

1. A valid existing dataset is reused without importing Kaggle. An existing invalid
   target is preserved; choose another parent or inspect it manually.
2. A new run creates a unique `.utkface-download-*` directory inside the destination
   parent. Only `utkface-new.zip` **inside that staging directory** is accepted.
   Existing ZIPs in the parent and unrelated ZIPs in staging are never selected.
3. Before extraction, archive metadata is checked for unsafe paths, traversal,
   symlinks/special files, duplicate member paths, encryption and size bounds.
   Only selected valid-name image members are extracted. Other archive contents
   never become user dataset files.
4. The staged image directory must contain at least 18,000 regular files with valid
   UTK age/gender/ethnicity names. Up to the first five sorted images are fully
   decoded as a sanity check. This does not verify every image or its annotations.
5. The final target directory is reserved with exclusive creation. Images are
   copied with exclusive file creation; an existing target, including one created
   during download, is never overwritten. If publishing fails, only files created
   by that publication are removed; unexpected files from another process remain.
6. Only this run's temporary staging directory is cleaned, on success or failure.
   Existing archives, siblings and credentials are untouched. A newly created
   destination parent may remain empty after failure.

Publication uses file copies after staged validation; it is not an atomic directory
swap or a concurrent-reader transaction. Wait for successful completion before
preparing/training. Resource bounds are 200,000 archive members, 8 GiB total declared
uncompressed bytes, and 32 MiB per selected image. These are acceptance limits,
not measured dataset sizes or a complete defense against every malicious archive.

`--check-only` exits zero for a passing local sanity check and one for missing,
incomplete or corrupt sampled data. It neither creates staging/output nor imports
the SDK. It does not verify source authenticity, identity leakage, near-duplicates
or all decodes. A symlink image root is rejected. Successful download is not a
preparation/split protocol and does not regenerate CSVs.

## Next: prepare a separate protocol

```bash
python scripts/prepare_data.py --dataset-dir dataset/utkcropped \
  --csv-dir csv_dataset/protocol_v2 --check-decode \
  --source jangedoo/utkface-new --dataset-version unknown
```

Record an actual verified version instead of `unknown` when available. Preparation
performs the full decode option, byte-hash/duplicate checks and versioned split
manifest described in [DATA_PROTOCOL.md](DATA_PROTOCOL.md). A successful local
download check must not be presented as a new benchmark or proof of leakage absence.

Tests use fake APIs and tiny synthetic ZIPs, including network/auth failures,
missing SDK, malformed/unsafe archives, nested/ambiguous layouts, corrupt images,
publication/extraction failures and unrelated-file preservation. Basic CI does not
contact Kaggle, use credentials, install the optional extra or download real data.
