# Contributing to Facial Age Estimation PyTorch

Thank you for considering contributing! Whether it's a bug fix, a new backbone,
a better loss function, or improved documentation — all contributions are welcome.

## Getting Started

```bash
# 1. Fork & clone
git clone https://github.com/<your-handle>/Facial_Age_estimation_PyTorch.git
cd Facial_Age_estimation_PyTorch

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # or use conda

# 3. Install dev dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify everything works
pytest tests/ -v
```

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the code style below.

3. **Run linting & tests** before pushing:
   ```bash
   ruff check src/ scripts/ tests/
   pytest tests/ -v
   ```

4. **Push and open a Pull Request** against `main`.

## Code Style

- Formatter & linter: **Ruff** (`ruff check --fix .`)
- Line length: **100 characters**
- All public functions must have **docstrings** with `Args:` and `Returns:`.
- Use **type hints** throughout (`from __future__ import annotations`).
- Imports: `stdlib → third-party → local`, separated by blank lines.

## Adding a New Backbone

1. Look up the architecture's feature dimension (the `in_features` of the final linear layer).
2. Add it to `_BACKBONE_DIMS` in `src/models/age_estimator.py`.
3. Add an `elif` branch in `AgeEstimationModel.__init__` following the same pattern.
4. Add a new config file under `configs/model/` if the backbone needs specific settings.
5. Add the backbone to the `BACKBONES` list in `tests/test_model.py`.

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/issues/new)
and include:

- Your operating system and Python version
- GPU model and CUDA version (`nvidia-smi`, `python -c "import torch; print(torch.version.cuda)"`)
- The exact command you ran
- The full error traceback

## License

By contributing, you agree that your contributions will be licensed under the **MIT License**.
