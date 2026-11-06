"""CLI, evaluation, export and predictor share the same artifact contract."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts import evaluate, export_model, inference
from src.configs import load_config
from src.inference import AgePredictor
from src.models import build_model
from src.utils.checkpoint import create_checkpoint, load_checkpoint


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    cfg = load_config(overrides=["model.name=mobilenet_v3", "model.pretrained=false", "data.img_size=64",
                                 "training.batch_size=2",
                                 "loss.start_age=10", "loss.end_age=20", "model.output_dim=11",
                                 "data.max_age=21", "data.mean=[0.1,0.2,0.3]"])
    path = root / "model.pt"
    model = build_model(cfg).eval()
    torch.save(create_checkpoint(model, cfg), path)
    image = Image.new("RGB", (92, 81), (40, 80, 130))
    image_path = root / "image.png"
    image.save(image_path)
    return path, image, image_path


def test_cli_and_evaluation_match_shared_predictor(sample):
    path, image, image_path = sample
    predictor = AgePredictor.from_checkpoint(path, device="cpu")
    result = predictor.predict(image)
    model, loss_fn, transform, device = inference.load_model_and_pipeline(path, device="cpu")
    assert inference.predict_age(model, loss_fn, transform, device, image_path) == result.age
    loader = DataLoader([(transform(image), torch.tensor([15.]), "Male", "White")], batch_size=1)
    ages, _, _, _ = evaluate.run_evaluation(model, loader, loss_fn, device, use_amp=False)
    assert ages == pytest.approx([result.age])
    assert result.ages.tolist() == list(range(10, 21))
    assert result.probabilities.sum().item() == pytest.approx(1)


def test_demo_matches_shared_predictor(sample):
    pytest.importorskip("gradio")
    from demo import app
    path, image, _ = sample
    app.load_model(path, device="cpu")
    expected = AgePredictor.from_checkpoint(path, device="cpu").predict(image)
    text, figure = app.predict(image)
    assert f"{expected.age:.1f}" in text
    assert len(figure.axes[0].patches) == 11
    assert figure.axes[0].get_xlim() == (9, 21)
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_evaluation_cli_uses_artifact_and_runtime_data_paths(sample, tmp_path, monkeypatch):
    path, image, _ = sample
    images, csvs = tmp_path / "images", tmp_path / "csvs"
    images.mkdir()
    csvs.mkdir()
    for split in ["train", "valid", "test"]:
        names = [f"{split}_a.png", f"{split}_b.png"]
        for name in names:
            image.save(images / name)
        pd.DataFrame({"image_name": names, "age": [15, 15], "gender": "Male", "ethnicity": "White"}).to_csv(
            csvs / f"{split}_set.csv", index=False)
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr("sys.argv", ["eval-age", "--checkpoint", str(path), "--device", "cpu",
                                     "--root-dir", str(images), "--csv-dir", str(csvs),
                                     "--num-workers", "0", "--save-csv", str(output)])
    evaluate.main()
    expected = AgePredictor.from_checkpoint(path, device="cpu").predict(image).age
    assert pd.read_csv(output).predicted_age.tolist() == pytest.approx([expected, expected])


def test_torchscript_uses_saved_size_and_logits_contract(sample, tmp_path, monkeypatch):
    path, image, _ = sample
    output = tmp_path / "nested" / "model.jit"
    monkeypatch.setattr("sys.argv", ["export-age", "--checkpoint", str(path), "--output", str(output)])
    export_model.main()
    bundle = load_checkpoint(path, device="cpu")
    with torch.no_grad():
        tensor = bundle.transform(image).unsqueeze(0)
        torch.testing.assert_close(torch.jit.load(str(output))(tensor), bundle.model(tensor))
    metadata = json.loads((tmp_path / "nested/model.jit.json").read_text())
    assert metadata["preprocessing"]["img_size"] == 64 and metadata["output"] == "logits"
    assert metadata["predictor"]["start_age"] == 10


def test_export_cannot_override_saved_resolution(sample, monkeypatch):
    monkeypatch.setattr("sys.argv", ["export-age", "--checkpoint", str(sample[0]), "--img-size", "224"])
    with pytest.raises(ValueError, match="preprocessing contract"):
        export_model.main()


def test_raw_regression_export_preserves_bounded_age_policy(tmp_path, monkeypatch):
    cfg = load_config(overrides=["model.name=mobilenet_v3", "model.pretrained=false",
                                 "model.output_dim=1", "loss.type=l1", "data.img_size=64"])
    model = build_model(cfg).eval()
    final = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(-7)
    path, output = tmp_path / "regression.pt", tmp_path / "regression.jit"
    torch.save(create_checkpoint(model, cfg), path)
    monkeypatch.setattr("sys.argv", ["export-age", "--checkpoint", str(path), "--output", str(output)])
    export_model.main()
    bundle = load_checkpoint(path, device="cpu")
    with torch.no_grad():
        raw = torch.jit.load(str(output))(torch.zeros(1, 3, 64, 64))
    assert raw.item() == -7
    assert bundle.loss_fn.predict_age(raw).item() == 0
    metadata = json.loads((tmp_path / "regression.jit.json").read_text())
    assert metadata["predictor"]["postprocess"] == "clamp_0_100"
