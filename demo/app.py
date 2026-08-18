#!/usr/bin/env python3
"""Gradio web demo for facial age estimation.

Launch
------
  python demo/app.py --checkpoint checkpoints/best_*.pt

  # Share publicly via Gradio tunnel:
  python demo/app.py --checkpoint checkpoints/best_*.pt --share

The demo:
  1. Accepts a face photo (upload or webcam).
  2. Runs the age estimator.
  3. Displays the predicted age AND a bar-chart probability distribution
     over all 101 age classes — so users can see model confidence visually.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

from src.models import AgeEstimationModel
from src.training import get_loss_fn

matplotlib.use("Agg")  # non-interactive backend — needed for Gradio

# ── Global state (loaded once at startup) ────────────────────────────────────
_model: AgeEstimationModel | None = None
_loss_fn = None
_transform = None
_device: torch.device | None = None


def load_model(checkpoint_path: str, config_path: str) -> None:
    """Load model and loss function into global state."""
    global _model, _loss_fn, _transform, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(config_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", cfg.model.name)
    output_dim = ckpt.get("output_dim", cfg.model.output_dim)

    _model = AgeEstimationModel(
        model_name=model_name,
        pretrained=False,
        output_dim=output_dim,
        dropout=0.0,
    ).to(_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    _loss_fn = get_loss_fn(cfg).to(_device)

    img_size = cfg.data.img_size
    mean = list(cfg.data.mean)
    std = list(cfg.data.std)
    _transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    print(
        f"✓ Model loaded: {model_name} "
        f"(output_dim={output_dim}, device={_device})"
    )


@torch.no_grad()
def predict(image: Image.Image) -> tuple[str, plt.Figure]:
    """Run inference and return (age_text, probability_plot).

    Args:
        image: PIL image from the Gradio UI.

    Returns:
        Tuple of (formatted string with age, matplotlib Figure with distribution).
    """
    if _model is None:
        return "⚠️ Model not loaded.", plt.figure()

    # Preprocess
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_device)

    # Forward
    logits = _model(tensor)
    pred_age = _loss_fn.predict_age(logits).item()
    pred_age_int = round(pred_age)

    result_text = f"🎂 Estimated Age: **{pred_age:.1f} years** (~{pred_age_int})"

    # ── Build probability distribution plot ──────────────────────────────
    if _model.output_dim == 101:
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        ages = np.arange(0, 101)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(ages, probs * 100, color="#4C72B0", alpha=0.85, width=0.9)
        ax.axvline(x=pred_age, color="#DD4444", linewidth=2.5, linestyle="--",
                   label=f"Predicted: {pred_age:.1f}y")
        ax.set_xlabel("Age", fontsize=11)
        ax.set_ylabel("Probability (%)", fontsize=11)
        ax.set_title("Model Age Probability Distribution", fontsize=12, fontweight="bold")
        ax.set_xlim(-1, 101)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"Age: {pred_age:.1f}", ha="center", va="center", fontsize=20)
        ax.axis("off")

    return result_text, fig


# ── Gradio Interface ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Facial Age Estimator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎂 Facial Age Estimator
            Upload a face photo to estimate the person's age using a deep learning model
            trained on the UTKFace dataset.

            **Model:** EfficientNet-B3 with Mean-Variance Loss | **Dataset:** UTKFace (~20k images)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload a face photo",
                    height=300,
                )
                submit_btn = gr.Button("Estimate Age 🚀", variant="primary")

            with gr.Column(scale=2):
                age_output = gr.Markdown(label="Prediction")
                dist_output = gr.Plot(label="Age Probability Distribution")

        # Example images (placed in demo/examples/)
        examples_dir = Path(__file__).parent / "examples"
        if examples_dir.exists():
            example_files = sorted(examples_dir.glob("*.jpg"))[:4]
            if example_files:
                gr.Examples(
                    examples=[[str(f)] for f in example_files],
                    inputs=image_input,
                )

        submit_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[age_output, dist_output],
        )
        image_input.change(
            fn=predict,
            inputs=image_input,
            outputs=[age_output, dist_output],
        )

        gr.Markdown(
            """
            ---
            **Note:** This model was trained on UTKFace data. 
            Accuracy may vary due to lighting, pose, and makeup.
            """
        )

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the Gradio age estimation demo")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.checkpoint, args.config)
    demo = build_interface()
    demo.launch(share=args.share, server_port=args.port)
