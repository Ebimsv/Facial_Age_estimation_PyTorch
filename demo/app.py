#!/usr/bin/env python3
"""Gradio web demo for facial age estimation.

Launch
------
  python demo/app.py --checkpoint checkpoints/best_*.pt

  # Share publicly via Gradio tunnel:
  python demo/app.py --checkpoint checkpoints/best_*.pt --share

The demo:
  1. Accepts a cropped face photo.
  2. Runs the age estimator.
  3. Displays the predicted age AND a bar-chart probability distribution
     over the checkpoint's age support. The distribution is not calibrated confidence.
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
from PIL import Image

from src.inference import AgePredictor
from src.utils.checkpoint import add_checkpoint_arguments

matplotlib.use("Agg")  # non-interactive backend — needed for Gradio

_predictor: AgePredictor | None = None


def load_model(checkpoint_path: str, config_path: str | None = None, **kwargs) -> None:
    """Load the same artifact-backed pipeline used by CLI inference."""
    global _predictor
    _predictor = AgePredictor.from_checkpoint(checkpoint_path, config_path, **kwargs)


def predict(image: Image.Image) -> tuple[str, plt.Figure]:
    if _predictor is None or image is None:
        return "Upload a cropped face image first.", plt.figure()
    prediction = _predictor.predict(image)
    result = f"🎂 Estimated Age: **{prediction.age:.1f} years**"
    fig, ax = plt.subplots(figsize=(10, 3))
    if prediction.probabilities is not None:
        ages = prediction.ages.numpy()
        ax.bar(ages, prediction.probabilities.numpy() * 100, color="#4C72B0")
        ax.axvline(prediction.age, color="#DD4444", linestyle="--", label=f"Predicted: {prediction.age:.1f}y")
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability (%)")
        ax.set_xlim(float(ages[0]) - 1, float(ages[-1]) + 1)
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"Age: {prediction.age:.1f}", ha="center", va="center", fontsize=20)
        ax.axis("off")
    fig.tight_layout()
    return result, fig


# ── Gradio Interface ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Facial Age Estimator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🎂 Facial Age Estimator
            Upload a face photo to estimate the person's age using a deep learning model
            trained on the UTKFace dataset.

            **Model:** {_predictor.checkpoint.cfg.model.name if _predictor else 'not loaded'}
            | The distribution is not calibrated confidence.
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
    add_checkpoint_arguments(p)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.checkpoint, args.config, device=args.device,
               legacy_resnet50=args.legacy_resnet50, trust_legacy_pickle=args.trust_legacy_pickle)
    demo = build_interface()
    demo.launch(share=args.share, server_port=args.port)
