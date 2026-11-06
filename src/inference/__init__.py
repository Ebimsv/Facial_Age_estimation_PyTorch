"""Shared checkpoint-backed image inference."""

from .predictor import AgePredictor, predict_image

__all__ = ["AgePredictor", "predict_image"]
