"""
inference/__init__.py

Public API for the inference package.
"""

from inference.predict import predict_fullscene, prob_to_binary

__all__ = ["predict_fullscene", "prob_to_binary"]
