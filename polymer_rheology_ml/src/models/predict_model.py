"""
Model prediction module.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Union
from loguru import logger


class Predictor:
    """Make predictions using trained models."""

    def __init__(self, model_path: str = None, model=None):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model file
            model: Pre-loaded model object
        """
        if model_path:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        predictions = self.model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")
        return predictions

    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Make predictions with confidence intervals (for tree-based models).

        Args:
            X: Features

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if hasattr(self.model, 'estimators_'):
            # For ensemble models
            predictions_all = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            mean_pred = predictions_all.mean(axis=0)
            std_pred = predictions_all.std(axis=0)

            # 95% confidence interval
            lower = mean_pred - 1.96 * std_pred
            upper = mean_pred + 1.96 * std_pred

            return mean_pred, lower, upper
        else:
            predictions = self.model.predict(X)
            return predictions, predictions, predictions
