"""
Model evaluation module.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score
from loguru import logger


class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def evaluate(y_true, y_pred) -> dict:
        """
        Calculate evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    @staticmethod
    def cross_validate(model, X, y, cv=5) -> dict:
        """
        Perform cross-validation.

        Args:
            model: ML model
            X: Features
            y: Targets
            cv: Number of folds

        Returns:
            Dictionary with CV scores
        """
        scores = {
            'mse': -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'),
            'mae': -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error'),
            'r2': cross_val_score(model, X, y, cv=cv, scoring='r2')
        }

        results = {
            'mse_mean': scores['mse'].mean(),
            'mse_std': scores['mse'].std(),
            'mae_mean': scores['mae'].mean(),
            'mae_std': scores['mae'].std(),
            'r2_mean': scores['r2'].mean(),
            'r2_std': scores['r2'].std(),
        }

        logger.info(f"Cross-validation results: {results}")
        return results
