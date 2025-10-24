"""
Model training module.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

from loguru import logger


class ModelTrainer:
    """Train machine learning models for rheology prediction."""

    def __init__(self, model_type: str = 'xgboost', **model_params):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model ('xgboost', 'random_forest', 'lightgbm', 'svr')
            **model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = self._initialize_model()
        self.is_trained = False

        logger.info(f"ModelTrainer initialized with model: {model_type}")

    def _initialize_model(self):
        """Initialize the model based on type."""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**self.model_params)
        elif self.model_type == 'svr':
            return SVR(**self.model_params)
        elif self.model_type == 'ridge':
            return Ridge(**self.model_params)
        elif self.model_type == 'lasso':
            return Lasso(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training {self.model_type} model...")

        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use early stopping for XGBoost and LightGBM
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True
        logger.info("Model training completed")

    def save_model(self, path: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from file."""
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
