#!/usr/bin/env python
"""
Script to generate remaining project files.
Run this to complete the project template setup.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# File contents
FILES = {
    # Models module
    "src/models/train_model.py": '''"""
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
''',

    "src/models/predict_model.py": '''"""
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
''',

    "src/models/evaluate_model.py": '''"""
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
''',

    # Visualization module
    "src/visualization/__init__.py": '''"""Visualization module."""

from .visualize import Visualizer

__all__ = ["Visualizer"]
''',

    "src/visualization/visualize.py": '''"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


class Visualizer:
    """Create visualizations for model results."""

    def __init__(self, style='seaborn-v0_8', figsize=(10, 6)):
        """Initialize visualizer."""
        plt.style.use(style)
        self.figsize = figsize

    def plot_predictions(self, y_true, y_pred, save_path=None):
        """Plot predictions vs true values."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title('Predictions vs True Values')

        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predictions')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """Plot feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            plt.figure(figsize=self.figsize)
            plt.title(f'Top {top_n} Feature Importances')
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
''',

    # Utils module
    "src/utils/__init__.py": '''"""Utility functions."""

from .helpers import load_config, set_seed

__all__ = ["load_config", "set_seed"]
''',

    "src/utils/helpers.py": '''"""
Helper utilities.
"""

import yaml
import random
import numpy as np
import torch
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
''',

    # Scripts
    "scripts/run_pipeline.py": '''#!/usr/bin/env python
"""
Main pipeline script.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import RheologyDataLoader
from src.data.preprocessing import DataPreprocessor, DataSplitter
from src.features.build_features import FeatureBuilder, FeatureSelector
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.visualization.visualize import Visualizer
from src.utils.helpers import load_config, set_seed
from loguru import logger


def main():
    """Run the complete ML pipeline."""
    # Load configuration
    config = load_config('config/config.yaml')
    set_seed(config.get('seed', 42))

    logger.info("Starting ML pipeline...")

    # 1. Load data
    logger.info("Step 1: Loading data")
    loader = RheologyDataLoader(config['data']['raw_data_path'])
    df = loader.load_data()

    # 2. Preprocess
    logger.info("Step 2: Preprocessing data")
    preprocessor = DataPreprocessor(config.get('preprocessing', {}))
    df_clean = preprocessor.clean_data(df)

    # 3. Feature engineering
    logger.info("Step 3: Feature engineering")
    builder = FeatureBuilder(config.get('feature_engineering', {}))
    df_featured = builder.create_all_features(df_clean)

    # 4. Prepare data
    target_col = config['data']['target_column']
    X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]

    # 5. Select features
    logger.info("Step 4: Feature selection")
    selector = FeatureSelector(method='importance')
    selected_features = selector.select_features(X, y, threshold=0.01)
    X = X[selected_features]

    # 6. Normalize
    X_normalized = preprocessor.normalize_data(X)

    # 7. Split data
    logger.info("Step 5: Splitting data")
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(
        X_normalized, y,
        test_size=config['data']['test_size'],
        val_size=config['data'].get('validation_size', 0.1)
    )

    # 8. Train model
    logger.info("Step 6: Training model")
    trainer = ModelTrainer(
        model_type=config['model']['type'],
        **config['model'].get(config['model']['type'], {})
    )
    trainer.train(X_train, y_train, X_val, y_val)

    # 9. Evaluate
    logger.info("Step 7: Evaluating model")
    y_pred = trainer.model.predict(X_test)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred)
    print(f"\\nTest Metrics:\\n{metrics}")

    # 10. Visualize
    logger.info("Step 8: Creating visualizations")
    viz = Visualizer()
    viz.plot_predictions(y_test, y_pred, 'reports/figures/predictions.png')
    viz.plot_feature_importance(
        trainer.model, selected_features,
        save_path='reports/figures/feature_importance.png'
    )

    # 11. Save model
    trainer.save_model('models/best_model.pkl')

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
''',

    # Tests
    "tests/__init__.py": "",

    "tests/test_data_loader.py": '''"""
Tests for data loader.
"""

import pytest
import pandas as pd
from src.data.data_loader import create_sample_data


def test_create_sample_data():
    """Test sample data creation."""
    df = create_sample_data(100)
    assert len(df) == 100
    assert 'viscosity' in df.columns
    assert 'molecular_weight' in df.columns
''',

    # Gitkeep files
    "models/.gitkeep": "",
    "reports/figures/.gitkeep": "",
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "data/external/.gitkeep": "",
}


def create_files():
    """Create all project files."""
    for filepath, content in FILES.items():
        full_path = BASE_DIR / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created: {filepath}")


if __name__ == "__main__":
    print("Generating project files...")
    create_files()
    print("\\nProject generation complete!")
    print("\\nNext steps:")
    print("1. cd polymer_rheology_ml")
    print("2. python generate_project.py")
    print("3. pip install -r requirements.txt")
    print("4. python src/data/data_loader.py  # Generate sample data")
    print("5. python scripts/run_pipeline.py  # Run pipeline")
