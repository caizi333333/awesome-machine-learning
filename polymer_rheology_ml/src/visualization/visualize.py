"""
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
