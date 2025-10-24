#!/usr/bin/env python
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
    print(f"\nTest Metrics:\n{metrics}")

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
