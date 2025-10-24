"""
Data preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from loguru import logger


class DataPreprocessor:
    """
    Preprocess rheology data.

    Handles missing values, outliers, and normalization.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self.imputer = None
        self.feature_names = None

        logger.info("DataPreprocessor initialized")

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_missing: bool = True,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Clean data by removing duplicates, handling missing values, and outliers.

        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)

        logger.info(f"Starting data cleaning. Initial rows: {initial_rows}")

        # Remove duplicates
        if drop_duplicates:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")

        # Handle missing values
        if handle_missing:
            df_clean = self.handle_missing_values(df_clean)

        # Remove outliers
        if remove_outliers:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_clean = self.remove_outliers(df_clean, col)

        final_rows = len(df_clean)
        logger.info(f"Data cleaning completed. Final rows: {final_rows} "
                   f"({initial_rows - final_rows} rows removed)")

        return df_clean

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'drop')
            columns: Columns to process (None for all numeric columns)

        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()

        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        missing_count = df_processed[columns].isnull().sum().sum()

        if missing_count == 0:
            logger.info("No missing values found")
            return df_processed

        logger.info(f"Handling {missing_count} missing values using strategy: {strategy}")

        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=columns)
        else:
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy=strategy)
                df_processed[columns] = self.imputer.fit_transform(df_processed[columns])
            else:
                df_processed[columns] = self.imputer.transform(df_processed[columns])

        logger.info("Missing values handled successfully")
        return df_processed

    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from a specific column.

        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            method: Method to use ('iqr', 'zscore', 'percentile')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        initial_count = len(df_clean)

        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[column]))
            mask = z_scores < threshold

        elif method == 'percentile':
            lower_percentile = df_clean[column].quantile(threshold / 100)
            upper_percentile = df_clean[column].quantile(1 - threshold / 100)
            mask = (df_clean[column] >= lower_percentile) & (df_clean[column] <= upper_percentile)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        df_clean = df_clean[mask]
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers from column '{column}' "
                       f"using {method} method")

        return df_clean

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize numerical columns.

        Args:
            df: Input DataFrame
            columns: Columns to normalize (None for all numeric columns)
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            DataFrame with normalized columns
        """
        df_normalized = df.copy()

        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = columns

        logger.info(f"Normalizing {len(columns)} columns using {method} method")

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        df_normalized[columns] = self.scaler.fit_transform(df_normalized[columns])

        logger.info("Normalization completed")
        return df_normalized

    def inverse_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Inverse transform normalized data.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("No scaler fitted. Call normalize_data() first.")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            result[self.feature_names] = self.scaler.inverse_transform(result[self.feature_names])
            return result
        else:
            return self.scaler.inverse_transform(data)

    def get_preprocessing_info(self) -> dict:
        """
        Get information about preprocessing steps applied.

        Returns:
            Dictionary with preprocessing information
        """
        info = {
            'scaler': type(self.scaler).__name__ if self.scaler else None,
            'imputer': type(self.imputer).__name__ if self.imputer else None,
            'feature_names': self.feature_names,
        }

        if self.scaler:
            if hasattr(self.scaler, 'mean_'):
                info['scaler_mean'] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, 'scale_'):
                info['scaler_scale'] = self.scaler.scale_.tolist()

        return info


class DataSplitter:
    """Split data into train, validation, and test sets."""

    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = False
    ) -> tuple:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
            random_state: Random seed
            stratify: Whether to stratify split (for classification)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        logger.info(f"Splitting data: test={test_size}, val={val_size}")

        # First split: separate test set
        stratify_param = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        # Second split: separate validation set
        val_size_adjusted = val_size / (1 - test_size)
        stratify_param = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )

        logger.info(f"Split completed: train={len(X_train)}, "
                   f"val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    from .data_loader import create_sample_data

    # Create sample data
    df = create_sample_data(1000)

    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_normalized = preprocessor.normalize_data(df_clean)

    print("Preprocessing completed")
    print(df_normalized.head())
