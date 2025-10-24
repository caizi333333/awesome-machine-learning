"""
Data loading utilities for rheology data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from loguru import logger


class RheologyDataLoader:
    """
    Load and manage rheology experimental data.

    Attributes:
        data_path: Path to the data directory or file
        data: Loaded DataFrame
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader.

        Args:
            data_path: Path to data file or directory
        """
        self.data_path = Path(data_path)
        self.data = None
        logger.info(f"RheologyDataLoader initialized with path: {self.data_path}")

    def load_data(
        self,
        file_format: str = 'csv',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            file_format: File format ('csv', 'excel', 'json', 'parquet')
            **kwargs: Additional arguments passed to pandas read function

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")

        if file_format == 'csv':
            self.data = pd.read_csv(self.data_path, **kwargs)
        elif file_format == 'excel':
            self.data = pd.read_excel(self.data_path, **kwargs)
        elif file_format == 'json':
            self.data = pd.read_json(self.data_path, **kwargs)
        elif file_format == 'parquet':
            self.data = pd.read_parquet(self.data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        self._log_data_info()

        return self.data

    def load_multiple_files(
        self,
        file_patterns: List[str],
        concat_axis: int = 0
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple data files.

        Args:
            file_patterns: List of file patterns to match
            concat_axis: Axis for concatenation (0=rows, 1=columns)

        Returns:
            Concatenated DataFrame
        """
        dfs = []

        for pattern in file_patterns:
            files = list(self.data_path.parent.glob(pattern))
            logger.info(f"Found {len(files)} files matching pattern: {pattern}")

            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)

        self.data = pd.concat(dfs, axis=concat_axis, ignore_index=True)
        logger.info(f"Concatenated {len(dfs)} files. Final shape: {self.data.shape}")

        return self.data

    def _log_data_info(self):
        """Log basic information about the loaded data."""
        logger.info("Data Info:")
        logger.info(f"  Columns: {self.data.columns.tolist()}")
        logger.info(f"  Data types:\n{self.data.dtypes}")
        logger.info(f"  Missing values:\n{self.data.isnull().sum()}")
        logger.info(f"  Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics of the data.

        Returns:
            DataFrame with summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        summary = self.data.describe()
        logger.info("Summary statistics computed")
        return summary

    def check_data_quality(self) -> dict:
        """
        Check data quality and return report.

        Returns:
            Dictionary with data quality metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        quality_report = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist(),
        }

        logger.info("Data quality check completed")
        return quality_report

    def save_data(
        self,
        output_path: Union[str, Path],
        file_format: str = 'csv',
        **kwargs
    ):
        """
        Save data to file.

        Args:
            output_path: Output file path
            file_format: File format ('csv', 'excel', 'parquet')
            **kwargs: Additional arguments passed to pandas to_* function
        """
        if self.data is None:
            raise ValueError("No data to save. Load or process data first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == 'csv':
            self.data.to_csv(output_path, index=False, **kwargs)
        elif file_format == 'excel':
            self.data.to_excel(output_path, index=False, **kwargs)
        elif file_format == 'parquet':
            self.data.to_parquet(output_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Data saved to {output_path}")


def create_sample_data(
    n_samples: int = 1000,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create sample rheology data for testing.

    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the sample data (optional)

    Returns:
        Sample DataFrame
    """
    np.random.seed(42)

    # Generate synthetic data
    data = {
        'molecular_weight': np.random.uniform(10000, 500000, n_samples),
        'pdi': np.random.uniform(1.0, 3.0, n_samples),
        'temperature': np.random.uniform(150, 250, n_samples),
        'shear_rate': np.random.uniform(0.1, 1000, n_samples),
        'concentration': np.random.uniform(1, 50, n_samples),
    }

    # Calculate synthetic viscosity with some realistic relationships
    data['viscosity'] = (
        (data['molecular_weight'] / 10000) ** 1.5 *
        np.exp(-0.01 * (data['temperature'] - 200)) *
        (data['shear_rate'] + 1) ** -0.5 *
        (1 + 0.1 * data['concentration']) *
        np.random.normal(1.0, 0.1, n_samples)  # Add noise
    )

    df = pd.DataFrame(data)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")

    logger.info(f"Generated {n_samples} sample data points")
    return df


if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data(1000, 'data/raw/sample_data.csv')
    print(sample_data.head())
