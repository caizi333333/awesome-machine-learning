"""
Feature engineering for rheology data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from loguru import logger


class FeatureBuilder:
    """
    Build features for rheology prediction models.

    Creates polynomial features, interaction features, and domain-specific features.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize feature builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.created_features = []
        logger.info("FeatureBuilder initialized")

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features based on configuration.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        df_featured = df.copy()

        # Create domain-specific features
        if self.config.get('create_domain_features', True):
            df_featured = self.create_domain_features(df_featured)

        # Create polynomial features
        if self.config.get('create_polynomial', False):
            degree = self.config.get('polynomial_degree', 2)
            poly_columns = self.config.get('polynomial_columns', [])
            if poly_columns:
                df_featured = self.create_polynomial_features(
                    df_featured, poly_columns, degree
                )

        # Create interaction features
        if self.config.get('create_interactions', False):
            interaction_pairs = self.config.get('interaction_pairs', [])
            if interaction_pairs:
                df_featured = self.create_interaction_features(
                    df_featured, interaction_pairs
                )

        logger.info(f"Created {len(df_featured.columns) - len(df.columns)} new features")
        return df_featured

    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for rheology.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with domain-specific features
        """
        df_new = df.copy()
        logger.info("Creating domain-specific features")

        # Molecular weight features
        if 'molecular_weight' in df_new.columns:
            df_new['log_mw'] = np.log10(df_new['molecular_weight'])
            df_new['sqrt_mw'] = np.sqrt(df_new['molecular_weight'])
            self.created_features.extend(['log_mw', 'sqrt_mw'])

        # Temperature features
        if 'temperature' in df_new.columns:
            # Absolute temperature
            df_new['temp_kelvin'] = df_new['temperature'] + 273.15
            # Reciprocal temperature (for Arrhenius-type relations)
            df_new['temp_reciprocal'] = 1 / df_new['temp_kelvin']
            # Log temperature
            df_new['log_temp'] = np.log(df_new['temp_kelvin'])
            self.created_features.extend(['temp_kelvin', 'temp_reciprocal', 'log_temp'])

        # Shear rate features
        if 'shear_rate' in df_new.columns:
            # Log shear rate
            df_new['log_shear_rate'] = np.log10(df_new['shear_rate'] + 1e-10)
            # Square root
            df_new['sqrt_shear_rate'] = np.sqrt(df_new['shear_rate'])
            self.created_features.extend(['log_shear_rate', 'sqrt_shear_rate'])

        # PDI features
        if 'pdi' in df_new.columns:
            df_new['pdi_squared'] = df_new['pdi'] ** 2
            df_new['log_pdi'] = np.log(df_new['pdi'])
            self.created_features.extend(['pdi_squared', 'log_pdi'])

        # Combined features
        if 'molecular_weight' in df_new.columns and 'pdi' in df_new.columns:
            # Molecular weight distribution width
            df_new['mw_distribution_width'] = df_new['molecular_weight'] * (df_new['pdi'] - 1)
            self.created_features.append('mw_distribution_width')

        # WLF equation related features
        if 'temperature' in df_new.columns:
            # Assuming reference temperature is 200°C
            T_ref = 200
            df_new['T_minus_Tref'] = df_new['temperature'] - T_ref
            self.created_features.append('T_minus_Tref')

        # Arrhenius-type activation energy features
        if 'temperature' in df_new.columns and 'viscosity' in df_new.columns:
            R = 8.314  # Gas constant
            df_new['arrhenius_term'] = np.log(df_new['viscosity']) * df_new['temp_reciprocal'] * R
            self.created_features.append('arrhenius_term')

        logger.info(f"Created {len(self.created_features)} domain-specific features")
        return df_new

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree
            include_bias: Whether to include bias column

        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features (degree={degree}) for {len(columns)} columns")

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)

        # Create DataFrame with polynomial features
        df_poly = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=df.index
        )

        # Remove original columns (they're duplicated)
        new_columns = [col for col in df_poly.columns if col not in columns]
        df_poly = df_poly[new_columns]

        # Concatenate with original DataFrame
        df_result = pd.concat([df, df_poly], axis=1)

        logger.info(f"Created {len(new_columns)} polynomial features")
        return df_result

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features.

        Args:
            df: Input DataFrame
            feature_pairs: List of feature pairs to create interactions

        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        logger.info(f"Creating interaction features for {len(feature_pairs)} pairs")

        for feat1, feat2 in feature_pairs:
            if feat1 in df_new.columns and feat2 in df_new.columns:
                # Multiplication
                interaction_name = f'{feat1}_x_{feat2}'
                df_new[interaction_name] = df_new[feat1] * df_new[feat2]

                # Division
                division_name = f'{feat1}_div_{feat2}'
                df_new[division_name] = df_new[feat1] / (df_new[feat2] + 1e-10)

                # Addition
                addition_name = f'{feat1}_plus_{feat2}'
                df_new[addition_name] = df_new[feat1] + df_new[feat2]

                # Difference
                difference_name = f'{feat1}_minus_{feat2}'
                df_new[difference_name] = df_new[feat1] - df_new[feat2]

                self.created_features.extend([
                    interaction_name, division_name,
                    addition_name, difference_name
                ])
            else:
                logger.warning(f"Feature pair ({feat1}, {feat2}) not found in DataFrame")

        logger.info(f"Created {len(feature_pairs) * 4} interaction features")
        return df_new

    def get_created_features(self) -> List[str]:
        """
        Get list of created feature names.

        Returns:
            List of feature names
        """
        return self.created_features


class FeatureSelector:
    """
    Select important features for modeling.

    Supports various feature selection methods.
    """

    def __init__(self, method: str = 'importance'):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('importance', 'correlation', 'univariate', 'rfe')
        """
        self.method = method
        self.selected_features = None
        self.feature_scores = None
        logger.info(f"FeatureSelector initialized with method: {method}")

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Select features based on specified method.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            threshold: Threshold for selection (method-dependent)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using {self.method} method")

        if self.method == 'importance':
            selected = self._select_by_importance(X, y, threshold or 0.01)
        elif self.method == 'correlation':
            selected = self._select_by_correlation(X, y, threshold or 0.1)
        elif self.method == 'univariate':
            selected = self._select_by_univariate(X, y, n_features or 10)
        elif self.method == 'rfe':
            selected = self._select_by_rfe(X, y, n_features or 10)
        elif self.method == 'mutual_info':
            selected = self._select_by_mutual_info(X, y, n_features or 10)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features")
        return selected

    def _select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float
    ) -> List[str]:
        """Select features based on Random Forest importance."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_scores = importances

        selected = importances[importances['importance'] > threshold]['feature'].tolist()
        logger.info(f"Top 5 important features:\n{importances.head()}")

        return selected

    def _select_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float
    ) -> List[str]:
        """Select features based on correlation with target."""
        correlations = pd.DataFrame({
            'feature': X.columns,
            'correlation': [abs(X[col].corr(y)) for col in X.columns]
        }).sort_values('correlation', ascending=False)

        self.feature_scores = correlations

        selected = correlations[correlations['correlation'] > threshold]['feature'].tolist()
        logger.info(f"Top 5 correlated features:\n{correlations.head()}")

        return selected

    def _select_by_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int
    ) -> List[str]:
        """Select features using univariate statistical tests."""
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)

        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        self.feature_scores = scores

        selected_indices = selector.get_support(indices=True)
        selected = X.columns[selected_indices].tolist()

        logger.info(f"Top 5 features by F-score:\n{scores.head()}")
        return selected

    def _select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int
    ) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True)
        selected = X.columns[selected_indices].tolist()

        rankings = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')

        self.feature_scores = rankings
        logger.info(f"Feature rankings:\n{rankings.head(10)}")

        return selected

    def _select_by_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int
    ) -> List[str]:
        """Select features using mutual information."""
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)

        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        self.feature_scores = scores

        selected_indices = selector.get_support(indices=True)
        selected = X.columns[selected_indices].tolist()

        logger.info(f"Top 5 features by mutual information:\n{scores.head()}")
        return selected

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get feature scores from last selection.

        Returns:
            DataFrame with feature scores
        """
        if self.feature_scores is None:
            raise ValueError("No feature selection performed yet")
        return self.feature_scores


if __name__ == "__main__":
    # Example usage
    from ..data.data_loader import create_sample_data

    # Create sample data
    df = create_sample_data(1000)

    # Build features
    builder = FeatureBuilder({
        'create_domain_features': True,
        'create_polynomial': True,
        'polynomial_degree': 2,
        'polynomial_columns': ['temperature', 'shear_rate'],
        'create_interactions': True,
        'interaction_pairs': [('temperature', 'shear_rate')]
    })

    df_featured = builder.create_all_features(df)
    print(f"Original features: {len(df.columns)}")
    print(f"After feature engineering: {len(df_featured.columns)}")

    # Select features
    X = df_featured.drop('viscosity', axis=1)
    y = df_featured['viscosity']

    selector = FeatureSelector(method='importance')
    selected_features = selector.select_features(X, y, threshold=0.01)
    print(f"Selected {len(selected_features)} features")
