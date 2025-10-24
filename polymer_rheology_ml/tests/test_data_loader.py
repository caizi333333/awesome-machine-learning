"""
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
