import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    expected: baseline (train set column)
    actual: comparison (test/val set column)
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.histogram(
        np.percentile(expected, breakpoints), bins=buckets
    )[0] / len(expected)
    actual_percents = np.histogram(
        np.percentile(actual, breakpoints), bins=buckets
    )[0] / len(actual)
    
    psi_values = (expected_percents - actual_percents) * np.log(
        (expected_percents + 1e-6) / (actual_percents + 1e-6)
    )
    return np.sum(psi_values)

# Example usage inside evaluation
def evaluate_distribution_shift(X_train, X_test, feature="Close"):
    train_col = X_train[feature].dropna().values
    test_col = X_test[feature].dropna().values
    psi = calculate_psi(train_col, test_col)
    print(f"PSI for {feature}: {psi:.4f}")
    return psi