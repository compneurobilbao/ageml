import pytest
import numpy as np
import ageml.processing as processing


def test_find_correlations():
    # Test a very simple correlation
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Y = np.array([1, 2, 3])
    result = processing.find_correlations(X, Y)[0]
    expected = np.array([1, 1, -1])
    
    assert np.allclose(result, expected, rtol=1e-10) == True
