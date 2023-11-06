import pytest
import numpy as np
import ageml.processing as processing


def test_find_correlations():
    # Test a very simple correlation
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Y = np.array([1, 2, 3])
    result = processing.find_correlations(X, Y)[0]
    expected = np.array([1, 1, -1])
    
    assert np.allclose(result, expected, rtol=1e-10) is True


@pytest.mark.parametrize('X, Y, exception_msg', [
    (np.array([[2, 4, np.nan], [4, 8, -12], [6, 12, -18], [8, 16, -24]]), np.array([1, 2, 3, 4]), "NaN entrie(s) found in X."),
    (np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18], [8, 16, -24]]), np.array([1, 2, 3, np.nan]), "NaN entrie(s) found in Y.")
])
def test_find_correlations_nans(X, Y, exception_msg):
    with pytest.raises(ValueError) as exc_info:
        processing.find_correlations(X, Y)
        assert exc_info.type == ValueError
    assert str(exc_info.value) == exception_msg
