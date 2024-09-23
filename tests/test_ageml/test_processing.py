import pytest
import numpy as np
import ageml.processing as processing


def test_find_correlations():
    # Test a very simple correlation
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Y = np.array([1, 2, 3])
    corrs, order, p_values = processing.find_correlations(X, Y)
    corrs_expected = np.array([1, 1, -1])
    order_expected = np.array([2, 1, 0])
    p_values_expected = np.array([0.0, 0.0, 0.0])

    assert np.allclose(corrs, corrs_expected, rtol=1e-10) is True
    assert np.array_equal(order, order_expected) is True
    assert np.allclose(p_values, p_values_expected, atol=1e-7) is True


@pytest.mark.parametrize(
    "X, Y, exception_msg",
    [
        (
            np.array([[2, 4, np.nan], [4, 8, -12], [6, 12, -18], [8, 16, -24]]),
            np.array([1, 2, 3, 4]),
            "NaN entrie(s) found in X.",
        ),
        (
            np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18], [8, 16, -24]]),
            np.array([1, 2, 3, np.nan]),
            "NaN entrie(s) found in Y.",
        ),
    ],
)
def test_find_correlations_nans(X, Y, exception_msg):
    with pytest.raises(ValueError) as exc_info:
        processing.find_correlations(X, Y)
        assert exc_info.type == ValueError
    assert str(exc_info.value) == exception_msg


def test_covariate_correction():
    # Variables for testing
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Z = np.array([1, 2, 3]).reshape(-1, 1)

    # Check ValueError raies with NaNs
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, np.array([1, 2, np.nan]).reshape(-1, 1))
        assert exc_info.type == ValueError
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(np.array([2.0, np.nan]), Z)
        assert exc_info.type == ValueError
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, Z, beta=np.array([2.0, np.nan]).reshape(-1, 1))
        assert exc_info.type == ValueError

    # Check ValueError raises with incompatible shapes
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, np.array([1, 2]))
        assert exc_info.type == ValueError

    # Test a very simple correlation
    _, beta = processing.covariate_correction(X, Z)
    beta_expected = np.array([2.0, 4.0, -6.0])
    assert np.allclose(beta, beta_expected)
