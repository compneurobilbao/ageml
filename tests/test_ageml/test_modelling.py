import pytest
import numpy as np
import ageml.modelling as modelling


# TODO: Do this as a fixture
# Class for quickly initializing
class AgeMLTest(modelling.AgeML):
    def __init__(
        self,
        scaler="standard",
        scaler_params={"with_mean": True},
        model="linear_reg",
        model_params={"fit_intercept": True},
        CV_split=5,
        seed=42,
    ):
        self.scaler = scaler
        self.scaler_params = scaler_params  # For the 'copy' param. Placeholding
        self.model = model
        self.model_params = model_params  # For the 'fit_intercept' param. Placeholding
        self.CV_split = CV_split
        self.seed = seed
        super().__init__(
            self.scaler,
            self.scaler_params,
            self.model,
            self.model_params,
            self.CV_split,
            self.seed,
        )


@pytest.fixture
def dummy_classifier():
    return modelling.Classifier()


def test_set_unavailable_scaler():
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy = AgeMLTest(scaler="Mondong")
        del age_ml_dummy  # Avoid linting error regarding unused variable
    assert exc_info.type is ValueError
    assert str(exc_info.value) == f"Scaler 'Mondong' not registered. Available: {modelling.ScalerRegistry.list_scalers()}"


def test_set_unavailable_model():
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy = AgeMLTest(model="Mondong")
        del age_ml_dummy  # To avoid linting error regarding unused variable
    assert exc_info.type is ValueError
    assert str(exc_info.value) == f"Model 'Mondong' not registered. Available: {modelling.ModelRegistry.list_models()}"


def test_set_pipeline_none_model():
    # Instantiate a correct object
    age_ml_dummy = AgeMLTest()

    # Set a None model in an unauthorized manner (direct modification)
    age_ml_dummy.model = None  # Do not do this in practice please!

    # Set the pipeline to trigger the ValueError
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy.set_pipeline()
    assert exc_info.type is ValueError
    error_message = "Must set a valid model before setting pipeline."
    assert str(exc_info.value) == error_message

    # Restore for the next case
    age_ml_dummy.set_model("linear_reg")

    # Set None scaler in an unauthorized manner (direct modification)
    age_ml_dummy.scaler = None
    age_ml_dummy.set_pipeline()
    # Check that the pipeline only has one step now
    assert len(age_ml_dummy.pipeline.steps) == 1

    # Pipeline should stay valid when model is reset to a registered model
    age_ml_dummy.set_model("ridge")
    age_ml_dummy.set_pipeline()
    assert age_ml_dummy.pipeline is not None


# TODO: test: metrics, summary_metrics, fit_age_bias, predict_age_bias, fit_age, predict_age
# TODO: check all errors raised

# def test_fit_age():
#     pass


def test_classifier_fit_age(dummy_classifier):
    # Create data
    x = np.concatenate((np.zeros(1000), np.ones(1000)))
    y = np.concatenate((np.zeros(1000), np.ones(1000)))

    # Modify x
    x[0] = 1
    x[1000] = 0
    x = x.reshape(-1, 1)

    # Fit
    y_pred = dummy_classifier.fit_model(x, y)

    # Assert
    assert y_pred[0] > 0.5
    assert y_pred[1000] < 0.5
    for i in range(1, 1000):
        assert y_pred[i] < 0.5
        assert y_pred[i + 1000] > 0.5


def test_classification_predict_error(dummy_classifier):
    # Data
    x = [1, 2, 3]
    with pytest.raises(ValueError) as exc_info:
        dummy_classifier.predict(x)
    assert exc_info.type is ValueError
    error_message = "Must fit the classifier before calling predict."
    assert str(exc_info.value) == error_message

def test_classifier_predict_scale(dummy_classifier):
    """Test that the predict method raises an error if the model was not fitted with scaling and scale=True is passed."""

    # Data
        # Create data
    x = np.concatenate((np.zeros(1000), np.ones(1000)))
    y = np.concatenate((np.zeros(1000), np.ones(1000)))

    # Modify x
    x[0] = 1
    x[1000] = 0
    x = x.reshape(-1, 1)

    # Fit
    _ = dummy_classifier.fit_model(x, y)

    # New data
    x_new = np.array([0.5, 0.5]).reshape(-1, 1)

    # Assert error risen as not trained with scale
    with pytest.raises(ValueError) as exc_info:
        dummy_classifier.predict(x_new, scale=True)
    assert exc_info.type is ValueError
    error_message = "Must fit the model with scaling before calling predict with scaling."
    assert str(exc_info.value) == error_message


def test_fit_age_permutation_null_model():
    """Permutation null model should produce finite null MAEs and a valid p-value."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = 50 + 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.5, size=40)

    age_ml = modelling.AgeML(
        "standard",
        {"with_mean": True},
        "linear_reg",
        {"fit_intercept": True},
        CV_split=4,
        seed=42,
        null_model_permutations=5,
    )

    pred_age, corrected_age = age_ml.fit_age(X, y)

    assert pred_age.shape == y.shape
    assert corrected_age.shape == y.shape
    assert age_ml.null_model_scores.shape == (5,)
    assert np.isfinite(age_ml.null_model_scores).all()
    assert age_ml.null_model_p_value is not None
    assert 0.0 <= age_ml.null_model_p_value <= 1.0
