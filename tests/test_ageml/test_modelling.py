import pytest
import os
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
    assert exc_info.type == ValueError
    assert str(exc_info.value) == f"Must select an available scaler type. Available: {list(AgeMLTest().scaler_dict.keys())}"


def test_set_unavailable_model():
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy = AgeMLTest(model="Mondong")
        del age_ml_dummy  # To avoid linting error regarding unused variable
    assert exc_info.type == ValueError
    assert str(exc_info.value) == f"Must select an available model type. Available: {list(AgeMLTest().model_dict.keys())}"


def test_set_pipeline_none_model():
    # Instantiate a correct object
    age_ml_dummy = AgeMLTest()

    # Set a None model in an unauthorized manner (direct modification)
    age_ml_dummy.model = None  # Do not do this in practice please!

    # Set the pipeline to trigger the ValueError
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy.set_pipeline()
    assert exc_info.type == ValueError
    error_message = "Must set a valid model before setting pipeline."
    assert str(exc_info.value) == error_message

    # Restore for the next case
    age_ml_dummy.set_model("linear_reg")

    # Set None scaler in an unauthorized manner (direct modification)
    age_ml_dummy.scaler = None
    age_ml_dummy.set_pipeline()
    # Check that the pipeline only has one step now
    assert len(age_ml_dummy.pipeline.steps) == 1

    # Set the model to 'hyperopt' to check that the pipeline is none
    age_ml_dummy.set_model("hyperopt")
    age_ml_dummy.set_pipeline()
    assert age_ml_dummy.pipeline is None


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
    assert exc_info.type == ValueError
    error_message = "Must fit the classifier before calling predict."
    assert str(exc_info.value) == error_message
