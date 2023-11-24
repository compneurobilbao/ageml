import pytest
import os
import ageml.modelling as modelling


# Class for quickly initializing
class AgeMLTest(modelling.AgeML):
    def __init__(
        self,
        scaler="standard",
        scaler_params={"with_mean": True},
        model="linear",
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


def test_set_unavailable_scaler():
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy = AgeMLTest(scaler="Mondong")
        del age_ml_dummy  # To avoid linting error regarding unused variable
    assert exc_info.type == ValueError
    assert str(exc_info.value) == "Must select an available scaler type."


def test_set_unavailable_model():
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy = AgeMLTest(model="Mondong")
        del age_ml_dummy  # To avoid linting error regarding unused variable
    assert exc_info.type == ValueError
    assert str(exc_info.value) == "Must select an available model type."


def test_set_pipeline_none_model():
    # Instantiate a correct object
    age_ml_dummy = AgeMLTest()

    # Set a None model in an unauthorized manner (direct modification)
    age_ml_dummy.model = None  # Do not do this in practice please!

    # Set the pipeline to trigger the ValueError
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy.set_pipeline()
    assert exc_info.type == ValueError
    error_message = "Must set a valid model or scaler before setting pipeline."
    assert str(exc_info.value) == error_message

    # Restore for the next case
    age_ml_dummy.set_model("linear")

    # Set None scaler in an unauthorized manner (direct modification)
    age_ml_dummy.scaler = None
    # Set the pipeline to trigger the Value Error again
    with pytest.raises(ValueError) as exc_info:
        age_ml_dummy.set_pipeline()
    assert exc_info.type == ValueError
    error_message = "Must set a valid model or scaler before setting pipeline."
    assert str(exc_info.value) == error_message


# TODO: test: metrics, summary_metrics, fit_age_bias, predict_age_bias, fit_age, predict_age
# TODO: check all errors raised


def test_fit_age():
    pass
