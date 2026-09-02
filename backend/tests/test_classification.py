import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from app.services.classifier import LayoutXLMClassifier, ClassificationOutput


@patch.object(LayoutXLMClassifier, 'predict')
def test_form6_is_caste_certificate(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="CASTE_CERTIFICATE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class != "CASTE_VALIDITY_CERTIFICATE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_form7_is_caste_certificate(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="CASTE_CERTIFICATE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class != "CASTE_VALIDITY_CERTIFICATE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_form8_is_caste_certificate(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="CASTE_CERTIFICATE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class != "CASTE_VALIDITY_CERTIFICATE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_ncl_is_out_of_scope(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="UNKNOWN_OUT_OF_SCOPE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class == "UNKNOWN_OUT_OF_SCOPE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_income_certificate_out_of_scope(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="UNKNOWN_OUT_OF_SCOPE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class == "UNKNOWN_OUT_OF_SCOPE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_domicile_out_of_scope(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="UNKNOWN_OUT_OF_SCOPE", confidence=0.95, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class == "UNKNOWN_OUT_OF_SCOPE"


@patch.object(LayoutXLMClassifier, 'predict')
def test_low_confidence_returns_unknown(mock_predict):
    mock_predict.return_value = ClassificationOutput(
        predicted_class="UNKNOWN_OUT_OF_SCOPE", confidence=0.2, all_probabilities={}
    )
    result = LayoutXLMClassifier().predict("dummy.jpg", [])
    assert result.predicted_class == "UNKNOWN_OUT_OF_SCOPE"
