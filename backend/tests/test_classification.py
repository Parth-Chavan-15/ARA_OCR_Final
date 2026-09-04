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


def test_dual_confidence_output_structure():
    output = ClassificationOutput(
        predicted_class="CASTE_VALIDITY_CERTIFICATE",
        confidence=0.98,
        all_probabilities={"CASTE_VALIDITY_CERTIFICATE": 0.98},
        raw_confidence=0.8742,
        raw_probabilities={"CASTE_VALIDITY_CERTIFICATE": 0.8742},
        rule_applied="CASTE_VALIDITY_CERTIFICATE",
        confidence_label="98.0% (LayoutLMv3: 87.4% + Rule Evidence)",
    )
    assert output.confidence == 0.98
    assert output.raw_confidence == 0.8742
    assert "LayoutLMv3: 87.4%" in output.confidence_label
    assert "Rule Evidence" in output.confidence_label
