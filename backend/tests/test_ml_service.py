"""Unit tests for the patient risk scoring service.

These exercise the deterministic, rule-based parts of MLService: flag
detection, the recommendation text, and the fallback used when
scikit-learn is unavailable. None of them train a model or touch disk,
so the suite stays fast and has no external dependencies.
"""

import pytest

from services.ml_service import MLService

HEALTHY = {
    "age": 34,
    "systolic_bp": 118,
    "diastolic_bp": 76,
    "heart_rate": 68,
    "spo2": 98.0,
    "temperature": 36.8,
    "respiratory_rate": 14,
    "blood_glucose": 5.1,
    "consciousness": 0,
}


def vitals(**overrides):
    """A healthy baseline patient with the given values replaced."""
    patient = dict(HEALTHY)
    patient.update(overrides)
    return patient


@pytest.fixture
def service():
    # No load_model() call: these tests only cover logic that does not
    # need a trained pipeline.
    return MLService()


class TestGetFlags:
    def test_healthy_patient_raises_no_flags(self, service):
        assert service._get_flags(HEALTHY) == []

    @pytest.mark.parametrize(
        "overrides, expected_substring",
        [
            ({"spo2": 88.0}, "Low SpO"),
            ({"systolic_bp": 84}, "Hypotension"),
            ({"systolic_bp": 195}, "Hypertensive crisis"),
            ({"heart_rate": 135}, "Tachycardia"),
            ({"heart_rate": 44}, "Bradycardia"),
            ({"respiratory_rate": 28}, "Tachypnoea"),
            ({"temperature": 39.2}, "Fever"),
            ({"temperature": 34.9}, "Hypothermia"),
            ({"blood_glucose": 2.8}, "Hypoglycaemia"),
            ({"blood_glucose": 19.0}, "Hyperglycaemia"),
            ({"consciousness": 1}, "Altered consciousness"),
        ],
    )
    def test_single_abnormal_value_is_flagged(
        self, service, overrides, expected_substring
    ):
        flags = service._get_flags(vitals(**overrides))
        assert any(expected_substring in flag for flag in flags), flags

    def test_boundaries_are_not_flagged(self, service):
        """Values sitting exactly on a threshold stay unflagged."""
        borderline = vitals(
            spo2=94.0, systolic_bp=90, heart_rate=120, respiratory_rate=24
        )
        assert service._get_flags(borderline) == []

    def test_multiple_abnormal_values_produce_multiple_flags(self, service):
        flags = service._get_flags(vitals(spo2=85.0, heart_rate=140, consciousness=1))
        assert len(flags) == 3

    def test_missing_keys_fall_back_to_healthy_defaults(self, service):
        assert service._get_flags({}) == []


class TestRuleBasedFallback:
    def test_no_flags_scores_low(self, service):
        assert service._rule_based_fallback(HEALTHY)["risk_level"] == "low"

    def test_one_flag_scores_medium(self, service):
        result = service._rule_based_fallback(vitals(temperature=39.0))
        assert result["risk_level"] == "medium"

    def test_three_flags_score_high(self, service):
        result = service._rule_based_fallback(
            vitals(spo2=85.0, heart_rate=140, consciousness=1)
        )
        assert result["risk_level"] == "high"

    def test_result_shape_is_complete(self, service):
        result = service._rule_based_fallback(HEALTHY)
        assert set(result) == {
            "risk_level",
            "risk_score",
            "probabilities",
            "colour",
            "icon",
            "flags",
            "recommendation",
        }
        assert set(result["probabilities"]) == {"low", "medium", "high"}

    def test_predict_uses_fallback_when_no_pipeline_is_loaded(self, service):
        service.pipeline = None
        result = service.predict(vitals(spo2=85.0, heart_rate=140, consciousness=1))
        assert result["risk_level"] == "high"


class TestRecommendation:
    @pytest.mark.parametrize(
        "risk, marker",
        [("high", "URGENT"), ("medium", "ALERT"), ("low", "ROUTINE")],
    )
    def test_each_risk_level_has_its_own_guidance(self, service, risk, marker):
        assert marker in service._recommendation(risk, [])
