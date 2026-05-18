"""
ML Service — Patient Risk Classifier
Trains a gradient-boosted classifier on synthetic vitals data.
Input:  age, systolic_bp, diastolic_bp, heart_rate, spo2,
        temperature, respiratory_rate, blood_glucose, consciousness (0/1)
Output: risk_level (low / medium / high) + probability scores
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_PATH = Path(__file__).parent.parent / "models" / "risk_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

# Feature order (must match frontend form)
FEATURES = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate",
    "spo2", "temperature", "respiratory_rate", "blood_glucose",
    "consciousness",  # 0 = alert, 1 = altered
]

RISK_LABELS  = ["low", "medium", "high"]
RISK_COLOURS = {"low": "#16a34a", "medium": "#d97706", "high": "#dc2626"}
RISK_ICONS   = {"low": "✅", "medium": "⚠️", "high": "🚨"}


def _generate_training_data(n: int = 2000):
    """
    Generate synthetic training data based on clinical knowledge.
    Each sample represents a patient encounter.
    Labels are rule-based so the model learns real clinical patterns.
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(n):
        age       = int(rng.integers(18, 90))
        sbp       = int(rng.integers(80, 200))
        dbp       = int(rng.integers(50, 120))
        hr        = int(rng.integers(40, 150))
        spo2      = float(rng.uniform(82, 100))
        temp      = float(rng.uniform(35.0, 40.5))
        rr        = int(rng.integers(8, 35))
        bg        = float(rng.uniform(2.5, 25.0))
        cons      = int(rng.integers(0, 2))

        # NEWS2-inspired risk labelling
        score = 0
        if rr <= 8 or rr >= 25:        score += 3
        elif rr >= 21:                  score += 2
        elif rr in [9, 10, 11]:        score += 1
        if spo2 < 92:                  score += 3
        elif spo2 < 94:                score += 2
        elif spo2 < 96:                score += 1
        if sbp < 90 or sbp >= 220:     score += 3
        elif sbp < 100:                score += 2
        elif sbp >= 180 or sbp < 110:  score += 1
        if hr < 40 or hr >= 131:       score += 3
        elif hr >= 111 or hr < 51:     score += 1
        if temp < 35.1 or temp >= 39.1: score += 2
        elif temp >= 38.1:             score += 1
        if cons == 1:                  score += 3
        if bg < 3.0 or bg > 20.0:     score += 2
        elif bg > 14.0:               score += 1
        if age > 75:                   score += 1

        if   score >= 7: label = 2   # high
        elif score >= 5: label = 1   # medium
        else:            label = 0   # low

        X.append([age, sbp, dbp, hr, spo2, temp, rr, bg, cons])
        y.append(label)

    return np.array(X), np.array(y)


class MLService:
    """Trains and serves the patient risk classification model."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.available = SKLEARN_AVAILABLE

    def load_model(self):
        if not self.available:
            print("⚠️  scikit-learn not installed — ML risk scoring disabled")
            return
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.pipeline = pickle.load(f)
            print("🤖 Risk model loaded from cache")
        else:
            self._train_and_save()

    def _train_and_save(self):
        print("🏋️  Training risk classifier on synthetic data...")
        X, y = _generate_training_data(2000)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.08, random_state=42,
            )),
        ])
        self.pipeline.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)
        print("✅ Risk model trained and saved")

    def predict(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available or self.pipeline is None:
            return self._rule_based_fallback(patient)

        features = np.array([[
            patient.get("age", 45),
            patient.get("systolic_bp", 120),
            patient.get("diastolic_bp", 80),
            patient.get("heart_rate", 72),
            patient.get("spo2", 98),
            patient.get("temperature", 37.0),
            patient.get("respiratory_rate", 16),
            patient.get("blood_glucose", 5.5),
            patient.get("consciousness", 0),
        ]])

        pred  = int(self.pipeline.predict(features)[0])
        proba = self.pipeline.predict_proba(features)[0].tolist()

        risk_label = RISK_LABELS[pred]

        # Generate clinical reasoning
        flags = self._get_flags(patient)

        return {
            "risk_level":   risk_label,
            "risk_score":   round(proba[pred] * 100),
            "probabilities": {
                "low":    round(proba[0] * 100, 1),
                "medium": round(proba[1] * 100, 1),
                "high":   round(proba[2] * 100, 1),
            },
            "colour": RISK_COLOURS[risk_label],
            "icon":   RISK_ICONS[risk_label],
            "flags":  flags,
            "recommendation": self._recommendation(risk_label, flags),
        }

    def _get_flags(self, p: Dict) -> List[str]:
        flags = []
        if p.get("spo2", 100) < 94:     flags.append(f"Low SpO₂: {p['spo2']}%")
        if p.get("systolic_bp", 120) < 90: flags.append(f"Hypotension: {p['systolic_bp']} mmHg")
        if p.get("systolic_bp", 120) > 180: flags.append(f"Hypertensive crisis: {p['systolic_bp']} mmHg")
        if p.get("heart_rate", 72) > 120: flags.append(f"Tachycardia: {p['heart_rate']} bpm")
        if p.get("heart_rate", 72) < 50:  flags.append(f"Bradycardia: {p['heart_rate']} bpm")
        if p.get("respiratory_rate", 16) > 24: flags.append(f"Tachypnoea: {p['respiratory_rate']} breaths/min")
        if p.get("temperature", 37.0) > 38.5: flags.append(f"Fever: {p['temperature']}°C")
        if p.get("temperature", 37.0) < 35.5: flags.append(f"Hypothermia: {p['temperature']}°C")
        if p.get("blood_glucose", 5.5) < 3.5: flags.append(f"Hypoglycaemia: {p['blood_glucose']} mmol/L")
        if p.get("blood_glucose", 5.5) > 15:   flags.append(f"Hyperglycaemia: {p['blood_glucose']} mmol/L")
        if p.get("consciousness", 0) == 1:      flags.append("Altered consciousness")
        return flags

    def _recommendation(self, risk: str, flags: List[str]) -> str:
        if risk == "high":
            return "🚨 URGENT: Immediate medical review required. Consider emergency escalation. Monitor continuously."
        elif risk == "medium":
            return "⚠️ ALERT: Clinical review needed within 30 minutes. Increase monitoring frequency."
        else:
            return "✅ ROUTINE: Continue standard monitoring. Review at next scheduled assessment."

    def _rule_based_fallback(self, p: Dict) -> Dict:
        flags = self._get_flags(p)
        risk = "high" if len(flags) >= 3 else "medium" if len(flags) >= 1 else "low"
        return {
            "risk_level": risk, "risk_score": 60 if risk == "high" else 40 if risk == "medium" else 20,
            "probabilities": {"low": 20.0, "medium": 40.0, "high": 40.0},
            "colour": RISK_COLOURS[risk], "icon": RISK_ICONS[risk],
            "flags": flags, "recommendation": self._recommendation(risk, flags),
        }
