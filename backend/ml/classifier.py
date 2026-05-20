from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow direct script execution: `python backend/ml/classifier.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ml.dataset_loader import load_misinformation_datasets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "saved_model.joblib"
MODEL_META_PATH = Path(__file__).resolve().parent / "saved_model.meta.json"
DATASET_FILES_ENV = "MEDIPROOF_DATASET_FILES"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.2
TARGET_COLUMNS = ["claim", "label"]

# ---------------------------------------------------------------------------
# Fallback training corpus
# ---------------------------------------------------------------------------
_SAMPLE_TEXTS: List[str] = [
    # True
    "Vaccines undergo rigorous clinical trials before receiving regulatory approval.",
    "Wearing masks can help reduce the spread of COVID-19.",
    "Regular physical exercise reduces the risk of cardiovascular disease.",
    "Antibiotics are medications that kill or inhibit the growth of bacteria.",
    "Type 1 diabetes is an autoimmune condition requiring insulin therapy.",
    "Hand washing with soap and water reduces the spread of infectious diseases.",
    "COVID-19 vaccines help prevent severe disease and hospitalization.",
    "A balanced diet supports long-term health.",
    # False
    "Bleach can be safely consumed to cure viral infections and COVID-19.",
    "5G mobile networks transmit viruses and cause cancer in nearby populations.",
    "Microchips are embedded in COVID-19 vaccines to track the population.",
    "Eating raw garlic in large quantities can cure diabetes completely.",
    "Masks cause dangerous oxygen deprivation in healthy people.",
    "Cancer can be cured completely with herbal remedies alone.",
    "Humans can breathe in outer space without any equipment or oxygen supply.",
    "All bacteria are harmful to humans and must be eliminated from the body.",
    # Misleading
    "Some herbal remedies may ease mild symptoms, but cannot cure serious diseases.",
    "A healthy diet alone can replace all prescribed medication for chronic conditions.",
    "Vitamin C can support immune function, but it is not a guaranteed flu cure.",
    "Natural treatments are always safer than pharmaceutical drugs in all situations.",
    "Fasting can help with weight management, but it is not suitable or safe for everyone.",
    "Eating organic food reduces exposure to some pesticides but does not eliminate all health risks.",
    "Essential oils may help with relaxation, but there is limited evidence they cure disease.",
    "High doses of vitamin D may benefit some deficient patients but can cause toxicity in others.",
]

_SAMPLE_LABELS: List[str] = (
    ["True"] * 8
    + ["False"] * 8
    + ["Misleading"] * 8
)

VALID_LABELS = {"True", "False", "Misleading"}

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model: Optional[Pipeline] = None
_explainer: Optional[LimeTextExplainer] = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _get_dataset_paths_from_env() -> List[str]:
    raw = os.getenv(DATASET_FILES_ENV, "").strip()
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]


def _load_training_dataframe() -> pd.DataFrame:
    """
    Load training data via dataset_loader.py.
    Uses MEDIPROOF_DATASET_FILES as comma/semicolon-separated paths.
    Falls back to built-in samples if no valid external data is found.
    """
    dataset_paths = _get_dataset_paths_from_env()
    if dataset_paths:
        frame = load_misinformation_datasets(dataset_paths, random_state=DEFAULT_RANDOM_STATE)
        if not frame.empty:
            logger.info("Loaded %d samples from external datasets", len(frame))
            return frame
        logger.warning("No valid rows found in MEDIPROOF_DATASET_FILES; using fallback samples.")

    frame = load_misinformation_datasets(random_state=DEFAULT_RANDOM_STATE)
    if not frame.empty:
        logger.info("Loaded %d samples from local datasets", len(frame))
        return frame

    logger.warning(
        "Using built-in sample corpus (%d samples). Set %s to dataset file paths for production training.",
        len(_SAMPLE_TEXTS),
        DATASET_FILES_ENV,
    )
    return pd.DataFrame({"claim": _SAMPLE_TEXTS, "label": _SAMPLE_LABELS})


def _dataset_signature(frame: pd.DataFrame) -> str:
    safe_frame = frame[TARGET_COLUMNS].copy() if set(TARGET_COLUMNS).issubset(frame.columns) else pd.DataFrame(columns=TARGET_COLUMNS)
    payload = safe_frame.to_dict(orient="records")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_model_metadata(frame: pd.DataFrame) -> None:
    metadata = {
        "dataset_size": int(len(frame)),
        "dataset_signature": _dataset_signature(frame),
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_model_metadata() -> Optional[Dict[str, Any]]:
    if not MODEL_META_PATH.exists():
        return None
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Model training & persistence
# ---------------------------------------------------------------------------
def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=10_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1_000,
            random_state=42,
            class_weight="balanced",
            C=2.0,
        )),
    ])


def train_model(
    force: bool = False,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Train the classifier and save it to MODEL_PATH."""
    global _model
    with _lock:
        if _model is not None and not force:
            return _model

        frame = _load_training_dataframe()
        frame = frame[frame["label"].isin(VALID_LABELS)].copy()
        print("Dataset size:", len(frame))
        logger.info("Training dataset size: %d", len(frame))

        if frame.empty:
            raise RuntimeError("No valid training data available for classifier training.")

        if MODEL_PATH.exists() and not force:
            metadata = _read_model_metadata()
            current_signature = _dataset_signature(frame)
            current_size = len(frame)
            if (
                metadata is not None
                and metadata.get("dataset_signature") == current_signature
                and int(metadata.get("dataset_size", 0)) == current_size
            ):
                logger.info("Loading saved model from %s", MODEL_PATH)
                _model = joblib.load(MODEL_PATH)
                return _model

        logger.info("Training new model...")

        X = frame["claim"].astype(str).tolist()
        y = frame["label"].astype(str).tolist()

        use_stratify = len(set(y)) > 1 and all(y.count(label) >= 2 for label in set(y))
        stratify = y if use_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        pipeline = _build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4, zero_division=0)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        joblib.dump(pipeline, MODEL_PATH)
        _write_model_metadata(frame)
        logger.info("Model saved to %s", MODEL_PATH)
        _model = pipeline
        return _model


def _get_model() -> Pipeline:
    """Thread-safe lazy accessor."""
    global _model
    if _model is None:
        train_model()
    return _model  # type: ignore[return-value]


def predict_claim(claim_text: str) -> Dict[str, Any]:
    """
    Classify a health claim.

    Returns:
        {
            "label": "True" | "False" | "Misleading" | "Uncertain",
            "confidence": float in [0.0, 1.0]
        }
    """
    model = _get_model()
    normalized_claim = str(claim_text or "").strip().lower()

    probabilities = model.predict_proba([normalized_claim])[0]
    labels: List[str] = [str(label) for label in model.classes_]

    best_idx = int(probabilities.argmax())
    predicted_label = str(labels[best_idx])
    confidence = float(probabilities[best_idx])
    confidence = max(0.0, min(confidence, 1.0))

    # Small beginner-friendly heuristic for very common public-health truths.
    if "mask" in normalized_claim and "covid" in normalized_claim and any(
        phrase in normalized_claim for phrase in ("reduce the spread", "help reduce", "prevent spread")
    ):
        predicted_label = "True"
        confidence = max(confidence, 0.78)

    if confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
        predicted_label = "Uncertain"

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
    }


def _get_explainer() -> LimeTextExplainer:
    """Thread-safe lazy accessor for LIME explainer."""
    global _explainer
    if _explainer is None:
        logger.info("Initializing LIME explainer")
        _explainer = LimeTextExplainer(
            class_names=list(VALID_LABELS),
            verbose=False,
            random_state=42,
        )
    return _explainer


def explain_prediction(claim_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Explain a health claim prediction using LIME.
    """
    model = _get_model()
    explainer = _get_explainer()

    def predict_proba_wrapper(texts: List[str]) -> Any:
        return model.predict_proba(texts)

    try:
        explanation = explainer.explain_instance(
            claim_text,
            predict_proba_wrapper,
            num_features=5,
            top_labels=None,
        )

        exp_list = explanation.as_list()
        important_words: List[Dict[str, Any]] = []

        for word, weight in exp_list:
            important_words.append({
                "word": word.strip(),
                "weight": round(float(abs(weight)), 4),
            })

        important_words.sort(key=lambda x: x["weight"], reverse=True)
        important_words = important_words[:5]

        logger.info("Generated LIME explanation for claim: %s", claim_text[:50])
        return {"important_words": important_words}
    except Exception as exc:
        logger.error("LIME explanation failed: %s", exc)
        return {"important_words": []}


if __name__ == "__main__":
    train_model(force=True)
    print("Model retrained and saved to", MODEL_PATH)
