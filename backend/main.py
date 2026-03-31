from __future__ import annotations

import html
import logging
import time
import re
from typing import List, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ml.classifier import predict_claim
from rag.embeddings import retrieve_evidence
from rag.explanation_generator import generate_explanation
from rules.medical_rules import detect_medical_risk, extract_suspicious_keywords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediProof API",
    description="Explainable health misinformation detection system",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CLAIM_LENGTH = 2000
EVIDENCE_TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.4
STRONG_EVIDENCE_SCORE_THRESHOLD = 0.4

CURE_KEYWORDS = ("cure", "completely cure", "permanent cure", "cures")
CONTRADICTION_PHRASES = (
    "does not cure", "no evidence", "not effective",
    "cannot cure", "no scientific evidence", "not supported",
    "no cure", "scientifically unfounded", "misinformation",
)


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=MAX_CLAIM_LENGTH)

    @field_validator("claim", mode="before")
    @classmethod
    def sanitize_claim(cls, v: str):
        v = html.escape(v.strip())
        v = re.sub(r"\s+", " ", v)
        return v


class EvidenceItem(BaseModel):
    content: str
    source: str
    url: str
    score: float


class VerifyResponse(BaseModel):
    claim: str
    verdict: Literal["True", "False", "Misleading", "Uncertain"]
    confidence_score: float = Field(..., ge=1.0, le=10.0)
    confidence_level: Literal["Low", "High"]
    risk_level: Literal["Low", "Medium", "High"]
    risk_explanation: str
    suspicious_keywords: List[str]
    corrective_information: str
    evidence: List[EvidenceItem]


# ---------------------------------------------------------------------------
# Exception Handling
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "MediProof backend running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pipeline Functions (Modular & Testable)
# ---------------------------------------------------------------------------

def _get_ml_classification(claim: str) -> tuple[str, float]:
    """
    Run ML classifier on the claim.
    
    Returns:
        (verdict, confidence) where verdict is one of:
        "True", "False", "Misleading", "Uncertain"
    """
    try:
        result = predict_claim(claim)
        verdict = result.get("label", "Uncertain")
        confidence = float(result.get("confidence", 0.0))
        return verdict, confidence
    except Exception as exc:
        logger.error("ML classification failed: %s", exc)
        return "Uncertain", 0.0


def _get_medical_risk(claim: str) -> tuple[str, str]:
    """
    Apply medical rules to detect health risks.
    
    Returns:
        (risk_level, explanation) where risk_level is one of:
        "Low", "Medium", "High"
    """
    try:
        result = detect_medical_risk(claim)
        return result.get("risk_level", "Low"), result.get("explanation", "")
    except Exception as exc:
        logger.error("Medical risk detection failed: %s", exc)
        return "Low", "Unable to assess medical risk."


def _get_evidence(claim: str, top_k: int = EVIDENCE_TOP_K) -> List[EvidenceItem]:
    """
    Retrieve evidence using RAG retrieval.
    
    Returns:
        List of EvidenceItem objects ranked by similarity score.
    """
    try:
        raw_evidence = retrieve_evidence(claim, top_k=top_k)
        return [
            EvidenceItem(
                content=e.get("content", ""),
                source=e.get("source", "Unknown"),
                url=e.get("url", ""),
                score=float(e.get("score", 0.0)),
            )
            for e in raw_evidence
        ]
    except Exception as exc:
        logger.error("Evidence retrieval failed: %s", exc)
        return []


def _evidence_contradicts_cure_claim(claim: str, evidence: List[EvidenceItem]) -> bool:
    """
    Check if claim contains cure language but evidence contradicts it.
    """
    if not evidence:
        return False

    claim_lower = claim.lower()
    has_cure_claim = any(keyword in claim_lower for keyword in CURE_KEYWORDS)
    if not has_cure_claim:
        return False

    top_evidence_text = evidence[0].content.lower()
    has_contradiction = any(phrase in top_evidence_text for phrase in CONTRADICTION_PHRASES)
    return has_contradiction


def _combine_verdict(
    claim: str,
    ml_verdict: str,
    ml_confidence: float,
    risk_level: str,
    evidence: List[EvidenceItem],
) -> str:
    """
    Combine ML, rules, and RAG evidence into final verdict.

    Priority rules:
    1) If risk_level is High => Misleading
    2) If evidence contradicts cure claim => Misleading
    3) If top evidence score > 0.4 and risk_level is Low => True
    4) If ML confidence < 0.4 and no strong evidence => Uncertain
    5) Otherwise => ML prediction
    """
    normalized_ml = ml_verdict if ml_verdict in {"True", "False", "Misleading", "Uncertain"} else "Uncertain"
    top_score = evidence[0].score if evidence else 0.0
    has_strong_evidence = top_score > STRONG_EVIDENCE_SCORE_THRESHOLD

    if risk_level == "High":
        return "Misleading"

    if _evidence_contradicts_cure_claim(claim, evidence):
        return "Misleading"

    if has_strong_evidence and risk_level == "Low":
        return "True"

    if ml_confidence < LOW_CONFIDENCE_THRESHOLD and not has_strong_evidence:
        return "Uncertain"

    return normalized_ml


# ---------------------------------------------------------------------------
# Main Verification Endpoint
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
def verify_claim(payload: VerifyRequest):
    """
    Verify a health claim using an integrated ML + rules + evidence pipeline.
    
    Pipeline:
        1. ML classification (True/False/Misleading/Uncertain)
        2. Medical risk detection (rules-based)
        3. Suspicious keyword extraction
        4. Evidence retrieval (RAG + semantic search)
        5. Verdict fusion
        6. Corrective information generation
    """
    start = time.perf_counter()
    
    claim = payload.claim
    logger.info("Processing claim: %s", claim)
    
    # Step 1: ML Classification
    ml_label, ml_confidence = _get_ml_classification(claim)
    
    # Step 2: Medical risk detection
    risk_level, risk_explanation = _get_medical_risk(claim)

    # Step 3: Suspicious keyword extraction
    suspicious_keywords = extract_suspicious_keywords(claim)
    
    # Step 4: Evidence retrieval
    evidence = _get_evidence(claim, top_k=EVIDENCE_TOP_K)
    
    # Step 5: Combine signals for final verdict
    final_verdict = _combine_verdict(
        claim=claim,
        ml_verdict=ml_label,
        ml_confidence=ml_confidence,
        risk_level=risk_level,
        evidence=evidence,
    )

    # Step 6: Convert confidence to 1-10 scale
    confidence_score = round(ml_confidence * 10, 2)
    confidence_level = "Low" if confidence_score < 5 else "High"

    # Step 7: Evidence-based explanation
    explanation = generate_explanation(
        claim,
        [item.model_dump() for item in evidence],
    )
    
    elapsed = time.perf_counter() - start
    logger.info(
        "Verification complete: ml_verdict=%s, final_verdict=%s, confidence=%.2f, risk=%s, time=%.2fs",
        ml_label, final_verdict, ml_confidence, risk_level, elapsed,
    )

    return VerifyResponse(
        claim=claim,
        verdict=final_verdict,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        risk_level=risk_level,
        risk_explanation=risk_explanation,
        suspicious_keywords=suspicious_keywords,
        corrective_information=explanation,
        evidence=evidence,
    )


from fastapi import UploadFile, File

@app.post("/verify-multimodal")
async def verify_multimodal(
    text: str = None,
    file: UploadFile = File(None)
):
    if text:
        claim = text

    elif file:
        file_path = f"temp_{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Detect file type
        if file.filename.endswith(("png", "jpg", "jpeg")):
            from multimodal.image_input import process_image
            claim = process_image(file_path)

        elif file.filename.endswith(("mp3", "wav")):
            from multimodal.audio_input import process_audio
            claim = process_audio(file_path)

        elif file.filename.endswith(("mp4", "mov")):
            from multimodal.video_input import process_video
            claim = process_video(file_path)

        else:
            return {"error": "Unsupported file type"}

    else:
        return {"error": "No input provided"}

    #  IMPORTANT LINE (CALL YOUR EXISTING FUNCTION)
    payload = VerifyRequest(claim=claim)
    return verify_claim(payload)