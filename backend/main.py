from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import html
import logging
import time
import re
from pathlib import Path
from typing import List, Literal, Optional

from multimodal.claim_extractor_llm import extract_claim_llm
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
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
LOW_CONFIDENCE_THRESHOLD = 0.25
STRONG_EVIDENCE_SCORE_THRESHOLD = 0.3
UPLOAD_DIR = Path("uploads")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS

CURE_KEYWORDS = ("cure", "completely cure", "permanent cure", "cures")
CONTRADICTION_PHRASES = (
    "does not cure", "no evidence", "not effective",
    "cannot cure", "no scientific evidence", "not supported",
    "no cure", "scientifically unfounded", "misinformation",
)
SUPPORT_PHRASES = (
    "effective", "recommended", "reduces the risk",
    "supports", "proven", "approved", "undergo rigorous clinical trials",
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


try:
    import multipart  # type: ignore # noqa: F401
    MULTIPART_INSTALLED = True
except ImportError:
    MULTIPART_INSTALLED = False


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
        print("ML prediction:", {"label": verdict, "confidence": confidence})
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
        evidence_items = [
            EvidenceItem(
                content=e.get("content", ""),
                source=e.get("source", "Unknown"),
                url=e.get("url", ""),
                score=float(e.get("score", 0.0)),
            )
            for e in raw_evidence
        ]
        print("Retrieved evidence:", [item.model_dump() for item in evidence_items])
        return evidence_items
    except Exception as exc:
        logger.error("Evidence retrieval failed: %s", exc)
        fallback_item = EvidenceItem(
            content=(
                "Trusted medical guidance should come from reliable public health sources such as WHO, CDC, or NHS. "
                "Evidence retrieval is temporarily unavailable, so this claim should be verified carefully."
            ),
            source="MediProof Fallback Guidance",
            url="",
            score=0.1,
        )
        print("Retrieved evidence:", [fallback_item.model_dump()])
        return [fallback_item]


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


def _evidence_contradicts_claim(evidence: List[EvidenceItem]) -> bool:
    """
    Check whether the top evidence contains contradiction language.
    """
    if not evidence:
        return False

    top_evidence_text = evidence[0].content.lower()
    return any(phrase in top_evidence_text for phrase in CONTRADICTION_PHRASES)


def _evidence_supports_claim(evidence: List[EvidenceItem]) -> bool:
    """
    Check whether the top evidence contains supportive language.
    """
    if not evidence:
        return False

    top_evidence_text = evidence[0].content.lower()
    return any(phrase in top_evidence_text for phrase in SUPPORT_PHRASES)


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
    3) If strong contradiction evidence exists => False
    4) If strong support evidence exists and risk_level is Low => True
    5) If risk_level is Medium => Misleading
    6) If ML has a usable label => ML prediction
    7) Otherwise => Uncertain
    """
    normalized_ml = ml_verdict if ml_verdict in {"True", "False", "Misleading", "Uncertain"} else "Uncertain"
    top_score = evidence[0].score if evidence else 0.0
    has_strong_evidence = top_score > STRONG_EVIDENCE_SCORE_THRESHOLD
    contradiction_detected = _evidence_contradicts_claim(evidence)
    support_detected = _evidence_supports_claim(evidence)

    if risk_level == "High":
        return "Misleading"

    if _evidence_contradicts_cure_claim(claim, evidence):
        return "Misleading"

    if has_strong_evidence and contradiction_detected:
        return "False"

    if has_strong_evidence and support_detected and risk_level == "Low":
        return "True"

    if risk_level == "Medium":
        return "Misleading"

    if normalized_ml != "Uncertain" and ml_confidence >= LOW_CONFIDENCE_THRESHOLD:
        return normalized_ml

    if normalized_ml in {"False", "Misleading"} and contradiction_detected:
        return normalized_ml

    if normalized_ml == "True" and support_detected:
        return "True"

    return "Uncertain"


def _save_uploaded_file(file: UploadFile, content: bytes) -> Path:
    """
    Save the uploaded file to a local uploads folder.
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "uploaded_file").name
    file_path = UPLOAD_DIR / safe_name

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    return file_path


def _extract_claim_from_file(file_path: Path) -> tuple[str, str]:
    """
    Process an image or audio file and return:
    (extracted_text, final_claim)
    """
    suffix = file_path.suffix.lower()

    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        from multimodal.image_input import process_image

        extracted_text = process_image(str(file_path))
    elif suffix in SUPPORTED_AUDIO_EXTENSIONS:
        from multimodal.audio_input import process_audio

        extracted_text = process_audio(str(file_path))
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PNG, JPG, JPEG, MP3, or WAV.",
        )

    cleaned_claim = extract_claim_llm(extracted_text)
    final_claim = (cleaned_claim or extracted_text or "").strip()

    return extracted_text.strip(), final_claim


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
    print("Claim received:", claim)
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
    print("Final verdict:", final_verdict)

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


@app.post("/verify-text")
async def verify_text_input(
    text: str = Form(
        ...,
        description="Enter a plain text health claim to verify.",
    ),
):
    cleaned_text = (text or "").strip()

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="No text provided.")

    print("Extracted text:", cleaned_text)
    print("Final claim:", cleaned_text)

    payload = VerifyRequest(claim=cleaned_text)
    verification = verify_claim(payload)

    return {
        "extracted_text": cleaned_text,
        "final_claim": cleaned_text,
        "verification": verification.model_dump(),
    }


@app.post("/verify-multimodal")
async def verify_multimodal(
    file: UploadFile = File(
        ...,
        description="Choose an image or audio file containing a health claim.",
    ),
):
    if not MULTIPART_INSTALLED:
        raise HTTPException(
            status_code=500,
            detail="File upload support is missing. Install it with: pip install python-multipart",
        )

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PNG, JPG, JPEG, MP3, or WAV.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_path = _save_uploaded_file(file, file_bytes)
    extracted_text, claim = _extract_claim_from_file(file_path)

    print("Received file:", file.filename)
    print("Extracted text:", extracted_text)
    print("Final claim:", claim)

    if not claim or len(claim.strip()) < 3:
        return {
            "message": "Could not extract a clear medical claim from the uploaded file.",
            "extracted_text": extracted_text,
            "final_claim": "",
        }

    payload = VerifyRequest(claim=claim)
    verification = verify_claim(payload)

    return {
        "extracted_text": extracted_text,
        "final_claim": claim,
        "verification": verification.model_dump(),
    }
