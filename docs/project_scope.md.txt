# MediProof – Project Scope Document

## 1. Project Overview

MediProof is an Explainable Multimodal Health Misinformation Detection and Correction System.

The system verifies health-related claims using:
- Machine Learning classification
- Retrieval-Augmented Generation (RAG)
- Rule-based medical reasoning
- Explainable AI (LIME & SHAP)
- Risk and confidence scoring

The goal is to build a structured, evidence-based, and explainable verification framework.


------------------------------------------------------------

## 2. Phase 1 – Core Features (Must Have)

These features must be fully working before adding advanced features.

1. Text-based health claim input
2. Claim preprocessing and extraction
3. ML classification (True / False / Misleading)
4. Retrieval-Augmented Generation (RAG) for evidence retrieval
5. Rule-based medical reasoning engine
6. LIME explanation for prediction
7. SHAP explanation for feature importance
8. Confidence score generation
9. Risk severity classification (Low / Medium / High)
10. Corrective medical clarification output
11. Structured JSON output format
12. Backend API using FastAPI


------------------------------------------------------------

## 3. Phase 2 – Advanced Features

These features will be implemented after Phase 1 is stable.

1. Image input using OCR (Tesseract)
2. Audio input using Speech-to-Text
3. Video input processing
4. Supabase database integration
5. Claim history logging
6. User interface improvements
7. Deployment to cloud platform


------------------------------------------------------------

## 4. Features NOT Included (To Prevent Scope Creep)

The following will NOT be included:

1. Real-time clinical diagnosis
2. Live doctor consultation
3. Medical prescription system
4. Training large LLM models from scratch
5. Enterprise-scale infrastructure


------------------------------------------------------------

## 5. Success Criteria

The project will be considered successful if:

1. Text-based verification works reliably
2. RAG retrieves relevant medical evidence
3. Rule-based reasoning detects contradictions
4. Explainability (LIME & SHAP) generates meaningful insights
5. Confidence and risk scores are calculated
6. Corrective medical clarification is shown
7. The system is deployable and demo-ready


------------------------------------------------------------

## 6. Development Phases Timeline

Month 1:
- Backend setup
- Text classification
- Basic ML model

Month 2:
- RAG implementation
- Rule-based reasoning
- Risk & confidence scoring

Month 3:
- Multimodal input (OCR, audio,video)
- Frontend integration

Month 4:
- Optimization
- Testing
- Documentation
- Deployment


------------------------------------------------------------

Project Scope Frozen On:
[22/2/2026]