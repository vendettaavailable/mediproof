# LIME Explainability for MediProof Classifier

## Overview
The MediProof classifier now includes LIME (Local Interpretable Model-agnostic Explanations) for understanding why individual predictions are made.

## Usage Example

### Basic Usage

```python
from ml.classifier import predict_claim, explain_prediction

# Make a prediction
claim = "Papaya cures dengue fever completely"
prediction = predict_claim(claim)
print(f"Prediction: {prediction['label']}")
print(f"Confidence: {prediction['confidence']}")

# Get explanation of why the model made this prediction
explanation = explain_prediction(claim)
print(f"Important words influencing prediction:")
for word_info in explanation['important_words']:
    print(f"  - '{word_info['word']}': {word_info['weight']:.4f}")
```

### Example Output

```
Prediction: Misleading
Confidence: 0.7234

Important words influencing prediction:
  - 'cure': 0.4200
  - 'dengue': 0.1850
  - 'completely': 0.0920
  - 'papaya': 0.0650
```

## Implementation Details

### New Functions

#### `explain_prediction(claim_text: str) -> Dict`
Generates LIME-based explanation for a claim prediction.

**Parameters:**
- `claim_text` (str): The health claim to explain

**Returns:**
```json
{
  "important_words": [
    {"word": "cure", "weight": 0.42},
    {"word": "vitamin", "weight": 0.18},
    {"word": "miracle", "weight": 0.12},
    {"word": "guaranteed", "weight": 0.08},
    {"word": "herbal", "weight": 0.06}
  ]
}
```

**Features:**
- Returns top 5 most important words
- Weights are absolute values, sorted descending
- Provided weight indicates the word's influence on the prediction
- Negative weights suggest evidence against the prediction
- Positive weights suggest evidence for the prediction

#### `_get_explainer() -> LimeTextExplainer`
Thread-safe lazy accessor for the LIME explainer instance.
- Initializes once and reuses
- Uses model's class names: "True", "False", "Misleading"
- Fixed random state for reproducibility

### How LIME Works

LIME explains predictions by:
1. Perturbing the input text (removing/changing words randomly)
2. Running predictions on all perturbations
3. Training a local linear model around the original prediction
4. Extracting coefficient weights as feature importance
5. Returning top k features by absolute weight

## Integration with Existing Pipeline

The LIME explainability works with:
- **TF-IDF Vectorizer**: Extracts features from text
- **Logistic Regression**: Classifier that LIME explains
- **predict_proba()**: Used to get prediction probabilities for perturbations
- **Thread-safe lazy loading**: Both model and explainer are cached

## Performance Notes

- First call to `explain_prediction()` is slower (initializes explainer)
- Subsequent calls are faster (reuses explainer)
- LIME generates ~1000 perturbations by default (configurable)
- Typical explanation time: 1-5 seconds per claim

## API Integration

To add explanations to the `/verify` endpoint:

```python
from main import verify_claim
from ml.classifier import explain_prediction

response = verify_claim(request)
explanation = explain_prediction(request.claim)

# Add explanation to response
response.explanation = explanation
```

## Error Handling

- If LIME explanation fails, returns empty `important_words` list
- Errors are logged but don't crash the prediction
- Prediction confidence scores remain available even if explanation fails

## Example Claims and Explanations

### Claim 1: Dangerous Misinformation
**Input:** "Bleach can cure COVID-19"
**Prediction:** False (Confidence: 0.92)
**Important words:**
- bleach: 0.55
- cure: 0.38
- covid: 0.18

### Claim 2: Misleading Health Claim
**Input:** "Garlic supplements will cure all infections"
**Prediction:** Misleading (Confidence: 0.78)
**Important words:**
- cure: 0.42
- garlic: 0.25
- supplements: 0.12
- all: 0.08

### Claim 3: Accurate Health Information
**Input:** "Regular exercise improves cardiovascular health"
**Prediction:** True (Confidence: 0.85)
**Important words:**
- exercise: 0.48
- health: 0.31
- improves: 0.15
- regular: 0.10

## References

- LIME Paper: ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- LIME Library: https://github.com/marcotcr/lime
- Documentation: https://lime-ml.readthedocs.io/
