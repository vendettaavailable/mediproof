import os

try:
    from groq import Groq
except ImportError:
    Groq = None

API_KEY = os.getenv("GROQ_API_KEY")

# Load client once globally
client = Groq(api_key=API_KEY) if Groq and API_KEY else None

PROMPT = """
You extract the main medical claim from noisy text.

Rules:
- Extract only the main medical or health-related claim.
- Ignore greetings, ads, opinions, social media noise, and unrelated text.
- Return only one clean sentence.
- Return only the claim.
- Do not add explanation.
- If there is no medical claim, return an empty string.
""".strip()


def extract_claim_llm(text):
    """
    Extract the main medical claim from noisy text using Groq.

    Args:
        text: Input text from OCR, Whisper, or other multimodal sources.

    Returns:
        Clean claim text as a single sentence, or an empty string on failure.
    """
    original_text = (text or "").strip()

    if not original_text:
        print("Groq fallback: input text is empty, returning placeholder text.")
        return "No claim found"

    if client is None:
        print("Groq fallback: client is not available, returning original text.")
        return original_text

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": original_text},
            ],
            temperature=0,
        )

        claim = response.choices[0].message.content.strip()

        if len(claim) < 3:
            print("Groq fallback: empty or too-short LLM output, returning original text.")
            return original_text

        return claim
    except Exception as exc:
        print(f"Groq fallback: request failed ({exc}), returning original text.")
        return original_text
