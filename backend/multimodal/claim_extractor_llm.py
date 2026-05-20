import os

try:
    from groq import Groq
except ImportError:
    Groq = None

# ✅ Correct: use variable name, not actual key
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
    original_text = (text or "").strip()

    if not original_text:
        return "No claim found"

    if client is None:
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
            return original_text

        return claim

    except Exception:
        return original_text