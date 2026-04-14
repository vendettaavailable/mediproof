import os

import whisper

# Load Whisper model only once
model = whisper.load_model("base")


def process_audio(audio_path):
    result = model.transcribe(audio_path)

    full_text = result.get("text", "").lower()

    # keywords for medical claim
    keywords = ["cure", "prevent", "treat"]

    words = full_text.split()

    best_sentence = ""
    best_score = 0

    # sliding window (same trick we used)
    for i in range(len(words)):
        for j in range(i+4, min(i+12, len(words))):
            phrase = " ".join(words[i:j])

            score = 0

            if any(word in phrase for word in keywords):
                score += 5

            length = len(phrase.split())
            if 5 <= length <= 12:
                score += 3

            if score > best_score:
                best_score = score
                best_sentence = phrase

    return best_sentence if best_sentence else full_text