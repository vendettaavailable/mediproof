import re

import easyocr

# Load reader globally (only once)
reader = easyocr.Reader(['en'])

# Medical claim keywords
CLAIM_KEYWORDS = ['cure', 'prevent', 'treat', 'reduce', 'stop', 'heal']

# Health-related terms to strengthen claim detection
HEALTH_TERMS = [
    'disease', 'illness', 'pain', 'symptom', 'condition', 'infection',
    'virus', 'bacteria', 'health', 'medical', 'treatment', 'remedy',
    'drug', 'medicine', 'immunity', 'vitamin', 'flu'
]

# Words to avoid (spam/ads)
SPAM_KEYWORDS = ['click', 'buy', 'subscribe', 'download', 'call now', 'order', 'limited time', 'sale']


def _clean_line(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _box_stats(box):
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
        "height": max(1, bottom - top),
    }


def _merge_ocr_lines(ocr_result):
    """
    Merge OCR fragments that belong to one poster sentence.
    """
    entries = []
    for item in ocr_result:
        box, text = item[0], _clean_line(item[1])
        if not text:
            continue
        entries.append({"text": text, "box": _box_stats(box)})

    if not entries:
        return []

    entries.sort(key=lambda item: (item["box"]["top"], item["box"]["left"]))
    merged = []

    for entry in entries:
        if not merged:
            merged.append(entry)
            continue

        prev = merged[-1]
        vertical_gap = entry["box"]["top"] - prev["box"]["bottom"]
        center_diff = abs(entry["box"]["center_x"] - prev["box"]["center_x"])
        left_diff = abs(entry["box"]["left"] - prev["box"]["left"])
        height_limit = max(prev["box"]["height"], entry["box"]["height"])

        same_block = (
            vertical_gap <= height_limit * 1.4
            and center_diff <= 250
            and left_diff <= 250
            and len(prev["text"].split()) <= 10
        )

        if same_block:
            prev["text"] = f'{prev["text"]} {entry["text"]}'.strip()
            prev["box"]["left"] = min(prev["box"]["left"], entry["box"]["left"])
            prev["box"]["right"] = max(prev["box"]["right"], entry["box"]["right"])
            prev["box"]["top"] = min(prev["box"]["top"], entry["box"]["top"])
            prev["box"]["bottom"] = max(prev["box"]["bottom"], entry["box"]["bottom"])
            prev["box"]["center_x"] = (prev["box"]["left"] + prev["box"]["right"]) / 2
            prev["box"]["center_y"] = (prev["box"]["top"] + prev["box"]["bottom"]) / 2
            prev["box"]["height"] = max(1, prev["box"]["bottom"] - prev["box"]["top"])
        else:
            merged.append(entry)

    return merged

def extract_claim(texts):
    # Words often found in social media UI, not real claims
    ui_words = {
        "instagram", "post", "like", "likes", "share", "comment",
        "follow", "followers", "following", "reel", "story"
    }

    # Medical claim keywords
    claim_words = {
        "cure", "cures", "prevent", "prevents", "prevention",
        "treat", "treats", "treatment", "heal", "heals",
        "reduce", "reduces", "boost", "boosts"
    }

    comment_words = {"love", "bought", "sufficient", "tips", "amazing", "wow"}

    if texts and isinstance(texts[0], (list, tuple)) and len(texts[0]) >= 2:
        candidates = _merge_ocr_lines(texts)
        image_bottom = max(item["box"]["bottom"] for item in candidates) if candidates else 1
    else:
        candidates = [{"text": _clean_line(text), "box": None} for text in texts]
        image_bottom = 1

    best_line = ""
    best_score = float("-inf")

    for item in candidates:
        line = item["text"]
        if not line:
            continue

        lower_line = line.lower()

        # Split into words
        words = re.findall(r"\b\w+\b", lower_line)
        word_count = len(words)

        # Ignore very short text
        if word_count < 3:
            continue

        # Ignore UI-like text
        if any(ui_word in lower_line for ui_word in ui_words):
            continue

        score = 0

        keyword_matches = sum(word in claim_words for word in words)
        health_matches = sum(word in HEALTH_TERMS for word in words)

        score += keyword_matches * 4
        score += health_matches * 2

        # Prefer meaningful sentence length: 5–12 words
        if 5 <= word_count <= 12:
            score += 3
        elif word_count < 5:
            score -= 2

        if line.isupper() and word_count <= 4:
            score -= 3

        first_word = words[0] if words else ""
        if "_" in first_word or any(char.isdigit() for char in first_word):
            score -= 5
        if any(word in comment_words for word in words):
            score -= 2

        if any(spam_word in lower_line for spam_word in SPAM_KEYWORDS):
            score -= 3

        if item["box"] is not None:
            y_ratio = item["box"]["center_y"] / image_bottom
            if y_ratio > 0.78:
                score -= 6
            elif 0.2 <= y_ratio <= 0.75:
                score += 2

        score += min(word_count, 12) * 0.15

        if score > best_score:
            best_score = score
            best_line = line

    return best_line.rstrip(" :;,-")



def process_image(image_path):
    result = reader.readtext(image_path)
    return extract_claim(result)