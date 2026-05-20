from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sklearn.utils import shuffle


TARGET_COLUMNS = ["claim", "label"]
VALID_LABELS = {"True", "False", "Misleading"}
MEDIPROOF_TEXT_COLUMNS = ("text", "title", "claim", "content", "headline")

SAMPLE_MEDICAL_ROWS: List[Dict[str, str]] = [
	{"claim": "wearing masks can help reduce the spread of covid-19", "label": "True"},
	{"claim": "covid-19 vaccines are safe and help prevent severe disease", "label": "True"},
	{"claim": "regular hand washing helps reduce the spread of infectious diseases", "label": "True"},
	{"claim": "antibiotics do not treat viral infections such as the common cold", "label": "True"},
	{"claim": "exercise can reduce the risk of cardiovascular disease", "label": "True"},
	{"claim": "insulin is required for people with type 1 diabetes", "label": "True"},
	{"claim": "bleach can cure covid-19", "label": "False"},
	{"claim": "5g spreads covid-19", "label": "False"},
	{"claim": "vaccines contain microchips for tracking people", "label": "False"},
	{"claim": "garlic can permanently cure diabetes", "label": "False"},
	{"claim": "masks cause dangerous oxygen deprivation in healthy people", "label": "False"},
	{"claim": "cancer can be cured completely with herbal remedies alone", "label": "False"},
	{"claim": "vitamin c supports immunity but does not cure flu", "label": "Misleading"},
	{"claim": "a healthy diet supports health but cannot replace all prescribed medication", "label": "Misleading"},
	{"claim": "natural remedies may help mild symptoms but cannot cure serious diseases", "label": "Misleading"},
	{"claim": "supplements can help some people but megadoses are not proven cures", "label": "Misleading"},
	{"claim": "fasting may help weight management but is not safe for everyone", "label": "Misleading"},
	{"claim": "organic food may reduce some exposures but does not eliminate all health risks", "label": "Misleading"},
]

_LINK_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE_PATTERN = re.compile(r"\s+")


def _clean_claim_text(text: Any) -> str:
	if text is None:
		return ""

	cleaned = str(text).strip().lower()
	cleaned = _LINK_PATTERN.sub(" ", cleaned)
	cleaned = _SPACE_PATTERN.sub(" ", cleaned).strip()
	return cleaned


def _canonicalize_label(value: Any) -> Optional[str]:
	if value is None:
		return None

	raw = str(value).strip().lower()
	if not raw:
		return None

	label_map = {
		"true": "True",
		"real": "True",
		"factual": "True",
		"supported": "True",
		"1": "True",
		"yes": "True",
		"false": "False",
		"fake": "False",
		"hoax": "False",
		"refuted": "False",
		"incorrect": "False",
		"0": "False",
		"no": "False",
		"misleading": "Misleading",
		"mixed": "Misleading",
		"partly false": "Misleading",
		"partially false": "Misleading",
		"half true": "Misleading",
		"2": "Misleading",
	}

	if raw in label_map:
		return label_map[raw]

	if "mislead" in raw or "part" in raw or "mixed" in raw:
		return "Misleading"
	if "true" in raw or "real" in raw or "support" in raw:
		return "True"
	if "false" in raw or "fake" in raw or "refut" in raw or "hoax" in raw:
		return "False"

	return None


def _first_present_key(record: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
	lowered = {str(key).lower(): key for key in record.keys()}
	for candidate in candidates:
		if candidate.lower() in lowered:
			return lowered[candidate.lower()]
	return None


def _extract_claim_label_from_record(record: Dict[str, Any], dataset_hint: str) -> Optional[Dict[str, str]]:
	common_claim_keys = [
		"claim", "text", "statement", "content", "title", "headline", "news_text",
		"post", "body", "message",
	]
	common_label_keys = [
		"label", "class", "verdict", "rating", "truth", "truthfulness",
		"target", "y", "is_fake", "fake", "credibility",
	]

	dataset_hint = dataset_hint.lower()

	if "pubhealth" in dataset_hint:
		claim_candidates = ["claim", "statement", "text", "content", "headline", *common_claim_keys]
		label_candidates = ["label", "verdict", "rating", "class", "truthfulness", *common_label_keys]
	elif "coaid" in dataset_hint:
		claim_candidates = ["headline", "title", "content", "text", "post", *common_claim_keys]
		label_candidates = ["label", "class", "is_fake", "fake", "verdict", *common_label_keys]
	elif "fakehealth" in dataset_hint:
		claim_candidates = ["claim", "title", "headline", "news_text", "content", *common_claim_keys]
		label_candidates = ["label", "rating", "verdict", "class", "credibility", *common_label_keys]
	else:
		claim_candidates = common_claim_keys
		label_candidates = common_label_keys

	claim_key = _first_present_key(record, claim_candidates)
	label_key = _first_present_key(record, label_candidates)

	if claim_key is None or label_key is None:
		return None

	claim = _clean_claim_text(record.get(claim_key))
	label = _canonicalize_label(record.get(label_key))

	if not claim or label not in VALID_LABELS:
		return None

	return {"claim": claim, "label": label}


def _records_from_csv(path: Path, dataset_hint: str) -> List[Dict[str, str]]:
	frame = pd.read_csv(path)
	if frame.empty:
		return []

	records: List[Dict[str, str]] = []
	for row in frame.to_dict(orient="records"):
		normalized = _extract_claim_label_from_record(row, dataset_hint=dataset_hint)
		if normalized:
			records.append(normalized)
	return records


def _records_from_json(path: Path, dataset_hint: str) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if isinstance(payload, dict):
		if "data" in payload and isinstance(payload["data"], list):
			raw_records = payload["data"]
		elif "claims" in payload and isinstance(payload["claims"], list):
			raw_records = payload["claims"]
		else:
			raw_records = [payload]
	elif isinstance(payload, list):
		raw_records = payload
	else:
		raw_records = []

	records: List[Dict[str, str]] = []
	for item in raw_records:
		if not isinstance(item, dict):
			continue
		normalized = _extract_claim_label_from_record(item, dataset_hint=dataset_hint)
		if normalized:
			records.append(normalized)
	return records


def _records_from_jsonl(path: Path, dataset_hint: str) -> List[Dict[str, str]]:
	records: List[Dict[str, str]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				item = json.loads(line)
			except json.JSONDecodeError:
				continue
			if not isinstance(item, dict):
				continue
			normalized = _extract_claim_label_from_record(item, dataset_hint=dataset_hint)
			if normalized:
				records.append(normalized)
	return records


def _load_file_records(path: Path) -> List[Dict[str, str]]:
	dataset_hint = path.stem.lower()
	suffix = path.suffix.lower()

	if suffix == ".csv":
		return _records_from_csv(path, dataset_hint=dataset_hint)
	if suffix == ".json":
		return _records_from_json(path, dataset_hint=dataset_hint)
	if suffix in {".jsonl", ".ndjson"}:
		return _records_from_jsonl(path, dataset_hint=dataset_hint)

	raise ValueError(f"Unsupported dataset format: {path}")


def _mediproof_label_from_filename(path: Path) -> Optional[str]:
	stem = path.stem.strip().lower()
	explicit_map = {
		"fake": "False",
		"true": "True",
		"covid_fake": "False",
		"covid_real": "True",
	}

	if stem in explicit_map:
		return explicit_map[stem]

	if "fake" in stem:
		return "False"
	if "true" in stem or "real" in stem:
		return "True"

	return None


def _extract_mediproof_claims(frame: pd.DataFrame) -> pd.Series:
	lowered_columns = {str(column).strip().lower(): column for column in frame.columns}

	for candidate in MEDIPROOF_TEXT_COLUMNS:
		actual_column = lowered_columns.get(candidate)
		if actual_column is None:
			continue
		return frame[actual_column].astype(str).map(_clean_claim_text)

	return pd.Series(dtype="string")


def load_mediproof_dataset(datasets_root: Optional[str] = None) -> pd.DataFrame:
	"""
	Load MediProof datasets from fakehealth and coaid folders.

	Expected base layout:
		datasets/fakehealth/
		datasets/coaid/

	Returns:
		pandas.DataFrame with columns: claim, label.
	"""
	base_path = Path(datasets_root) if datasets_root else (Path(__file__).resolve().parents[1] / "datasets")
	dataset_dirs = [base_path / "fakehealth", base_path / "coaid"]

	rows: List[Dict[str, str]] = []
	for dataset_dir in dataset_dirs:
		if not dataset_dir.exists() or not dataset_dir.is_dir():
			continue

		for csv_path in sorted(dataset_dir.rglob("*.csv")):
			label = _mediproof_label_from_filename(csv_path)
			if label not in {"True", "False"}:
				continue

			try:
				frame = pd.read_csv(csv_path)
			except Exception:
				continue

			if frame.empty:
				continue

			claims = _extract_mediproof_claims(frame)
			if claims.empty:
				continue

			for claim in claims:
				if claim:
					rows.append({"claim": claim, "label": label})

	if not rows:
		return pd.DataFrame(columns=TARGET_COLUMNS)

	result = pd.DataFrame(rows, columns=TARGET_COLUMNS)
	result["claim"] = result["claim"].astype(str).map(_clean_claim_text)
	result = result[result["claim"].str.len() > 0]
	result = result.drop_duplicates().reset_index(drop=True)

	return result[TARGET_COLUMNS]


def build_sample_medical_dataset(random_state: int = 42) -> pd.DataFrame:
	"""
	Return a small built-in fallback dataset when no external dataset is available.
	"""
	frame = pd.DataFrame(SAMPLE_MEDICAL_ROWS, columns=TARGET_COLUMNS)
	frame["claim"] = frame["claim"].astype(str).map(_clean_claim_text)
	frame["label"] = frame["label"].astype(str).map(_canonicalize_label)
	frame = frame.dropna(subset=["claim", "label"])
	frame = frame[frame["claim"].str.len() > 0]
	frame = frame[frame["label"].isin(VALID_LABELS)]
	frame = frame.drop_duplicates(subset=["claim"]).reset_index(drop=True)
	frame = shuffle(frame, random_state=random_state).reset_index(drop=True)
	return frame[TARGET_COLUMNS]


def load_misinformation_datasets(file_paths: Optional[List[str]] = None, random_state: int = 42) -> pd.DataFrame:
	"""
	Load, normalize, and merge medical misinformation datasets.

	Supported source formats: CSV, JSON, JSONL/NDJSON.
	Typical dataset families: FakeHealth, PubHealth, CoAID.

	Args:
		file_paths: Dataset file paths to load and merge.
		random_state: Seed for deterministic row shuffle.

	Returns:
		pandas.DataFrame with columns: claim, label.
	"""
	if not file_paths:
		frame = load_mediproof_dataset()
		if frame.empty:
			frame = build_sample_medical_dataset(random_state=random_state)
		print("Dataset size:", len(frame))
		return frame

	merged_records: List[Dict[str, str]] = []

	for file_path in file_paths:
		path = Path(file_path)
		if not path.exists() or not path.is_file():
			continue
		merged_records.extend(_load_file_records(path))

	if not merged_records:
		frame = build_sample_medical_dataset(random_state=random_state)
		print("Dataset size:", len(frame))
		return frame

	frame = pd.DataFrame(merged_records, columns=TARGET_COLUMNS)

	frame["claim"] = frame["claim"].astype(str).map(_clean_claim_text)
	frame["label"] = frame["label"].astype(str).map(_canonicalize_label)

	frame = frame.dropna(subset=["claim", "label"])
	frame = frame[frame["claim"].str.len() > 0]
	frame = frame[frame["label"].isin(VALID_LABELS)]

	frame = frame.drop_duplicates(subset=["claim"]).reset_index(drop=True)
	frame = shuffle(frame, random_state=random_state).reset_index(drop=True)

	if frame.empty:
		frame = build_sample_medical_dataset(random_state=random_state)

	print("Dataset size:", len(frame))
	return frame[TARGET_COLUMNS]


__all__ = ["load_misinformation_datasets", "load_mediproof_dataset", "build_sample_medical_dataset"]

