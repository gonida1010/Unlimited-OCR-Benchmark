import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == " "
    )
    return text


def normalize_for_word_level(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^\w]", "", text)
    return text


def normalize_for_line_level(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    return text


NORMALIZATION_REGISTRY = {
    "iiit5k": normalize_for_word_level,
    "icdar2015": normalize_for_word_level,
    "iam": normalize_for_line_level,
}
