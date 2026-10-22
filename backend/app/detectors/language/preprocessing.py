import re
import unicodedata

word_tokenize = None
ViTokenizer = None

try:
    from underthesea import word_tokenize
except ImportError:
    try:
        from pyvi import ViTokenizer
    except ImportError:
        pass


def preprocess_text(text: str) -> dict[str, object]:
    normalized = unicodedata.normalize("NFC", text)
    normalized = "".join(
        character for character in normalized
        if character in "\n\t" or unicodedata.category(character)[0] != "C"
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lowered = normalized.casefold()

    if word_tokenize is not None:
        tokens = word_tokenize(lowered, format="text").split()
    elif ViTokenizer is not None:
        tokens = ViTokenizer.tokenize(lowered).split()
    else:
        tokens = lowered.split()

    return {
        "text": normalized,
        "lowered": lowered,
        "tokens": tokens,
        "urls": re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", lowered),
        "phone_numbers": re.findall(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)", lowered),
        "bank_accounts": re.findall(r"(?<!\d)\d(?:[ .-]?\d){7,18}(?!\d)", lowered),
    }


def vectorize_text(text: str) -> str:
    return " ".join(preprocess_text(text)["tokens"])
