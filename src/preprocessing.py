def clean_text(text: str) -> str:
    """
    Basic text preprocessing for inference.

    We intentionally avoid aggressive NLP cleaning
    because Transformer models work best with natural text.
    """

    if not isinstance(text, str):
        return ""

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text.strip()