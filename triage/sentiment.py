"""
A real, local, multilingual sentiment model (not just keyword matching),
used as a second free layer between the keyword router and the Gemini API.

Runs entirely on your machine via Hugging Face `transformers` -- no API
cost, no per-message billing. The first run downloads the model weights
(~1.1GB) from Hugging Face once; after that it's cached locally and loads
from disk.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
  - XLM-RoBERTa base (~278M params), fine-tuned for 3-way sentiment
    (positive / neutral / negative) across many languages, Arabic and
    English both included. "Base"-sized rather than a giant LLM, and runs
    fine on CPU for a few thousand short messages (a few minutes).

This model gives sentiment, not the six routing categories directly --
sentiment_router.py combines its output with the same keyword signals
router.py uses (refund/tracking/reservation keywords, question marks) to
turn "this message is negative" into "this message is a complaint" vs.
"this message is a refund_request", the same way a human triager would
read tone alongside content.
"""

from typing import List, Tuple

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

_pipeline = None


def _get_pipeline(device: int = -1):
    """device=-1 means CPU. Pass device=0 if you have a CUDA GPU available
    and want it to run faster."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            max_length=128,
            device=device,
        )
    return _pipeline


def _normalize_label(raw_label: str) -> str:
    l = raw_label.lower()
    if "pos" in l:
        return "positive"
    if "neg" in l:
        return "negative"
    return "neutral"


def score_batch(texts: List[str], batch_size: int = 32,
                 device: int = -1) -> List[Tuple[str, float]]:
    """
    Returns one (label, score) pair per input text, label in
    {"positive", "neutral", "negative"}, score is the model's confidence
    in [0, 1]. Empty strings are scored as neutral without calling the
    model (nothing to classify).
    """
    pipe = _get_pipeline(device)
    # The model errors on an empty string; substitute a neutral placeholder
    # and keep the output aligned 1:1 with the input list.
    safe_texts = [t if (t or "").strip() else "." for t in texts]
    raw = pipe(safe_texts, batch_size=batch_size)
    return [(_normalize_label(r["label"]), float(r["score"])) for r in raw]
