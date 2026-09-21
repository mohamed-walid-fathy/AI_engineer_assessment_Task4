"""
Stage 2 of the local pipeline: for messages the pure-keyword router
(router.py) couldn't decide, combine the sentiment model's output with the
same keyword signals to auto-label more cases before anything goes to the
paid API.

Sentiment alone can't produce the six routing categories -- "negative"
could be a complaint or a refund_request, "positive" could be a compliment
or just a friendly question. So this only auto-labels when sentiment
agrees with what the keyword signals already suggest, and only above a
confidence threshold. Anything sentiment is unsure about, or that still
has no category signal at all, still goes to the API.
"""

from dataclasses import dataclass
from typing import List, Tuple

from .loader import MessageRecord
from .router import RouterResult, extract_signals

POSITIVE_THRESHOLD = 0.70
NEGATIVE_THRESHOLD = 0.70


def classify_with_sentiment(rec: MessageRecord, sentiment_label: str,
                             sentiment_score: float) -> RouterResult:
    text = rec.text or ""
    sig = extract_signals(text)

    # Same hard rules as stage 1: never guess on risk language, spam, or
    # conflicting keyword signals -- sentiment doesn't override these.
    if sig.is_risk or sig.is_spam_pattern or sig.has_bare_url:
        return RouterResult(None, "low", "stage1_signal_escalated")
    if sig.has_refund:
        return RouterResult("refund_request", "high", "refund_keyword_stage2")
    if (sig.has_tracking or sig.has_reservation or sig.has_cancel) and \
            sentiment_label != "negative":
        return RouterResult("order_inquiry", "high", "tracking_keyword_stage2")

    # Negative sentiment + no refund ask + confident score -> complaint.
    if (sentiment_label == "negative" and sentiment_score >= NEGATIVE_THRESHOLD
            and not sig.has_refund):
        return RouterResult("complaint", "high",
                             f"sentiment_negative_{sentiment_score:.2f}")

    # Positive sentiment + not a question/decline + confident score ->
    # compliment. A positive-toned question ("thanks, any update?") is
    # still an inquiry, so has_question rules this out.
    if (sentiment_label == "positive" and sentiment_score >= POSITIVE_THRESHOLD
            and not sig.has_question and not sig.is_declining):
        return RouterResult("compliment", "high",
                             f"sentiment_positive_{sentiment_score:.2f}")

    # Neutral sentiment + genuine question + no negative/refund signal ->
    # information request (menu, price, hours, etc).
    if sentiment_label != "negative" and sig.has_question and not sig.has_refund:
        return RouterResult("order_inquiry", "high", "question_stage2")

    return RouterResult(None, "low", "sentiment_inconclusive")


def run_sentiment_stage(records: List[MessageRecord], device: int = -1,
                         batch_size: int = 32):
    """
    Runs the sentiment model over `records` and combines it with keyword
    signals. Returns (decided_rows, still_undecided_records).
    decided_rows are plain dicts matching the pipeline's row format minus
    id/thread_idx/etc, which the caller fills in.
    """
    from .sentiment import score_batch

    texts = [r.text for r in records]
    scores = score_batch(texts, batch_size=batch_size, device=device)

    decided = []
    undecided = []
    for rec, (label, score) in zip(records, scores):
        res = classify_with_sentiment(rec, label, score)
        if res.label:
            decided.append((rec, res))
        else:
            undecided.append(rec)
    return decided, undecided
