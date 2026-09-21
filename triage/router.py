"""
A cheap, offline, bilingual (Arabic/English) keyword + emoji lexicon router.

This is the "simple model that does most of the work" layer: it looks at
one message plus a little thread context and either (a) decides the label
itself for free, or (b) says "send this to the API," which happens whenever
signals conflict, nothing matches, or the message touches a high-stakes
category where a wrong local guess is expensive.

Design principle: precision over recall for auto-labeling. The router
should only ever claim a label it is genuinely confident about; anything
it isn't sure of falls through to the API rather than getting force-fit
into a category. spam and urgent_escalation in particular are cheap to get
wrong (a missed legal threat is a real cost) and rare in this corpus, so
the router never auto-labels urgent_escalation, and only auto-labels spam
on an unambiguous promotional/phishing pattern.

This is deliberately simple regex/keyword matching rather than a trained
sentiment model. A trained model (e.g. a small Arabic sentiment classifier)
would likely raise coverage further, but needs labeled training data we
don't have here, and regex is auditable -- you can read every rule below
and know exactly why a message got its label. See REPORT.md for coverage
numbers measured against this corpus.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .loader import MessageRecord

ORDER_NUM_RE = re.compile(r"#\s?\d{3,}|\b\d{3,}\s?(?:رقم|order)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

# High-stakes signals: if any of these appear, NEVER auto-label locally.
# Always send to the API, regardless of what else matches.
RISK_TRIGGERS = [
    "lawyer", "legal action", "sue ", "sue.", "attorney", "court",
    "consumer protection", "regulatory", "press", "media", "publicly",
    "chargeback", "dispute the charge", "reported to my bank",
    "محامي", "قضية", "هيئة حماية المستهلك", "شكوى رسمية", "إعلام",
    "سأنشر", "دعوى",
]

SPAM_PATTERNS = [
    "win a prize", "click here", "free money", "investment opportunity",
    "crypto", "% off today only", "اربح جائزة", "استثمر الآن", "اضغط هنا",
]

REFUND_KEYWORDS = [
    "refund", "money back", "reimburse",
    "استرجاع", "استرداد", "ارجعوا فلوس", "يرجعوا فلوس", "رجعوا فلوس",
    "استرد", "المبلغ يرجع", "اريد فلوسي", "عايز فلوسي", "عايزة فلوسي",
]

CANCEL_KEYWORDS = [
    "cancel", "الغاء", "إلغاء", "ألغي", "ألغى",
]

TRACKING_KEYWORDS = [
    "where is my order", "any update", "still waiting", "eta", "tracking",
    "wheres my", "where's my", "checking on", "following up",
    "وين طلبي", "فين الطلب", "متى يوصل", "هل وصل", "متابعة الطلب",
    "أي جديد", "لسه", "استفسار",
]

RESERVATION_KEYWORDS = [
    "reservation", "table for", "book a table",
    "حجز", "طاولة", "ترابيزة",
]

NEGATIVE_KEYWORDS = [
    "wrong order", "damaged", "broken", "rude", "disgusting",
    "not happy", "disappointed", "unacceptable", "terrible", "awful",
    "missing", "spilled", "worst",
    "سيء", "سيئ", "زعلان", "متضايق", "خربان", "ناقص",
    "قلة احترام", "غير طازة", "مو راضي", "مش راضي", "وسخ",
    "مقرف", "أسوأ",
    # "late"/"متأخر", "unfortunately"/"للأسف", and bare "cold"/"بارد" are
    # deliberately NOT here as standalone words: all three show up
    # constantly in neutral or self-referential contexts in this corpus
    # ("we're running late", "cold mezze" as a menu category, "cold
    # drinks"), not just in complaints about the business. Only count
    # these as a complaint signal in a specific compound phrase that
    # actually describes food/delivery arriving that way.
    "order is late", "delivery is late", "order was late",
    "طلب متأخر", "الطلب متأخر", "الأوردر متأخر", "التوصيل متأخر",
    "وصل متأخر", "اتأخر الطلب",
    "arrived cold", "came cold", "was cold", "tasted cold",
    "وصل بارد", "وصلت بارد", "وصلتني بارد", "جاء بارد", "اكل بارد",
    "الاكل بارد", "الطعام بارد",
]

POSITIVE_KEYWORDS = [
    "thank you", "thanks", "thx", "amazing", "great", "love it", "excellent",
    "perfect", "delicious", "best", "awesome",
    "شكرا", "شكراً", "تسلم", "تسلمون", "يعطيكم العافية", "الحمد لله",
    "ممتاز", "رائع", "حلو", "لذيذ", "احسن", "أحسن",
    # "تمام" was deliberately removed: it means "ok/fine/understood" far
    # more often in this corpus than "great!" as enthusiastic praise, so
    # on its own it isn't a reliable positive signal -- e.g. "خلاص تمام،
    # هطلب كفتة" ("ok fine, I'll order kofta") is placing an order, not
    # complimenting anything. shukran/mumtaz/etc are much less ambiguous.
]

NEGATIVE_EMOJI = {"😡", "😠", "😤", "😭", "👎", "🤮", "😞", "😢"}
POSITIVE_EMOJI = {"😍", "❤️", "🎉", "😊", "👍", "🥰", "💕"}

QUESTION_MARK_RE = re.compile(r"[?؟]")

# Negation particles that flip the meaning of a word right after them.
# "مش حلو" ("not nice") matching bare "حلو" ("nice") without checking for
# this is a real bug found in testing: it read as a compliment. Any
# positive/negative keyword hit preceded within NEGATION_WINDOW characters
# by one of these is treated as negated -- not flipped to the opposite
# category (too risky to guess the exact meaning), just excluded from
# being a confident signal at all.
NEGATION_MARKERS = [
    "مش", "مو ", "مب ", "مافي", "ماكو", "مافيش", "لا ", "ابدا", "أبدا",
    "not ", "isn't", "wasn't", "didn't", "doesn't", "don't", "no ",
]
NEGATION_WINDOW = 20

# Bare booking-flow details ("Ahmed Mansour, +971 50 888 1234") that show up
# as a reply after the brand asks for a name/phone to confirm a reservation.
# Only used as a signal when combined with short length and no sentiment
# words, so a phone number quoted inside a longer complaint doesn't trip it.
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


@dataclass
class RouterResult:
    label: Optional[str]     # None means "send to API"
    confidence: str          # "high" | "low" (only meaningful when label is set)
    reason: str              # short audit trail, kept in the output for review


@dataclass
class Signals:
    """Keyword/pattern hits for one message. Shared between the pure-keyword
    stage (classify_locally) and the sentiment-augmented stage
    (sentiment_router.py) so the two lexicons never drift apart."""
    is_risk: bool
    is_spam_pattern: bool
    has_bare_url: bool
    has_refund: bool
    has_negative_keyword: bool
    has_positive_keyword: bool
    is_declining: bool
    has_tracking: bool
    has_reservation: bool
    has_cancel: bool
    has_question: bool
    has_booking_detail: bool


def _has_any(text_lower: str, terms) -> bool:
    return any(t in text_lower for t in terms)


def _is_negated_at(text_lower: str, idx: int) -> bool:
    start = max(0, idx - NEGATION_WINDOW)
    preceding = text_lower[start:idx]
    return any(neg in preceding for neg in NEGATION_MARKERS)


def _has_any_unnegated(text_lower: str, terms) -> bool:
    """Like _has_any, but a match preceded closely by a negation particle
    ('مش', 'not', etc.) doesn't count -- see NEGATION_MARKERS above."""
    for t in terms:
        idx = text_lower.find(t)
        if idx != -1 and not _is_negated_at(text_lower, idx):
            return True
    return False


def _emoji_hits(text: str, emoji_set) -> bool:
    return any(ch in text for ch in emoji_set)


def extract_signals(text: str) -> Signals:
    tl = (text or "").lower()
    is_declining = "لا شكر" in tl or "no thanks" in tl or "لأ شكر" in tl
    has_url = bool(URL_RE.search(text))
    has_photo_context = any(w in tl for w in ("صورة", "photo", "screenshot", "image"))
    return Signals(
        is_risk=_has_any(tl, RISK_TRIGGERS),
        is_spam_pattern=_has_any(tl, SPAM_PATTERNS),
        has_bare_url=has_url and not (ORDER_NUM_RE.search(text) or has_photo_context),
        has_refund=_has_any(tl, REFUND_KEYWORDS),
        has_negative_keyword=_has_any_unnegated(tl, NEGATIVE_KEYWORDS) or _emoji_hits(text, NEGATIVE_EMOJI),
        has_positive_keyword=(_has_any_unnegated(tl, POSITIVE_KEYWORDS) or _emoji_hits(text, POSITIVE_EMOJI))
        and not is_declining,
        is_declining=is_declining,
        has_tracking=_has_any(tl, TRACKING_KEYWORDS) or bool(ORDER_NUM_RE.search(text)),
        has_reservation=_has_any(tl, RESERVATION_KEYWORDS),
        has_cancel=_has_any(tl, CANCEL_KEYWORDS),
        has_question=bool(QUESTION_MARK_RE.search(text)),
        has_booking_detail=bool(PHONE_RE.search(text) or EMAIL_RE.search(text)),
    )


def classify_locally(rec: MessageRecord) -> RouterResult:
    text = rec.text or ""
    if not text.strip():
        return RouterResult(None, "low", "empty_message")

    sig = extract_signals(text)

    # 1. High-stakes language always goes to the model. No exceptions.
    if sig.is_risk:
        return RouterResult(None, "low", "risk_trigger_escalated_to_api")

    # 2. Unambiguous spam pattern -- safe to auto-label, rare in practice.
    if sig.is_spam_pattern:
        return RouterResult("spam", "high", "spam_keyword")
    if sig.has_bare_url:
        # A bare link with no order/photo context is treated as ambiguous,
        # not auto-spam -- could be a receipt link. Let the model decide.
        return RouterResult(None, "low", "url_no_context_escalated")

    # 3. Conflicting signals -> don't guess.
    if sig.has_refund and sig.has_positive_keyword:
        return RouterResult(None, "low", "conflicting_refund_and_positive")
    if sig.has_negative_keyword and sig.has_positive_keyword:
        return RouterResult(None, "low", "conflicting_negative_and_positive")

    # 4. Refund ask wins over everything below it (matches the precedence
    #    rule used in the API prompt, so router and model stay consistent).
    if sig.has_refund:
        return RouterResult("refund_request", "high", "refund_keyword")

    # 5. Order number / tracking / reservation-change with no negative
    #    sentiment reads as a plain status question.
    if (sig.has_tracking or sig.has_reservation or sig.has_cancel) and not sig.has_negative_keyword:
        return RouterResult("order_inquiry", "high", "tracking_or_reservation_keyword")

    # 6. Clear negative sentiment, no refund ask -> complaint.
    if sig.has_negative_keyword and not sig.has_refund:
        return RouterResult("complaint", "high", "negative_keyword_or_emoji")

    # 7. Short, purely positive, no question -> compliment.
    if (sig.has_positive_keyword and not sig.has_negative_keyword
            and not sig.has_question and len(text) < 200):
        return RouterResult("compliment", "high", "positive_keyword_or_emoji")

    # 8. Any genuine question with no negative/refund signal is a plain
    #    information request (menu, allergens, pricing, availability,
    #    reservation changes).
    if sig.has_question and not sig.has_negative_keyword and not sig.has_refund:
        return RouterResult("order_inquiry", "high", "question_no_negative_signal")

    # 9. A short reply that's just booking details (name + phone, or an
    #    email) with no sentiment words -- almost always a reservation
    #    confirmation reply in this corpus.
    if (not sig.has_negative_keyword and not sig.has_positive_keyword
            and not sig.has_question and len(text) < 80 and sig.has_booking_detail):
        return RouterResult("order_inquiry", "low", "booking_detail_pattern")

    # 10. Nothing matched confidently -- the keyword lexicon has no opinion.
    #     The sentiment model (sentiment_router.py) gets the next attempt.
    return RouterResult(None, "low", "no_confident_signal")
