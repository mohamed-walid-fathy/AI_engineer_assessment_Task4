"""
Thin wrapper around the Gemini API for the messages the router couldn't
decide on its own. Everything here is about keeping the bill small and
predictable:

  - messages are sent in batches, so the instruction block is paid for
    once per BATCH_SIZE messages instead of once per message
  - output is constrained by a JSON schema (an enum of the six labels),
    so the model can't ramble and output stays ~10 tokens/message
  - thinking is set to the lowest level ("low"). Gemini's 3.x models bill
    thinking tokens at the output rate; the 2.5-era thinking_budget=0
    trick to disable it outright doesn't work on 3.x models (see the
    comment in classify_batch for the exact error this caused)
  - a shared CostMeter tracks real spend from each response's
    usage_metadata and the pipeline aborts before BUDGET_USD is hit
  - each finished batch is appended to a checkpoint file immediately, so
    a crash or rate-limit wall doesn't lose completed work
"""

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from .config import (BATCH_SIZE, MAX_CHARS_PER_MESSAGE, MAX_RETRIES,
                      NON_RETRYABLE_MARKERS, PRICING_USD_PER_1M)
from .loader import MessageRecord

LABELS = [
    "refund_request", "complaint", "order_inquiry",
    "compliment", "spam", "urgent_escalation",
]

SYSTEM_PROMPT = """You are a support-queue triage classifier for a restaurant/food-\
delivery brand's social-media DM inbox (Instagram, Facebook, TikTok, X). Messages \
are a mix of Arabic (Gulf and Egyptian dialects) and English. You assign exactly \
one routing label to each customer message, using the short conversation context \
given with it.

LABELS

urgent_escalation - Requires human attention within hours, not days. Only these \
signals qualify: threat of legal action, a lawyer, regulatory or consumer-protection \
complaint; an explicit bank dispute/chargeback already filed or threatened; a food \
safety or health/injury hazard (allergic reaction, contamination causing illness, \
foreign object in food); explicit threat to go public/to the press. Anger, \
profanity, or repeated follow-up alone do NOT qualify -- those are complaint or \
refund_request.

spam - Not a genuine message from a customer of this business: promotional blasts, \
phishing, crypto/investment pitches, bot gibberish, unrelated cold sales outreach.

refund_request - The customer wants money back, a refund, reimbursement for a \
double charge, or is following up on a refund/reimbursement already asked for.

order_inquiry - A question or information exchange about an order, reservation, \
menu, price, allergen, hours, branch, delivery status, or account -- with no money \
ask and no complaint about something having gone wrong. Includes reservation \
booking/cancellation and providing booking details (name, phone) when the thread \
context shows the brand asked for them.

complaint - Dissatisfaction with food quality, service, staff, or a delivery \
problem (late, wrong, missing item, cold, rude staff) where the customer is not \
asking for money back and it isn't urgent_escalation.

compliment - Praise, thanks, or positive feedback with no unresolved issue and no \
open question. A short "thanks"/"تمام" that is really just closing out a prior \
exchange (not fresh praise) is compliment; a "thanks" attached to a new question or \
request is order_inquiry, not compliment.

PRECEDENCE (apply in order, stop at first match)
1. urgent_escalation   2. spam   3. refund_request   4. order_inquiry
5. complaint   6. compliment

CONFIDENCE
"low" when ambiguous between two labels, when the message is too short/garbled to \
judge even with context, or when intent is unclear. Otherwise "high".

OUTPUT
Return a JSON array, one object per input message, in the given order. Echo back \
the integer index exactly as given."""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "i": {"type": "INTEGER"},
            "label": {"type": "STRING", "enum": LABELS},
            "conf": {"type": "STRING", "enum": ["high", "low"]},
        },
        "required": ["i", "label", "conf"],
    },
}


class CostMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.cost = 0.0
        self.in_tok = 0
        self.out_tok = 0
        self.calls = 0

    def add(self, model: str, prompt_tokens: int, output_tokens: int, thinking_tokens: int = 0):
        pin, pout = PRICING_USD_PER_1M.get(model, (0.10, 0.40))
        o = output_tokens + thinking_tokens
        with self.lock:
            self.in_tok += prompt_tokens
            self.out_tok += o
            self.calls += 1
            self.cost += (prompt_tokens / 1e6) * pin + (o / 1e6) * pout
            return self.cost

    def snapshot(self):
        with self.lock:
            return dict(cost=self.cost, in_tok=self.in_tok,
                        out_tok=self.out_tok, calls=self.calls)

    def report(self) -> str:
        s = self.snapshot()
        return (f"calls={s['calls']}  in={s['in_tok']:,}  out={s['out_tok']:,}  "
                f"cost=${s['cost']:.4f}")


def build_user_prompt(batch: List[MessageRecord]) -> str:
    lines = []
    for local_i, rec in enumerate(batch):
        text = (rec.text or "").strip().replace("\n", " ")[:MAX_CHARS_PER_MESSAGE]
        if not text:
            text = "(empty message)"
        ctx = rec.context_block()
        lines.append(f"[{local_i}] context: {ctx}\n    message: {text}")
    return f"Classify these {len(batch)} messages:\n\n" + "\n".join(lines)


@dataclass
class ApiLabel:
    id: str
    label: str
    conf: str
    missing: bool = False


class GeminiTriageClient:
    """
    dry_run=True skips real network calls and fabricates a plausible
    response so the rest of the pipeline (batching, checkpointing,
    merging, output writing) can be exercised and tested without an API
    key or network access. Real runs need dry_run=False and a working
    GEMINI_API_KEY.
    """

    def __init__(self, api_key: Optional[str], model: str, meter: CostMeter,
                 dry_run: bool = False):
        self.model = model
        self.meter = meter
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            from google import genai
            self._client = genai.Client(api_key=api_key)

    def classify_batch(self, batch: List[MessageRecord]) -> List[ApiLabel]:
        if self.dry_run:
            return self._fake_classify(batch)

        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.0,
            # Gemini's 3.x model family (3.5, 3.1, etc.) replaced the old
            # integer thinking_budget with a string thinking_level
            # (low/medium/high). Passing thinking_budget to a 3.x model
            # returns "400 INVALID_ARGUMENT" with no field name in the
            # message -- that's what this was. "low" keeps cost down
            # without disabling reasoning outright, which 3.x doesn't
            # cleanly support the way 2.5 did with budget=0.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=build_user_prompt(batch),
                    config=cfg,
                )
                if resp.usage_metadata:
                    um = resp.usage_metadata
                    self.meter.add(
                        self.model,
                        getattr(um, "prompt_token_count", 0) or 0,
                        getattr(um, "candidates_token_count", 0) or 0,
                        getattr(um, "thoughts_token_count", 0) or 0,
                    )
                parsed = json.loads(resp.text)
                return self._merge(batch, parsed)
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if any(marker in msg for marker in NON_RETRYABLE_MARKERS):
                    raise RuntimeError(
                        f"non-retryable error, stopping immediately: {e}") from e
                time.sleep(min(2 ** attempt + random.random(), 30))
        raise RuntimeError(f"batch failed after {MAX_RETRIES} attempts: {last_err}")

    def _merge(self, batch, parsed) -> List[ApiLabel]:
        by_index = {}
        if isinstance(parsed, list):
            for item in parsed:
                try:
                    by_index[int(item["i"])] = item
                except (KeyError, TypeError, ValueError):
                    continue
        out = []
        for local_i, rec in enumerate(batch):
            item = by_index.get(local_i)
            if item and item.get("label") in LABELS:
                out.append(ApiLabel(rec.id, item["label"], item.get("conf", "low")))
            else:
                # Dropped/mangled index -- don't silently guess, flag it.
                out.append(ApiLabel(rec.id, "order_inquiry", "low", missing=True))
        return out

    def _fake_classify(self, batch: List[MessageRecord]) -> List[ApiLabel]:
        """Deterministic stand-in used only for --dry-run pipeline testing."""
        approx_in = sum(len(build_user_prompt(batch)) for _ in [0]) // 4
        approx_out = len(batch) * 10
        self.meter.add(self.model, approx_in, approx_out, 0)
        out = []
        for rec in batch:
            t = (rec.text or "").lower()
            if "refund" in t or "استرجاع" in t or "استرداد" in t:
                lab = "refund_request"
            elif any(k in t for k in ["سيء", "بارد", "rude", "disappointed"]):
                lab = "complaint"
            elif "؟" in t or "?" in t:
                lab = "order_inquiry"
            else:
                lab = "order_inquiry"
            out.append(ApiLabel(rec.id, lab, "low"))
        return out
