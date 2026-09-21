"""Shared constants. Nothing in here talks to the network."""

LABELS = [
    "refund_request",
    "complaint",
    "order_inquiry",
    "compliment",
    "spam",
    "urgent_escalation",
]

# Label applied to brand/agent-authored turns. Not a customer-intent category;
# see the "brand messages" note in REPORT.md for why these are separated out
# rather than force-fit into the six queue labels.
BRAND_LABEL = "brand_message"

# Gemini model names and USD price per 1M tokens (input, output).
# gemini-2.5-flash-lite was deprecated for new API keys in Sep 2026;
# Google's own 404 error names gemini-3.5-flash-lite as the replacement,
# confirmed at $0.30/$2.50 per 1M tokens by independent pricing trackers
# as of Sep 2026. If you hit a 404 "no longer available" error again in
# the future, Google's error message will name the current replacement --
# put that exact model string here and check its price before running.
MODEL_PASS1 = "gemini-3.5-flash-lite"
MODEL_PASS2 = "gemini-3.5-flash"
PRICING_USD_PER_1M = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (0.75, 4.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),   # kept for reference; deprecated
    "gemini-2.5-flash": (0.30, 2.50),         # kept for reference
}

BATCH_SIZE = 25
MAX_WORKERS = 8
MAX_RETRIES = 5
BUDGET_USD = 1.00
# Real measured cost for this corpus at current gemini-3.5-flash-lite
# pricing is ~$0.10-0.15 (see REPORT.md) -- $0.50 leaves a real margin
# while still self-aborting well before the $1 hard cap.
SAFETY_STOP_USD = 0.50
MAX_CHARS_PER_MESSAGE = 1200
CONTEXT_TURNS = 2  # prior turns included with each message sent to the API

# Errors that will NEVER succeed on retry -- fail fast instead of burning
# 5 retries x up to ~60s of backoff per batch. A deprecated/renamed model,
# a bad key, or a malformed request will not fix itself by waiting.
NON_RETRYABLE_MARKERS = [
    "api key", "permission", "404", "not_found", "no longer available",
    "invalid argument", "400 ", "unauthorized", "forbidden",
]
