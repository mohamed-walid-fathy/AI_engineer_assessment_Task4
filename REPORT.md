# Message Triage — Approach

## The data

`dm_message_corpus_10k.json` is 2,204 DM threads (Instagram/Facebook/
TikTok/X) for a food-delivery brand, 10,000 messages total: 5,551 from
customers, 4,449 from the brand itself. Mixed Arabic (Gulf/Egyptian
dialect) and English, short messages, thread context matters (order
numbers/complaints often sit in an earlier turn).

Two data notes worth knowing:
- `seed_id` is not a unique thread key (70 collisions across 2,204
  threads). The code keys records by file position instead, which is
  unique.
- Brand-authored replies aren't customer intents ("please send your order
  number" isn't a complaint). Every message still gets an output row —
  brand turns get a separate `brand_message` label — but they're not
  force-fit into the six queue categories.

## Pipeline: three free-to-cheap stages

1. **Keyword router** (`router.py`) — bilingual regex/lexicon rules.
   Decides ~72% of customer messages for $0. Only auto-labels when a
   signal is unambiguous; anything conflicting or unclear passes through.
2. **Local sentiment model** (`sentiment.py`, `sentiment_router.py`) — a
   real multilingual model (`cardiffnlp/twitter-xlm-roberta-base-
   sentiment-multilingual`, ~278M params, runs on CPU), not just keywords.
   Combines its sentiment score with the same keyword signals (refund ask,
   tracking question, etc.) to decide more of what stage 1 couldn't —
   catches things like Arabic sentiment or tone that a fixed word list
   misses. Also free, runs locally, no API cost.
3. **Gemini API** (`gemini_client.py`) — only what's left after stages 1
   and 2, sent in batches of 25 with a JSON schema so output stays cheap
   and structured. `--refine` optionally re-checks every low-confidence
   row with a stronger model.

Precedence used throughout: `urgent_escalation > spam > refund_request >
order_inquiry > complaint > compliment`. `urgent_escalation` is narrow
(legal/regulatory threats, chargebacks, safety hazards) — anger alone
isn't enough — and the router never auto-labels it; anything touching
that language always goes to the model.

## Output format

**`labels.xlsx` by default**, two sheets:

- **`labels`** — one row per original message (all 10,000), same as
  before: `id, thread_idx, seed_id, turn_index, platform, from, text,
  label, confidence, source, reason`.
- **`thread_summary`** — one row per conversation (2,204), rolling up each
  thread's customer messages to a single disposition using the same
  precedence order (`urgent_escalation > spam > refund_request >
  order_inquiry > complaint > compliment`), plus `all_labels_seen` so you
  can tell a clean single-topic thread from a messier one. This doesn't
  replace the per-message rows — within one conversation a customer's
  intent genuinely changes turn to turn (a burnt-order complaint, then
  just the order number, then a friendly sign-off are three different,
  correctly different labels), so collapsing to one row per thread would
  lose real information. The rollup is there for anyone who wants "what's
  this conversation about" at a glance without losing the detail.

A real .xlsx rather than CSV: Arabic text in a plain CSV can render as
garbled characters in Excel, because Excel's CSV importer guesses the
wrong encoding on many locales; .xlsx stores text as Unicode directly, no
guessing involved. Pass a `.csv` path instead if you want CSV — it's
written with a UTF-8 BOM, the variant Excel does read correctly.

## Known keyword-router fixes

Testing surfaced two real bugs in the stage-1 lexicon, both fixed:

- **Negation wasn't checked.** "مش حلو" ("not nice") matched the bare
  positive word "حلو" and read as a compliment — the opposite of its
  actual meaning. Any keyword match now checks the ~20 characters before
  it for a negation particle (مش/مو/not/isn't/etc.) and is excluded (not
  flipped to the opposite label — too risky to guess the exact meaning,
  just no longer a false positive) if negated.
- **Overloaded bare words.** "تمام" (means "ok/fine/understood" far more
  often than "great!") and bare "cold"/"بارد" (also a menu category, e.g.
  "cold mezze") were auto-deciding on words too ambiguous to trust alone.
  Both removed as standalone triggers; "cold" now only counts in a
  specific arrival phrase ("وصل بارد", "arrived cold") that actually
  describes food showing up that way.

Net effect: stage-1 coverage dropped from 71.5% to 67.0% of customer
messages — fewer free auto-decisions, but the ones it still makes are
more trustworthy, and everything it no longer auto-decides falls through
to the sentiment model and Gemini instead of being silently wrong.

## Cost

At current pricing (`gemini-3.5-flash-lite`, $0.30/$2.50 per 1M tokens),
the ~1,500 messages left after the two free stages cost roughly
**$0.10–0.15** for the full corpus. `SAFETY_STOP_USD` in `config.py`
hard-aborts the run at $0.50 regardless of estimates, well under a $1 cap,
and the run is resumable from a checkpoint if that ever triggers.

## Running it

```bash
pip install -r requirements.txt
set GEMINI_API_KEY=your_key_here          # (export on Mac/Linux)
python main.py --input dm_message_corpus_10k.json --output labels.xlsx
```

First run downloads the sentiment model (~1.1GB from Hugging Face, one
time, needs internet) before it starts scoring. To skip that and send more
messages to the API instead, add `--skip-sentiment`. To test without
spending anything, add `--dry-run` (fakes the API) or `--limit 50` (first
50 threads only).

## Files

```
main.py                    CLI entry point
requirements.txt
triage/
  config.py                 labels, model names, pricing, budget
  loader.py                 loads JSON, flattens threads into records
  router.py                 stage 1: keyword lexicon
  sentiment.py               local sentiment model wrapper
  sentiment_router.py        stage 2: sentiment + keyword signals
  gemini_client.py          stage 3: Gemini API wrapper
  pipeline.py                orchestrates all three stages -> output
```
