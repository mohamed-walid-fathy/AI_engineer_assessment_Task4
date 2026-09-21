# DM Triage — Task 4

Classifies 10,000 social-media DM messages into 7 labels and produces `labels.xlsx`.

## Approach

Three-stage pipeline, ordered by cost (free → cheap):

1. **Keyword router** (`triage/router.py`) — bilingual (Arabic/English) regex and lexicon rules. Decides ~67% of customer messages for $0 with high-precision rules; anything ambiguous, conflicting, or touching high-stakes language passes through.
2. **Local sentiment model** (`triage/sentiment.py`, `triage/sentiment_router.py`) — `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` (~278M params, XLM-RoBERTa, CPU). Combines sentiment scores with the same keyword signals (refund, tracking, question) to label more of what Stage 1 could not. Free, runs locally.
3. **Gemini API** (`triage/gemini_client.py`) — only the remainder (~15%, ~1,500 messages) is sent to `gemini-3.5-flash-lite` in batches of 25 with a JSON-schema response. Optional `--refine` re-checks low-confidence rows with `gemini-3.5-flash`.

Brand-authored turns (4,449 messages) are labeled `brand_message` by rule — they are not customer intents and are not force-fit into the queue categories.

## Key implementation decisions

- **Precedence is consistent everywhere:** `urgent_escalation > spam > refund_request > order_inquiry > complaint > compliment` — same order in the router, sentiment router, Gemini prompt, and thread rollup. `urgent_escalation` is narrow (legal/regulatory, chargeback, safety hazard); anger alone does not qualify. The router **never auto-labels** `urgent_escalation`; any match is escalated to the model.
- **Precision over recall for auto-labeling.** The lexicon only labels when a signal is unambiguous. Bare `cold`/`بارد` and `تمام` are excluded as standalone triggers (menu category and "ok" respectively); negated keywords (`مش حلو` / `not nice`) are excluded, not flipped.
- **Unique row identity by `thread_idx:turn_index`** (file position), not `seed_id` — `seed_id` collides across 70 threads and would silently drop ~250 rows.
- **Output is `.xlsx`, not CSV** — Excel's CSV importer mangles Arabic encoding on many locales; `.xlsx` stores Unicode directly. CSV is still supported with UTF-8 BOM if requested.
- **Checkpointed and resumable.** Each completed API batch is appended to `checkpoint.jsonl` immediately; a crash or budget stop can be resumed with the same command.

## Model / algorithm / methodology

| Component | Role |
|---|---|
| **Keyword router** (regex/lexicon) | High-precision patterns for refund, tracking, reservation, complaint, compliment, spam risk signals |
| **Sentiment model** (`twitter-xlm-roberta-base-sentiment-multilingual`) | 3-way sentiment (positive/neutral/negative) fused with keyword signals at thresholds 0.70 |
| **Gemini** (`gemini-3.5-flash-lite`, `thinking_level="low"`, `temperature=0.0`, JSON schema) | Final classifier for residual messages; batched, schema-constrained to ~10 output tokens/message |

## Data / inputs

**Input:** `dm_message_corpus_10k.json` — 2,204 threads (Instagram, Facebook, TikTok, X,..etc), 10,000 messages (5,551 customer, 4,449 brand). Mixed Arabic (Gulf/Egyptian) and English. Each message carries thread context (2 prior turns included for classification).

**Processing:** `triage/loader.py` flattens threads into `MessageRecord` objects in file order, keyed by `thread_idx:turn_index`. `triage/pipeline.py` orchestrates the three stages and merges results (later stages override earlier only for rows that reach the API).

**Assumption:** Input JSON has `seed_id`, `platform`, `customer_name`, `messages[].from` (`customer`/`brand`) and `messages[].text`.

## Output

**`labels.xlsx`** (919 KB, already included) — two sheets:

- **`labels`** — 10,000 rows, one per original message: `id, thread_idx, seed_id, turn_index, platform, from, text, label, confidence, source, reason`. `source` is `router` / `sentiment` / `api` / `rule` (brand). `confidence` is `high` / `low` / `n/a` (brand).
- **`thread_summary`** — 2,204 rows, one per thread: `thread_idx, seed_id, platform, customer_name, n_customer_messages, primary_label, all_labels_seen, has_low_confidence_turn`. `primary_label` is the highest-precedence label across customer turns.

Label distribution: `order_inquiry` 3,395 / `compliment` 1,230 / `complaint` 732 / `refund_request` 133 / `spam` 35 / `urgent_escalation` 26 / `brand_message` 4,449.

## Important considerations / limitations

- **API cost and dependency.** Requires `GEMINI_API_KEY` (`GOOGLE_API_KEY` also accepted). Full corpus costs ~$0.10–0.15 at `gemini-3.5-flash-lite` pricing; hard budget stop at `$0.50` in `triage/config.py`. Model name is pinned to `gemini-3.5-flash-lite` — if Google deprecates it, the 404 error names the replacement; update `MODEL_PASS1` and `PRICING_USD_PER_1M` in `triage/config.py`.
- **Sentiment model download.** First run with sentiment enabled downloads ~1.1 GB from Hugging Face. Use `--skip-sentiment` to skip it (more messages go to the API). Use `--dry-run` or `--limit N` for no-cost pipeline tests.
- **Confidence flags.** 174 rows are `low` confidence — most likely to be wrong; `--refine` re-checks these with a stronger model.
- **No ground-truth evaluation.** The corpus is unlabeled; accuracy cannot be measured without human adjudication. Lexicon coverage (~72% locally decided) is not a quality metric.
- **Not production-hardened.** No PII redaction, no rate-limit dashboard beyond `CostMeter`, no human-in-the-loop for `urgent_escalation` beyond the label.

See `setup.md` for install and run instructions (Windows PowerShell and macOS/Linux).
