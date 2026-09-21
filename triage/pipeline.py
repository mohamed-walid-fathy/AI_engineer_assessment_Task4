"""
Ties the pieces together: load corpus -> route each customer message
locally or to the API -> merge everything -> write one CSV row per
original message (all 10,000, including brand-authored turns).
"""

import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from .config import (BATCH_SIZE, BRAND_LABEL, LABELS, MAX_WORKERS,
                      MODEL_PASS1, MODEL_PASS2, SAFETY_STOP_USD)
from .gemini_client import CostMeter, GeminiTriageClient
from .loader import MessageRecord, load_corpus, split_by_author
from .router import classify_locally
from .sentiment_router import run_sentiment_stage


def route_all(customer_records: List[MessageRecord]):
    """Returns (decided_rows, needs_api_records)."""
    decided = []
    needs_api = []
    for rec in customer_records:
        res = classify_locally(rec)
        if res.label:
            decided.append(dict(
                id=rec.id, thread_idx=rec.thread_idx, seed_id=rec.seed_id,
                turn_index=rec.turn_index,
                platform=rec.platform, frm=rec.frm,
                label=res.label, confidence=res.confidence,
                source="router", reason=res.reason,
            ))
        else:
            needs_api.append(rec)
    return decided, needs_api


def brand_rows(brand_records: List[MessageRecord]):
    return [dict(
        id=rec.id, thread_idx=rec.thread_idx, seed_id=rec.seed_id,
        turn_index=rec.turn_index,
        platform=rec.platform, frm=rec.frm,
        label=BRAND_LABEL, confidence="n/a", source="rule",
        reason="brand_authored_not_a_customer_intent",
    ) for rec in brand_records]


def load_checkpoint(path: str):
    done_batches, rows = set(), []
    if not Path(path).exists():
        return done_batches, rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            done_batches.add(r["batch"])
    return done_batches, rows


def run_api_pass(needs_api: List[MessageRecord], client: GeminiTriageClient,
                  checkpoint_path: str, batch_size: int, workers: int,
                  meter: CostMeter) -> List[dict]:
    batches = [(i // batch_size, needs_api[i:i + batch_size])
               for i in range(0, len(needs_api), batch_size)]
    done_batches, rows = load_checkpoint(checkpoint_path)
    todo = [(bid, b) for bid, b in batches if bid not in done_batches]
    print(f"  API pass: {len(batches)} batches, {len(done_batches)} already "
          f"done, {len(todo)} to go", file=sys.stderr)

    ck = open(checkpoint_path, "a", encoding="utf-8")
    ck_lock = threading.Lock()
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(client.classify_batch, batch): (bid, batch)
                       for bid, batch in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                bid, batch = futures[fut]
                try:
                    results = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  batch {bid} FAILED: {e}", file=sys.stderr)
                    continue

                batch_by_id = {rec.id: rec for rec in batch}
                out_rows = []
                for r in results:
                    rec = batch_by_id[r.id]
                    out_rows.append(dict(
                        id=rec.id, thread_idx=rec.thread_idx, seed_id=rec.seed_id,
                        turn_index=rec.turn_index,
                        platform=rec.platform, frm=rec.frm,
                        label=r.label, confidence=r.conf, source="api",
                        reason="api_missing_index" if r.missing else "api_classified",
                        batch=bid,
                    ))
                rows.extend(out_rows)
                with ck_lock:
                    for r in out_rows:
                        ck.write(json.dumps(r) + "\n")
                    ck.flush()

                if n % 10 == 0 or n == len(futures):
                    rate = n / max(time.time() - t0, 1e-6)
                    eta = (len(futures) - n) / rate / 60 if rate else 0
                    print(f"  {n}/{len(futures)} batches  {meter.report()}  "
                          f"eta {eta:.1f}m", file=sys.stderr)

                if meter.snapshot()["cost"] >= SAFETY_STOP_USD:
                    print(f"\n!! budget stop at ${SAFETY_STOP_USD}. Re-run to "
                          f"resume from checkpoint.", file=sys.stderr)
                    for f in futures:
                        f.cancel()
                    break
    except KeyboardInterrupt:
        print("\nInterrupted -- cancelling pending batches (this can take a "
              "few seconds for in-flight requests to unwind). Completed "
              "batches are already saved in the checkpoint; re-run the same "
              "command to resume.", file=sys.stderr)
        ex.shutdown(wait=False, cancel_futures=True)
        ck.close()
        raise
    finally:
        if not ck.closed:
            ck.close()

    return rows


def _row_for(m: MessageRecord, r: dict):
    return [m.id, m.thread_idx, m.seed_id, m.turn_index, m.platform, m.frm,
            m.text,
            r["label"] if r else "",
            r["confidence"] if r else "",
            r["source"] if r else "",
            r.get("reason", "") if r else ""]


HEADERS = ["id", "thread_idx", "seed_id", "turn_index", "platform",
           "from", "text", "label", "confidence", "source", "reason"]

# Precedence for rolling a thread's several customer-message labels up to
# one "what team should own this conversation" disposition -- the same
# order used to resolve a single ambiguous message, applied across turns
# instead of within one. A thread with any refund_request turn is a
# refund conversation even if a later turn is a friendly "thanks, bye".
THREAD_PRECEDENCE = {label: i for i, label in enumerate(LABELS)}


def build_thread_summary(by_id: Dict[str, dict], messages: List[MessageRecord]):
    """One row per thread: customer_name, platform, how many customer
    messages, the highest-precedence label seen, and whether the thread's
    labels actually agreed (a thread that's all order_inquiry is a clean
    status check; one that's order_inquiry then refund_request then
    complaint is a messier conversation worth a human glancing at)."""
    by_thread: Dict[int, dict] = {}
    for m in messages:
        if m.frm != "customer":
            continue
        r = by_id.get(m.id)
        if not r or r["label"] not in THREAD_PRECEDENCE:
            continue
        t = by_thread.setdefault(m.thread_idx, dict(
            thread_idx=m.thread_idx, seed_id=m.seed_id,
            platform=m.platform, customer_name=m.customer_name,
            labels=[], any_low_confidence=False,
        ))
        t["labels"].append(r["label"])
        if r["confidence"] == "low":
            t["any_low_confidence"] = True

    rows = []
    for t in sorted(by_thread.values(), key=lambda x: x["thread_idx"]):
        labels = t["labels"]
        primary = min(labels, key=lambda l: THREAD_PRECEDENCE[l])
        distinct = sorted(set(labels), key=lambda l: THREAD_PRECEDENCE[l])
        rows.append([
            t["thread_idx"], t["seed_id"], t["platform"], t["customer_name"],
            len(labels), primary, ", ".join(distinct),
            "yes" if t["any_low_confidence"] else "",
        ])
    return rows


def write_output(all_rows: List[dict], messages: List[MessageRecord], out_path: str):
    """
    Writes .xlsx if out_path ends with .xlsx (default, and what fixes the
    "weird characters" problem some people hit opening a plain CSV of
    Arabic text in Excel -- Excel's CSV import guesses the wrong encoding
    on many locales; a real .xlsx file has no such ambiguity because the
    text is stored as proper Unicode inside the file format itself, not
    guessed from bytes). Writes a UTF-8-with-BOM CSV for any other
    extension, which is the CSV variant Excel reliably reads as Unicode.
    """
    by_id: Dict[str, dict] = {}
    for r in all_rows:
        by_id[r["id"]] = r  # last write wins (later pass overrides earlier)

    if out_path.lower().endswith(".xlsx"):
        _write_xlsx(by_id, messages, out_path)
    else:
        _write_csv(by_id, messages, out_path)
    return by_id


def _write_xlsx(by_id: Dict[str, dict], messages: List[MessageRecord], out_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(HEADERS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for m in messages:
        ws.append(_row_for(m, by_id.get(m.id)))

    ws.freeze_panes = "A2"
    widths = [16, 10, 10, 10, 10, 9, 60, 16, 10, 8, 26]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws2 = wb.create_sheet("thread_summary")
    thread_headers = ["thread_idx", "seed_id", "platform", "customer_name",
                       "n_customer_messages", "primary_label",
                       "all_labels_seen", "has_low_confidence_turn"]
    ws2.append(thread_headers)
    for cell in ws2[1]:
        cell.font = Font(bold=True)
    for row in build_thread_summary(by_id, messages):
        ws2.append(row)
    ws2.freeze_panes = "A2"
    thread_widths = [11, 10, 10, 20, 12, 16, 40, 12]
    for i, w in enumerate(thread_widths, start=1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    wb.save(out_path)


def _write_csv(by_id: Dict[str, dict], messages: List[MessageRecord], out_path: str):
    # utf-8-sig (UTF-8 with a BOM) is the encoding Excel's CSV importer
    # reliably auto-detects as Unicode on Windows; plain utf-8 without a
    # BOM is what produces mojibake for Arabic when double-clicked open.
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for m in messages:
            w.writerow(_row_for(m, by_id.get(m.id)))


def run(corpus_path: str, output_path: str, checkpoint_path: str,
        api_key: str, model: str = MODEL_PASS1, dry_run: bool = False,
        batch_size: int = BATCH_SIZE, workers: int = MAX_WORKERS,
        limit: int = None, refine_low_confidence: bool = False,
        use_sentiment: bool = True, sentiment_device: int = -1):
    messages = load_corpus(corpus_path)
    if limit:
        # Keep whole threads together rather than cutting mid-thread.
        # Uses thread_idx (guaranteed unique) rather than seed_id, which
        # is NOT unique in this corpus -- see loader.load_corpus.
        keep = set(range(min(limit, messages[-1].thread_idx + 1)))
        messages = [m for m in messages if m.thread_idx in keep]

    customer, brand = split_by_author(messages)
    print(f"loaded {len(messages):,} messages ({len(customer):,} customer, "
          f"{len(brand):,} brand)", file=sys.stderr)

    router_rows, needs_stage2 = route_all(customer)
    print(f"stage 1 (keyword router) decided {len(router_rows):,}/{len(customer):,} "
          f"locally ({100*len(router_rows)/max(len(customer),1):.1f}%)",
          file=sys.stderr)

    sentiment_rows = []
    needs_api = needs_stage2
    if use_sentiment and needs_stage2:
        print(f"stage 2 (sentiment model) scoring {len(needs_stage2):,} "
              f"remaining messages...", file=sys.stderr)
        decided_pairs, needs_api = run_sentiment_stage(
            needs_stage2, device=sentiment_device)
        for rec, res in decided_pairs:
            sentiment_rows.append(dict(
                id=rec.id, thread_idx=rec.thread_idx, seed_id=rec.seed_id,
                turn_index=rec.turn_index, platform=rec.platform, frm=rec.frm,
                label=res.label, confidence=res.confidence,
                source="sentiment", reason=res.reason,
            ))
        print(f"stage 2 decided {len(sentiment_rows):,}/{len(needs_stage2):,} "
              f"more locally", file=sys.stderr)

    print(f"{len(needs_api):,} messages sent to the API "
          f"({100*len(needs_api)/max(len(customer),1):.1f}% of customer "
          f"messages)", file=sys.stderr)

    meter = CostMeter()
    client = GeminiTriageClient(api_key, model, meter, dry_run=dry_run)

    if needs_api and not dry_run:
        print(f"preflight check against {model}...", file=sys.stderr)
        try:
            client.classify_batch(needs_api[:1])
            print("preflight ok", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"\nPREFLIGHT FAILED -- stopping before spending on the "
                  f"remaining {len(needs_api)-1:,} messages.\n{e}\n",
                  file=sys.stderr)
            print("Common cause: the model name is wrong or deprecated. "
                  "Check the error above for a replacement model name from "
                  "Google, update MODEL_PASS1 in triage/config.py (and its "
                  "price in PRICING_USD_PER_1M), then re-run.", file=sys.stderr)
            raise

    api_rows = run_api_pass(needs_api, client, checkpoint_path, batch_size,
                             workers, meter)

    all_rows = router_rows + sentiment_rows + brand_rows(brand) + api_rows

    if refine_low_confidence:
        # Budget for pass 1 is typically ~$0.03-0.05 of the $1 cap, so
        # there's room to re-check every "low" confidence row (from either
        # the router or the API) with a stronger model. This targets the
        # rows most likely to be wrong rather than spending the remaining
        # budget uniformly.
        rec_by_id = {m.id: m for m in messages}
        low_ids = [r["id"] for r in all_rows if r.get("confidence") == "low"]
        low_records = [rec_by_id[i] for i in low_ids if rec_by_id[i].frm == "customer"]
        print(f"\nrefine pass: {len(low_records):,} low-confidence customer "
              f"rows -> {MODEL_PASS2}", file=sys.stderr)
        if low_records:
            refine_client = GeminiTriageClient(api_key, MODEL_PASS2, meter,
                                                dry_run=dry_run)
            refine_ckpt = checkpoint_path.replace(".jsonl", ".refine.jsonl")
            refine_rows = run_api_pass(low_records, refine_client, refine_ckpt,
                                        batch_size, workers, meter)
            for r in refine_rows:
                r["source"] = "api_refine"
            all_rows = all_rows + refine_rows  # appended rows win in write_output

    by_id = write_output(all_rows, messages, output_path)

    counts = {}
    low = 0
    for r in by_id.values():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
        if r["confidence"] == "low":
            low += 1

    print(f"\nDONE -> {output_path}", file=sys.stderr)
    print(f"stage1 router: {len(router_rows):,}  stage2 sentiment: "
          f"{len(sentiment_rows):,}  api: {len(api_rows):,}  "
          f"brand: {len(brand):,}  total labeled: {len(by_id):,}/{len(messages):,}",
          file=sys.stderr)
    print(meter.report(), file=sys.stderr)
    print(f"low-confidence rows: {low:,}", file=sys.stderr)
    for lab, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<20} {c:>6,}  {100*c/len(by_id):5.1f}%", file=sys.stderr)

    return dict(counts=counts, low_confidence=low, meter=meter.snapshot(),
                total=len(by_id))
