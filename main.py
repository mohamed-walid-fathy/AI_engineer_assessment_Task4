#!/usr/bin/env python3
"""
Classify the DM corpus into: refund_request, complaint, order_inquiry,
compliment, spam, urgent_escalation (customer messages), plus
brand_message for the brand's own outbound turns.

Three-stage pipeline: a free keyword router, then a free local sentiment
model, then the Gemini API only for what's left. Output is a real .xlsx
file by default (pass a .csv path instead if you'd rather have that).

Usage:
    pip install -r requirements.txt
    set GEMINI_API_KEY=...
    python main.py --input dm_message_corpus_10k.json --output labels.xlsx

    # Cheap smoke test on the first 50 threads before spending real money:
    python main.py --input dm_message_corpus_10k.json --output trial.xlsx --limit 50

    # Exercise the whole pipeline with no network / no API key:
    python main.py --input dm_message_corpus_10k.json --output dryrun.xlsx --dry-run
"""

import argparse
import os
import sys

from triage.config import BATCH_SIZE, MAX_WORKERS, MODEL_PASS1
from triage.pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to dm_message_corpus_10k.json")
    ap.add_argument("--output", default="labels.xlsx",
                    help=".xlsx (default) or .csv -- extension decides the format")
    ap.add_argument("--checkpoint", default="checkpoint.jsonl",
                    help="resumable log of completed API batches")
    ap.add_argument("--model", default=MODEL_PASS1)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N threads (for trial runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip real API calls; exercise the pipeline with a "
                         "fake classifier instead (no key/network needed)")
    ap.add_argument("--refine", action="store_true",
                    help="re-classify every low-confidence row with a "
                         "stronger model. Cheap here because pass 1 "
                         "typically uses a small fraction of budget.")
    ap.add_argument("--skip-sentiment", action="store_true",
                    help="skip the local sentiment model stage (faster "
                         "first run / no ~1.1GB model download; sends more "
                         "messages to the API instead)")
    ap.add_argument("--gpu", action="store_true",
                    help="run the sentiment model on GPU (device 0) "
                         "instead of CPU")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.dry_run and not api_key:
        sys.exit("Set GEMINI_API_KEY (or pass --dry-run to test without one)")

    try:
        run(
            corpus_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            api_key=api_key,
            model=args.model,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            workers=args.workers,
            limit=args.limit,
            refine_low_confidence=args.refine,
            use_sentiment=not args.skip_sentiment,
            sentiment_device=0 if args.gpu else -1,
        )
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume from the "
              "checkpoint.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
