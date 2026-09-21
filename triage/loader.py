"""
Loads dm_message_corpus_10k.json and flattens it into one record per
message, in thread order, with a small window of preceding turns attached
as context (order numbers and prior complaints usually live in earlier
turns of the same thread, not in the message being classified).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import CONTEXT_TURNS


@dataclass
class MessageRecord:
    id: str                # f"{thread_idx}:{turn_index}", stable and unique
    thread_idx: int         # position of the thread in the file (0..N-1)
    seed_id: int             # NOTE: not guaranteed unique -- see load_corpus
    turn_index: int
    platform: str
    frm: str                # "customer" or "brand"
    text: str
    customer_name: str = ""
    context: List[str] = field(default_factory=list)  # prior turns, oldest first

    def context_block(self) -> str:
        if not self.context:
            return "(first message in thread)"
        return " | ".join(self.context)


def load_corpus(path: str) -> List[MessageRecord]:
    """
    NOTE on IDs: `seed_id` in the source file is NOT a unique thread key --
    in dm_message_corpus_10k.json, 2,204 threads carry only 2,134 distinct
    seed_id values (70 are reused across different, unrelated threads). An
    id built from seed_id + turn_index alone therefore collides and silently
    drops rows. We use the thread's position in the file (thread_idx)
    instead, which is guaranteed unique, and keep seed_id only as a
    reference column in the output.
    """
    with open(path, encoding="utf-8") as f:
        threads = json.load(f)

    records = []
    for thread_idx, thread in enumerate(threads):
        seed_id = thread["seed_id"]
        platform = thread.get("platform", "")
        customer_name = thread.get("customer_name", "")
        msgs = thread["messages"]
        for idx, m in enumerate(msgs):
            prior = msgs[max(0, idx - CONTEXT_TURNS):idx]
            context = [f"{p['from']}: {p['text']}" for p in prior]
            records.append(MessageRecord(
                id=f"{thread_idx}:{idx}",
                thread_idx=thread_idx,
                seed_id=seed_id,
                turn_index=idx,
                platform=platform,
                frm=m["from"],
                text=m.get("text", "") or "",
                customer_name=customer_name,
                context=context,
            ))
    return records


def split_by_author(records: List[MessageRecord]):
    """Returns (customer_records, brand_records)."""
    customer = [r for r in records if r.frm == "customer"]
    brand = [r for r in records if r.frm == "brand"]
    return customer, brand
