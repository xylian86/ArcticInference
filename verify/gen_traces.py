#!/usr/bin/env python3
"""Generate JSONL trace files for shift parallelism verification tests.

Creates three trace files in verify/traces/:
  - case1_always_sp.jsonl: 1024 requests, long prompt (8192), short gen (128)
  - case2_always_tp.jsonl: 1 request, short prompt (256), long gen (8192)
  - case3_shifting.jsonl:  alternating waves that oscillate batch size around threshold
"""

import json
import os

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def write_trace(filename: str, entries: list[dict]) -> str:
    os.makedirs(TRACES_DIR, exist_ok=True)
    path = os.path.join(TRACES_DIR, filename)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  {path}  ({len(entries)} requests)")
    return path


def gen_case1():
    """Case 1: Always SP -- 1024 concurrent long-prompt, short-gen requests.

    All arrive at t=0. During decode, 1024 tokens/step >> threshold (64),
    so shift parallelism always picks SP mode.
    """
    entries = [
        {"timestamp": 0, "input_length": 8192, "output_length": 128}
        for _ in range(1024)
    ]
    write_trace("case1_always_sp.jsonl", entries)


def gen_case2():
    """Case 2: Always TP -- single request, short prompt, long generation.

    1 token/step in decode <= threshold (512), so shift parallelism
    always picks TP mode (fused full-world tensor parallel).
    """
    entries = [
        {"timestamp": 0, "input_length": 256, "output_length": 8192}
    ]
    write_trace("case2_always_tp.jsonl", entries)


def gen_case3():
    """Case 3: Frequent SP<->TP shifts.

    Many rapid bursts create frequent oscillation around threshold=64:

    Every 500ms a burst of 100 short-gen requests arrives (input=128, output=64).
    During decode the batch has ~100 tokens > threshold 64 --> SP mode.
    Requests finish within a few hundred ms (64 tokens * ~few ms/tok), so the
    batch drains below 64 --> TP mode before the next burst arrives.

    100 rounds x 2 transitions each = ~200 mode transitions.
    Total: 10,000 requests over 50 seconds.
    """
    entries = []
    num_rounds = 100
    burst_interval_ms = 500
    burst_size = 100

    for r in range(num_rounds):
        t = r * burst_interval_ms
        for _ in range(burst_size):
            entries.append({
                "timestamp": t,
                "input_length": 128,
                "output_length": 64,
            })

    write_trace("case3_shifting.jsonl", entries)


def main():
    print("Generating shift parallelism verification traces:")
    gen_case1()
    gen_case2()
    gen_case3()
    print("Done.")


if __name__ == "__main__":
    main()
