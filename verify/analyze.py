#!/usr/bin/env python3
"""Analyze shift parallelism verification test results.

Parses server logs and benchmark output files produced by run_all.sh.

Server logs contain per-iteration lines (from --enable-logging-iteration-details):
    Iteration(N): X context requests, Y context tokens, Z generation requests,
    W generation tokens, iteration elapsed time: T.TTT ms
and periodic shift counts:
    [shift-count] SP_steps=X TP_steps=Y

Usage:
    python analyze.py                  # analyze all cases
    python analyze.py --case case1     # analyze one case
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")


@dataclass
class IterationEntry:
    iteration: int
    timestamp: float
    prefill: int
    decode: int
    mode: str


@dataclass
class BenchResult:
    num_requests: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    e2el_ms: list[float] = field(default_factory=list)
    input_lens: list[int] = field(default_factory=list)
    output_lens: list[int] = field(default_factory=list)


@dataclass
class ServerStats:
    iterations: list[IterationEntry] = field(default_factory=list)
    sp_steps: int = 0
    tp_steps: int = 0
    shift_transitions: int = 0


VLLM_ITERATION_RE = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)
SHIFT_COUNT_RE = re.compile(r"\[shift-count\] SP_steps=(\d+) TP_steps=(\d+)")


def parse_server_log(path: str) -> ServerStats:
    """Parse iteration data from server log using vLLM's native iteration logging."""
    stats = ServerStats()

    if not os.path.isfile(path):
        return stats

    with open(path) as f:
        for line in f:
            m = VLLM_ITERATION_RE.search(line)
            if m:
                ctx_tokens = int(m.group(3))
                gen_tokens = int(m.group(5))
                elapsed_ms = float(m.group(6))
                entry = IterationEntry(
                    iteration=int(m.group(1)),
                    timestamp=elapsed_ms,
                    prefill=ctx_tokens,
                    decode=gen_tokens,
                    mode="",
                )
                stats.iterations.append(entry)
                continue

            m = SHIFT_COUNT_RE.search(line)
            if m:
                stats.sp_steps = int(m.group(1))
                stats.tp_steps = int(m.group(2))

    return stats


def infer_modes(stats: ServerStats, threshold: int):
    """Infer SP/TP mode from total scheduled tokens vs threshold.

    total_tokens > threshold => SP, else => TP.
    """
    stats.sp_steps = 0
    stats.tp_steps = 0
    stats.shift_transitions = 0
    prev_mode = None
    for it in stats.iterations:
        total = it.prefill + it.decode
        it.mode = "SP" if total > threshold else "TP"
        if it.mode == "SP":
            stats.sp_steps += 1
        else:
            stats.tp_steps += 1
        if prev_mode is not None and it.mode != prev_mode:
            stats.shift_transitions += 1
        prev_mode = it.mode


def parse_bench_result(path: str) -> BenchResult:
    result = BenchResult()
    if not os.path.isfile(path):
        return result

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.num_requests += 1
            result.ttft_ms.append(float(row["ttft_ms"]))
            result.tpot_ms.append(float(row["tpot_ms"]))
            result.e2el_ms.append(float(row["e2el_ms"]))
            result.input_lens.append(int(row["input_len"]))
            result.output_lens.append(int(row["output_len"]))

    return result


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def compute_throughput(bench: BenchResult) -> float:
    """Total output tokens / total wall time (approx from max e2el)."""
    if not bench.e2el_ms:
        return 0.0
    total_output = sum(bench.output_lens)
    wall_s = max(bench.e2el_ms) / 1000.0
    return total_output / wall_s if wall_s > 0 else 0.0


def iter_latencies(stats: ServerStats) -> list[float]:
    """Per-iteration latency in ms (directly from elapsed time)."""
    return [it.timestamp for it in stats.iterations]


def shift_boundary_latencies(stats: ServerStats) -> tuple[list[float], list[float]]:
    """Latencies at shift boundaries vs non-boundaries.

    Returns (boundary_latencies, non_boundary_latencies) in ms.
    A "boundary" iteration is one where the mode changed from the previous.
    """
    boundary = []
    non_boundary = []
    for i in range(1, len(stats.iterations)):
        lat = stats.iterations[i].timestamp
        if stats.iterations[i].mode != stats.iterations[i - 1].mode:
            boundary.append(lat)
        else:
            non_boundary.append(lat)
    return boundary, non_boundary


@dataclass
class AnnotatedIteration:
    """An iteration with its computed latency and shift flag."""
    iteration: int
    latency_ms: float
    prefill: int
    decode: int
    total_tokens: int
    mode: str
    is_boundary: bool


def annotate_iterations(stats: ServerStats) -> list[AnnotatedIteration]:
    """Build per-iteration records with latency and boundary flag."""
    result = []
    for i in range(len(stats.iterations)):
        cur = stats.iterations[i]
        prev_mode = stats.iterations[i - 1].mode if i > 0 else cur.mode
        result.append(AnnotatedIteration(
            iteration=cur.iteration,
            latency_ms=cur.timestamp,
            prefill=cur.prefill,
            decode=cur.decode,
            total_tokens=cur.prefill + cur.decode,
            mode=cur.mode,
            is_boundary=(cur.mode != prev_mode),
        ))
    return result


def decode_only_shift_cost(records: list[AnnotatedIteration]) -> dict:
    """Isolate shift cost using decode-only iterations (prefill=0).

    Comparing boundary vs non-boundary iterations that are all pure decode
    removes the confound of prefill workload differences.
    """
    boundary_decode = [r.latency_ms for r in records
                       if r.is_boundary and r.prefill == 0]
    non_boundary_decode = [r.latency_ms for r in records
                           if not r.is_boundary and r.prefill == 0]
    return {
        "boundary": boundary_decode,
        "non_boundary": non_boundary_decode,
    }


def bucketed_shift_cost(records: list[AnnotatedIteration],
                         bucket_size: int = 50) -> list[dict]:
    """Compare boundary vs non-boundary latency within decode-token buckets.

    Groups decode-only iterations by decode token count (rounded to
    bucket_size), then compares mean latency of boundary vs non-boundary
    within each bucket. This controls for batch-size differences.
    """
    from collections import defaultdict
    buckets_b: dict[int, list[float]] = defaultdict(list)
    buckets_nb: dict[int, list[float]] = defaultdict(list)

    for r in records:
        if r.prefill > 0:
            continue
        bucket = (r.decode // bucket_size) * bucket_size
        if r.is_boundary:
            buckets_b[bucket].append(r.latency_ms)
        else:
            buckets_nb[bucket].append(r.latency_ms)

    results = []
    all_keys = sorted(set(buckets_b.keys()) | set(buckets_nb.keys()))
    for k in all_keys:
        b = buckets_b.get(k, [])
        nb = buckets_nb.get(k, [])
        if not b and not nb:
            continue
        results.append({
            "bucket": f"{k}-{k + bucket_size}",
            "boundary_n": len(b),
            "boundary_mean": mean(b),
            "non_boundary_n": len(nb),
            "non_boundary_mean": mean(nb),
        })
    return results


def print_header(title: str):
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def print_bench_summary(label: str, bench: BenchResult):
    if not bench.ttft_ms:
        print(f"  [{label}] No benchmark data found.")
        return
    print(f"  [{label}]")
    print(f"    Requests:          {bench.num_requests}")
    tput = compute_throughput(bench)
    print(f"    Throughput:        {tput:.1f} tok/s")
    print(f"    TTFT  mean/p50/p99: {mean(bench.ttft_ms):.1f} / "
          f"{percentile(bench.ttft_ms, 50):.1f} / "
          f"{percentile(bench.ttft_ms, 99):.1f} ms")
    print(f"    TPOT  mean/p50/p99: {mean(bench.tpot_ms):.2f} / "
          f"{percentile(bench.tpot_ms, 50):.2f} / "
          f"{percentile(bench.tpot_ms, 99):.2f} ms")
    print(f"    E2EL  mean/p50/p99: {mean(bench.e2el_ms):.1f} / "
          f"{percentile(bench.e2el_ms, 50):.1f} / "
          f"{percentile(bench.e2el_ms, 99):.1f} ms")


def print_server_summary(label: str, stats: ServerStats):
    if not stats.iterations:
        print(f"  [{label}] No server iteration data found.")
        return
    print(f"  [{label}]")
    print(f"    Total iterations:     {len(stats.iterations)}")
    print(f"    SP steps / TP steps:  {stats.sp_steps} / {stats.tp_steps}")
    pct_sp = stats.sp_steps / max(stats.sp_steps + stats.tp_steps, 1) * 100
    print(f"    SP percentage:        {pct_sp:.1f}%")
    print(f"    Mode transitions:     {stats.shift_transitions}")


def analyze_case1():
    print_header("Case 1: Always SP (long prompt, short gen, batch=1024)")
    print("  Goal: shift enabled always picks SP. Verify no overhead vs pure SP.")
    print()

    bench_a = parse_bench_result(os.path.join(RESULTS_DIR, "case1_sp8_baseline.json"))
    bench_b = parse_bench_result(os.path.join(RESULTS_DIR, "case1_shift_sp.json"))
    server_a = parse_server_log(os.path.join(LOGS_DIR, "server_case1_sp8_baseline.log"))
    server_b = parse_server_log(os.path.join(LOGS_DIR, "server_case1_shift_sp.log"))

    if server_a.iterations:
        infer_modes(server_a, 64)
    if server_b.iterations:
        infer_modes(server_b, 64)

    print("--- Server Stats ---")
    print_server_summary("SP8 baseline", server_a)
    print_server_summary("Shift (SP)", server_b)

    if server_b.tp_steps > 0:
        print(f"\n  WARNING: Shift server used TP mode {server_b.tp_steps} times "
              f"(expected 0 for always-SP test)")

    print("\n--- Benchmark Results ---")
    print_bench_summary("SP8 baseline", bench_a)
    print_bench_summary("Shift (SP)", bench_b)

    if bench_a.e2el_ms and bench_b.e2el_ms:
        tput_a = compute_throughput(bench_a)
        tput_b = compute_throughput(bench_b)
        if tput_a > 0:
            overhead = (tput_a - tput_b) / tput_a * 100
            print(f"\n  Throughput delta: {overhead:+.1f}% "
                  f"({'shift slower' if overhead > 0 else 'shift faster'})")


def analyze_case2():
    print_header("Case 2: Always TP (short prompt, long gen, batch=1)")
    print("  Goal: shift TP matches vanilla TP=8 performance.")
    print()

    bench_a = parse_bench_result(os.path.join(RESULTS_DIR, "case2_tp8_baseline.json"))
    bench_b = parse_bench_result(os.path.join(RESULTS_DIR, "case2_shift_tp.json"))
    server_a = parse_server_log(os.path.join(LOGS_DIR, "server_case2_tp8_baseline.log"))
    server_b = parse_server_log(os.path.join(LOGS_DIR, "server_case2_shift_tp.log"))

    if server_a.iterations:
        infer_modes(server_a, 512)
    if server_b.iterations:
        infer_modes(server_b, 512)

    print("--- Server Stats ---")
    print_server_summary("TP8 baseline", server_a)
    print_server_summary("Shift (TP)", server_b)

    if server_b.sp_steps > 0:
        print(f"\n  NOTE: Shift server used SP mode {server_b.sp_steps} times "
              f"(may include prefill iterations)")

    print("\n--- Benchmark Results ---")
    print_bench_summary("TP8 baseline", bench_a)
    print_bench_summary("Shift (TP)", bench_b)

    if bench_a.tpot_ms and bench_b.tpot_ms:
        tpot_a = mean(bench_a.tpot_ms)
        tpot_b = mean(bench_b.tpot_ms)
        if tpot_a > 0:
            overhead = (tpot_b - tpot_a) / tpot_a * 100
            print(f"\n  TPOT delta: {overhead:+.1f}% "
                  f"({'shift slower' if overhead > 0 else 'shift faster'})")


def analyze_case3():
    print_header("Case 3: Frequent SP<->TP Shifts")
    print("  Goal: verify shift cost is small when mode changes frequently.")
    print()

    bench_a = parse_bench_result(os.path.join(RESULTS_DIR, "case3_shift.json"))
    bench_b = parse_bench_result(os.path.join(RESULTS_DIR, "case3_sp8_baseline.json"))
    server_a = parse_server_log(os.path.join(LOGS_DIR, "server_case3_shift.log"))
    server_b = parse_server_log(os.path.join(LOGS_DIR, "server_case3_sp8_baseline.log"))

    THRESHOLD = 64
    if server_a.iterations:
        infer_modes(server_a, THRESHOLD)
    if server_b.iterations:
        infer_modes(server_b, THRESHOLD)

    print("--- Server Stats ---")
    print_server_summary("Shift (oscillating)", server_a)
    print_server_summary("SP8 baseline", server_b)

    if server_a.shift_transitions == 0:
        print("\n  No mode transitions detected (unexpected for case 3).")
    else:
        records = annotate_iterations(server_a)

        # --- Raw boundary analysis (for reference) ---
        print("\n--- Raw Boundary Analysis (all iterations) ---")
        boundary, non_boundary = shift_boundary_latencies(server_a)
        if boundary and non_boundary:
            print(f"  Boundary iterations:    {len(boundary)}")
            print(f"  Non-boundary iters:     {len(non_boundary)}")
            print(f"  Mean latency boundary:  {mean(boundary):.2f} ms")
            print(f"  Mean latency non-bnd:   {mean(non_boundary):.2f} ms")
            print(f"  NOTE: raw comparison mixes prefill vs decode workloads")

        # --- Decode-only shift cost (controls for prefill confound) ---
        print("\n--- Decode-Only Shift Cost (prefill=0, isolates shift overhead) ---")
        dc = decode_only_shift_cost(records)
        b_dec = dc["boundary"]
        nb_dec = dc["non_boundary"]
        if b_dec and nb_dec:
            print(f"  Decode-only boundary iterations:     {len(b_dec)}")
            print(f"  Decode-only non-boundary iterations: {len(nb_dec)}")
            print(f"  Mean latency at shift (decode-only): {mean(b_dec):.3f} ms")
            print(f"  Mean latency no-shift (decode-only): {mean(nb_dec):.3f} ms")
            shift_cost = mean(b_dec) - mean(nb_dec)
            print(f"  >>> Pure shift cost:                 {shift_cost:+.3f} ms per iteration")
            if mean(nb_dec) > 0:
                pct = shift_cost / mean(nb_dec) * 100
                print(f"  >>> Shift overhead:                  {pct:+.1f}%")
            print(f"  P50 latency at shift (decode-only):  {percentile(b_dec, 50):.3f} ms")
            print(f"  P50 latency no-shift (decode-only):  {percentile(nb_dec, 50):.3f} ms")
            print(f"  P99 latency at shift (decode-only):  {percentile(b_dec, 99):.3f} ms")
            print(f"  P99 latency no-shift (decode-only):  {percentile(nb_dec, 99):.3f} ms")
        else:
            print(f"  Not enough decode-only data "
                  f"(boundary={len(b_dec)}, non-boundary={len(nb_dec)})")

        # --- Bucketed analysis (controls for batch size) ---
        print("\n--- Bucketed Shift Cost (by decode token count, prefill=0) ---")
        print(f"  {'Decode Toks':<14} {'Shift N':>8} {'Shift Mean':>11} "
              f"{'NoShift N':>10} {'NoShift Mean':>13} {'Delta':>10}")
        buckets = bucketed_shift_cost(records, bucket_size=50)
        for b in buckets:
            delta_str = ""
            if b["boundary_n"] > 0 and b["non_boundary_n"] > 0:
                d = b["boundary_mean"] - b["non_boundary_mean"]
                delta_str = f"{d:+.3f} ms"
            print(f"  {b['bucket']:<14} {b['boundary_n']:>8} "
                  f"{b['boundary_mean']:>10.3f}ms "
                  f"{b['non_boundary_n']:>10} "
                  f"{b['non_boundary_mean']:>12.3f}ms "
                  f"{delta_str:>10}")

    print("\n--- Benchmark Results ---")
    print_bench_summary("Shift (oscillating)", bench_a)
    print_bench_summary("SP8 baseline", bench_b)

    if bench_a.tpot_ms and bench_b.tpot_ms:
        tpot_a = mean(bench_a.tpot_ms)
        tpot_b = mean(bench_b.tpot_ms)
        if tpot_b > 0:
            delta = (tpot_a - tpot_b) / tpot_b * 100
            print(f"\n  TPOT delta vs SP8 baseline: {delta:+.1f}%")
    if bench_a.e2el_ms and bench_b.e2el_ms:
        tput_a = compute_throughput(bench_a)
        tput_b = compute_throughput(bench_b)
        if tput_b > 0:
            delta = (tput_a - tput_b) / tput_b * 100
            print(f"  Throughput vs SP8 baseline: {delta:+.1f}% "
                  f"({'shift faster' if delta > 0 else 'shift slower'})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze shift parallelism verification results"
    )
    parser.add_argument(
        "--case",
        choices=["case1", "case2", "case3", "all"],
        default="all",
        help="Which case to analyze (default: all)",
    )
    args = parser.parse_args()

    print("Shift Parallelism Verification -- Results Analysis")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Logs dir:    {LOGS_DIR}")

    if args.case in ("case1", "all"):
        analyze_case1()
    if args.case in ("case2", "all"):
        analyze_case2()
    if args.case in ("case3", "all"):
        analyze_case3()

    print()
    print("=" * 70)
    print("  Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
