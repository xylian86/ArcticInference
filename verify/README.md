# Shift Parallelism Verification Tests

Automated tests to verify that shift parallelism (dynamic SP/TP switching) works correctly and efficiently.

## Background

Shift parallelism dynamically switches between Sequence Parallelism (SP, via Ulysses) and Tensor Parallelism (TP) based on the number of scheduled tokens per iteration:

- **SP mode**: used when `num_scheduled_tokens > shift_parallel_threshold` (large batches benefit from sequence-parallel communication)
- **TP mode**: used when `num_scheduled_tokens <= shift_parallel_threshold` (small batches use fused full-world tensor parallel)

## Test Cases

| Case | Workload | Shift Config | Baseline | What it verifies |
|------|----------|-------------|----------|-----------------|
| **1** | 1024 requests, input=8192, output=128 (all at t=0) | SP8, threshold=64 | SP8 (no shift) | Shift always picks SP; no overhead vs pure SP |
| **2** | 1 request, input=256, output=8192 | SP8, threshold=512 | TP=8 (vanilla vLLM) | Shift always picks TP; matches vanilla TP=8 perf |
| **3** | 10,000 requests in 100 bursts of 100, input=128, output=64 | SP8, threshold=64 | SP8 (no shift) | Frequent SP/TP shifts; shift cost is small |

## Prerequisites

- 8 GPUs available
- vLLM + ArcticInference installed (optionally set `VENV` to a virtualenv path)
- Model weights accessible (default: `Qwen/Qwen3-4B-Instruct-2507`)

## Directory Structure

```
verify/
├── README.md            # this file
├── gen_traces.py        # generates JSONL trace files
├── run_all.sh           # orchestrates server lifecycle + benchmarks
├── analyze.py           # parses logs/results and prints analysis
├── .gitignore           # excludes generated artifacts below
├── traces/              # generated trace files (created by gen_traces.py)
├── logs/                # server and client logs (created by run_all.sh)
└── results/             # benchmark output JSONs (created by run_all.sh)
```

## How to Run

### Step 1: Generate trace files

```bash
python verify/gen_traces.py
```

This creates the three JSONL trace files under `verify/traces/`. Each file contains requests with `timestamp`, `input_length`, and `output_length` fields.

This step is optional -- `run_all.sh` will auto-generate traces if they don't exist.

### Step 2: Run the tests

Run all three cases (6 server configurations total):

```bash
bash verify/run_all.sh all
```

Or run a single case:

```bash
bash verify/run_all.sh case1   # Always SP
bash verify/run_all.sh case2   # Always TP
bash verify/run_all.sh case3   # Frequent shifts
```

Each case:
1. Starts a `vllm serve` instance with the appropriate configuration
2. Waits for the server to become healthy (up to 10 minutes for model loading + CUDA graph compilation)
3. Runs `vllm bench serve` against the trace file
4. Saves server logs and benchmark results
5. Kills the server

**Expected runtime**: ~30-40 minutes for all 6 configurations (model loading + CUDA graph compilation dominates).

You can override the model with:

```bash
MODEL=your/model VENV=/path/to/venv bash verify/run_all.sh all
```

### Step 3: Analyze results

```bash
python verify/analyze.py
```

Or analyze a single case:

```bash
python verify/analyze.py --case case1
python verify/analyze.py --case case2
python verify/analyze.py --case case3
```

## Understanding the Output

### Case 1 & 2: Overhead comparison

The analysis compares throughput and per-token latency (TPOT) between the shift-enabled server and the baseline. Key metrics:

- **Throughput delta**: percentage difference in output tokens/sec. Near-zero means no overhead.
- **TPOT delta**: percentage difference in time-per-output-token. Near-zero means no overhead.
- **SP/TP step counts**: confirms the shift server stayed in the expected mode.

### Case 3: Shift cost quantification

The analysis uses three methods to measure shift overhead, from coarsest to most precise:

1. **Raw Boundary Analysis**: compares iteration latency at mode transitions vs non-transitions. This is a rough estimate because boundary iterations often have different prefill/decode mixes.

2. **Decode-Only Shift Cost**: filters to iterations with zero prefill tokens (pure decode), then compares boundary vs non-boundary latency. This isolates the shift overhead from prefill workload differences. The reported **"Pure shift cost"** is the key metric (difference in mean latency, in ms).

3. **Bucketed Shift Cost**: further groups decode-only iterations by decode token count (buckets of 50 tokens) and compares within each bucket. This controls for batch-size variation on top of the prefill filter.

## Server Configuration Details

| Config | `ARCTIC_INFERENCE_ENABLED` | `--ulysses-sequence-parallel-size` | `--tensor-parallel-size` | `--enable-shift-parallel` | `--shift-parallel-threshold` |
|--------|---|---|---|---|---|
| Case 1A (SP8 baseline) | 1 | 8 | - | - | - |
| Case 1B (shift, always SP) | 1 | 8 | - | yes | 64 |
| Case 2A (TP8 baseline) | - | - | 8 | - | - |
| Case 2B (shift, always TP) | 1 | 8 | - | yes | 512 |
| Case 3A (shift, oscillating) | 1 | 8 | - | yes | 64 |
| Case 3B (SP8 baseline) | 1 | 8 | - | - | - |

All servers use `--max-num-batched-tokens 131072` and `--enable-logging-iteration-details`.

## Customization

- **Threshold**: edit `--shift-parallel-threshold` values in `run_all.sh`
- **Traces**: edit `gen_traces.py` to change request counts, input/output lengths, or arrival patterns
- **Model**: set `MODEL=...` environment variable
- **Virtualenv**: set `VENV=...` environment variable
- **HF cache**: set `HF_HOME=...` environment variable
