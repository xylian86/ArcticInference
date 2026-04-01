#!/bin/bash
# Shift parallelism verification tests
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACES_DIR="${SCRIPT_DIR}/traces"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOGS_DIR="${SCRIPT_DIR}/logs"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export VLLM_DISABLE_COMPILE_CACHE=1
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}"

if [ -n "${VENV:-}" ] && [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

if [ ! -f "${TRACES_DIR}/case1_always_sp.jsonl" ]; then
    echo "Trace files not found. Generating..."
    python "${SCRIPT_DIR}/gen_traces.py"
fi

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server failed to start within 10 minutes"
    return 1
}

kill_server() {
    echo "Stopping server..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 10
}

run_bench() {
    local label=$1
    local trace=$2
    echo ""
    echo "  Benchmarking: ${label}"
    vllm bench serve \
        --model "${MODEL}" \
        --host 127.0.0.1 \
        --port ${PORT} \
        --trace-dataset-path "${trace}" \
        --ignore-eos \
        --trace-output-path "${RESULTS_DIR}/${label}.json" \
        > "${LOGS_DIR}/client_${label}.log" 2>&1
    echo "  Client log: ${LOGS_DIR}/client_${label}.log"
    tail -20 "${LOGS_DIR}/client_${label}.log"
}

print_iter_stats() {
    local label=$1
    local slog="${LOGS_DIR}/server_${label}.log"
    echo ""
    local count=$(grep -c "Iteration(" "${slog}" 2>/dev/null || echo 0)
    echo "  Iteration log lines: ${count}"
    if [ "${count}" -gt 0 ]; then
        echo "  First 3:"
        grep "Iteration(" "${slog}" | head -3
        echo "  Last 3:"
        grep "Iteration(" "${slog}" | tail -3
    fi
}

# Common vllm serve flags
COMMON_FLAGS="--port ${PORT} --disable-log-requests --enable-logging-iteration-details --max-num-batched-tokens 131072"

# ---------------------------------------------------------------------------
# Case 1: Always SP
# ---------------------------------------------------------------------------
run_case1_sp8_baseline() {
    local label="case1_sp8_baseline"
    echo ""
    echo "============================================================"
    echo "  Case 1A: SP8 baseline (no shift)"
    echo "============================================================"
    kill_server

    ARCTIC_INFERENCE_ENABLED=1 \
    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --ulysses-sequence-parallel-size 8 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case1_always_sp.jsonl"
    print_iter_stats "${label}"
    kill_server
}

run_case1_shift_sp() {
    local label="case1_shift_sp"
    echo ""
    echo "============================================================"
    echo "  Case 1B: SP8 + shift enabled (threshold=64, always SP)"
    echo "============================================================"
    kill_server

    ARCTIC_INFERENCE_ENABLED=1 \
    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --ulysses-sequence-parallel-size 8 \
        --enable-shift-parallel \
        --shift-parallel-threshold 64 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case1_always_sp.jsonl"
    print_iter_stats "${label}"
    kill_server
}

# ---------------------------------------------------------------------------
# Case 2: Always TP
# ---------------------------------------------------------------------------
run_case2_tp8_baseline() {
    local label="case2_tp8_baseline"
    echo ""
    echo "============================================================"
    echo "  Case 2A: TP8 baseline (vanilla vLLM)"
    echo "============================================================"
    kill_server

    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --tensor-parallel-size 8 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case2_always_tp.jsonl"
    print_iter_stats "${label}"
    kill_server
}

run_case2_shift_tp() {
    local label="case2_shift_tp"
    echo ""
    echo "============================================================"
    echo "  Case 2B: SP8 + shift enabled (threshold=512, always TP)"
    echo "============================================================"
    kill_server

    ARCTIC_INFERENCE_ENABLED=1 \
    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --ulysses-sequence-parallel-size 8 \
        --enable-shift-parallel \
        --shift-parallel-threshold 512 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case2_always_tp.jsonl"
    print_iter_stats "${label}"
    kill_server
}

# ---------------------------------------------------------------------------
# Case 3: Frequent shifts
# ---------------------------------------------------------------------------
run_case3_shift() {
    local label="case3_shift"
    echo ""
    echo "============================================================"
    echo "  Case 3A: SP8 + shift enabled (threshold=64, frequent shifts)"
    echo "============================================================"
    kill_server

    ARCTIC_INFERENCE_ENABLED=1 \
    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --ulysses-sequence-parallel-size 8 \
        --enable-shift-parallel \
        --shift-parallel-threshold 64 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case3_shifting.jsonl"
    print_iter_stats "${label}"
    kill_server
}

run_case3_sp8_baseline() {
    local label="case3_sp8_baseline"
    echo ""
    echo "============================================================"
    echo "  Case 3B: SP8 baseline (no shift)"
    echo "============================================================"
    kill_server

    ARCTIC_INFERENCE_ENABLED=1 \
    vllm serve "${MODEL}" ${COMMON_FLAGS} \
        --ulysses-sequence-parallel-size 8 \
        > "${LOGS_DIR}/server_${label}.log" 2>&1 &
    echo "  Server PID: $!"

    wait_for_server
    run_bench "${label}" "${TRACES_DIR}/case3_shifting.jsonl"
    print_iter_stats "${label}"
    kill_server
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
CASE="${1:-all}"

case $CASE in
    case1)
        run_case1_sp8_baseline
        run_case1_shift_sp
        ;;
    case2)
        run_case2_tp8_baseline
        run_case2_shift_tp
        ;;
    case3)
        run_case3_shift
        run_case3_sp8_baseline
        ;;
    all)
        echo "Running all 3 verification cases (6 server configurations)..."
        echo ""

        run_case1_sp8_baseline
        run_case1_shift_sp

        run_case2_tp8_baseline
        run_case2_shift_tp

        run_case3_shift
        run_case3_sp8_baseline

        echo ""
        echo "============================================================"
        echo "  All verification tests complete!"
        echo "============================================================"
        echo "Run 'python verify/analyze.py' to compare results."
        ;;
    *)
        echo "Usage: $0 [case1|case2|case3|all]"
        exit 1
        ;;
esac
