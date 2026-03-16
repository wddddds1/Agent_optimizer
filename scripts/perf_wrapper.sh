#!/bin/bash
# Wrapper: perf record (sampling) + perf stat (hardware counters) for the target.
# PERF_DATA_FILE env var specifies where perf.data is written.
PERF_OUT="${PERF_DATA_FILE:-$(pwd)/perf.data}"
PERF_DIR="$(dirname "$PERF_OUT")"
mkdir -p "$PERF_DIR"
STAT_OUT="$PERF_DIR/perf_stat.txt"

# perf stat wraps perf record so both run in a single BWA invocation.
# --call-graph fp is near-zero overhead; stat counters are dominated by the target.
exec perf stat \
    -e instructions,cycles,cache-misses,cache-references,branch-misses,branches \
    --output "$STAT_OUT" \
    -- \
    perf record --call-graph fp -F 5 -o "$PERF_OUT" -- "$@"
