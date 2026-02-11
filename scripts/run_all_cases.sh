#!/bin/bash
# run_all_cases.sh - 自动运行所有/选定算例的优化
# 用法:
#   ./scripts/run_all_cases.sh              # 运行所有 source_patch 标签的算例
#   ./scripts/run_all_cases.sh --all        # 运行所有算例
#   ./scripts/run_all_cases.sh case1 case2  # 运行指定的算例

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 加载用户环境（API key/conda 等）
if [[ -f "$HOME/.zshrc" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.zshrc"
fi

# Python interpreter (prefer env if set)
PYTHON_BIN="${PYTHON_BIN:-python}"

# 时间戳
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="artifacts/batch_runs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# 默认预算（可通过环境变量覆盖）
MAX_ITERS="${MAX_ITERS:-20}"
MAX_RUNS="${MAX_RUNS:-100}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  HPC Agent Platform - Batch Run${NC}"
echo -e "${BLUE}  Started: $(date)${NC}"
echo -e "${BLUE}  Log dir: $LOG_DIR${NC}"
echo -e "${BLUE}============================================${NC}"

# 获取算例列表
get_cases() {
    if [[ "$1" == "--all" ]]; then
        # 所有算例
        "$PYTHON_BIN" -c "
import yaml
with open('configs/lammps_cases.yaml') as f:
    cases = yaml.safe_load(f)['cases']
for name in cases.keys():
    print(name)
"
    elif [[ -n "$1" ]]; then
        # 指定的算例
        echo "$@"
    else
        # 默认: source_patch 标签的算例
        "$PYTHON_BIN" -c "
import yaml
with open('configs/lammps_cases.yaml') as f:
    cases = yaml.safe_load(f)['cases']
for name, cfg in cases.items():
    tags = cfg.get('tags', [])
    if 'source_patch' in tags:
        print(name)
"
    fi
}

CASES=$(get_cases "$@")
CASES_ARRAY=($CASES)
TOTAL=${#CASES_ARRAY[@]}

echo -e "\n${YELLOW}Will run $TOTAL cases:${NC}"
for case in $CASES; do
    echo "  - $case"
done
echo ""

# 结果记录（bash 3.2 兼容）
RESULTS_FILE="$LOG_DIR/results.tsv"
echo -e "case\tstatus\tduration" > "$RESULTS_FILE"
PASSED=0
FAILED=0

# 运行单个算例
run_case() {
    local case_name=$1
    local case_log="$LOG_DIR/${case_name}.log"
    local start_time=$(date +%s)

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[$((CURRENT+1))/$TOTAL] Running: $case_name${NC}"
    echo -e "${BLUE}Log: $case_log${NC}"
    echo -e "${BLUE}Started: $(date +%H:%M:%S)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # 运行优化
    if "$PYTHON_BIN" -m orchestrator.main \
        --case "$case_name" \
        --ui console \
        --max-iters "$MAX_ITERS" \
        --max-runs "$MAX_RUNS" \
        > "$case_log" 2>&1; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${case_name}\tPASS\t${duration}" >> "$RESULTS_FILE"
        ((PASSED++))
        echo -e "${GREEN}✓ $case_name completed in ${duration}s${NC}"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${case_name}\tFAIL\t${duration}" >> "$RESULTS_FILE"
        ((FAILED++))
        echo -e "${RED}✗ $case_name failed after ${duration}s${NC}"
        # 显示最后几行错误
        echo -e "${RED}Last 10 lines of log:${NC}"
        tail -10 "$case_log" | sed 's/^/  /'
    fi
    echo ""
}

# 主循环
CURRENT=0
for case in $CASES; do
    run_case "$case"
    ((CURRENT++))
done

# 生成总结报告
SUMMARY_FILE="$LOG_DIR/summary.txt"
{
    echo "============================================"
    echo "  Batch Run Summary"
    echo "  Timestamp: $TIMESTAMP"
    echo "  Completed: $(date)"
    echo "============================================"
    echo ""
    echo "Results: $PASSED passed, $FAILED failed (total: $TOTAL)"
    echo ""
    echo "Case Results:"
    echo "─────────────────────────────────────────────"
    printf "%-25s %-8s %s\n" "Case" "Status" "Duration"
    echo "─────────────────────────────────────────────"
    while IFS=$'\t' read -r name status duration; do
        if [[ "$name" == "case" ]]; then
            continue
        fi
        printf "%-25s %-8s %ss\n" "$name" "$status" "$duration"
    done < "$RESULTS_FILE"
    echo "─────────────────────────────────────────────"
} > "$SUMMARY_FILE"

# 打印总结
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Batch Run Complete${NC}"
echo -e "${BLUE}  Finished: $(date)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
cat "$SUMMARY_FILE"

# 如果有失败，显示失败算例的日志位置
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed case logs:${NC}"
    while IFS=$'\t' read -r name status duration; do
        if [[ "$name" == "case" ]]; then
            continue
        fi
        if [[ "$status" == "FAIL" ]]; then
            echo "  $LOG_DIR/${name}.log"
        fi
    done < "$RESULTS_FILE"
fi

# 复制最新结果到固定位置方便查看
cp "$SUMMARY_FILE" "$LOG_DIR/../latest_summary.txt"
ln -sf "$TIMESTAMP" "$LOG_DIR/../latest"

echo ""
echo -e "${GREEN}Summary saved to: $SUMMARY_FILE${NC}"
echo -e "${GREEN}Latest link: artifacts/batch_runs/latest${NC}"

# 返回码
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
