#!/bin/bash
# run_overnight.sh - 后台运行所有算例优化（适合睡前执行）
# 用法:
#   ./scripts/run_overnight.sh              # 后台运行 source_patch 算例
#   ./scripts/run_overnight.sh --all        # 后台运行所有算例
#   ./scripts/run_overnight.sh case1 case2  # 后台运行指定算例
#
# 查看进度:
#   tail -f artifacts/batch_runs/latest/*.log
#   cat artifacts/batch_runs/latest_summary.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
NOHUP_LOG="artifacts/batch_runs/nohup_${TIMESTAMP}.out"
mkdir -p artifacts/batch_runs

echo "============================================"
echo "  Starting overnight batch run"
echo "  Time: $(date)"
echo "============================================"
echo ""
echo "Running in background with nohup..."
echo "Output log: $NOHUP_LOG"
echo ""

# 后台运行
nohup "$SCRIPT_DIR/run_all_cases.sh" "$@" > "$NOHUP_LOG" 2>&1 &
PID=$!

echo "Started with PID: $PID"
echo ""
echo "Commands to monitor:"
echo "  tail -f $NOHUP_LOG                    # 实时查看主日志"
echo "  tail -f artifacts/batch_runs/latest/*.log  # 查看当前算例日志"
echo "  cat artifacts/batch_runs/latest_summary.txt # 查看总结（完成后）"
echo "  ps aux | grep run_all_cases           # 检查是否还在运行"
echo "  kill $PID                             # 停止运行"
echo ""
echo "Sleep well! 💤"
