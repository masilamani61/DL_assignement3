#!/bin/bash
# run_all_experiments.sh
# Runs all 5 W&B experiments sequentially with logging

mkdir -p logs

echo "=================================================="
echo "  DA6401 Assignment 3 — Running All Experiments"
echo "=================================================="
echo ""

run_experiment() {
    local exp_num=$1
    local exp_file=$2
    local exp_name=$3

    echo "--------------------------------------------------"
    echo "  Starting Experiment $exp_num: $exp_name"
    echo "  Log: logs/exp${exp_num}.txt"
    echo "--------------------------------------------------"

    python experiments/$exp_file > logs/exp${exp_num}.txt 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Experiment $exp_num complete!"
    else
        echo "  ✗ Experiment $exp_num FAILED (exit code $EXIT_CODE)"
        echo "  Check logs/exp${exp_num}.txt for details"
    fi
    echo ""
}

# Run all experiments sequentially
run_experiment 1 "exp1.py"    "Noam vs Fixed LR"
run_experiment 2 "exp2.py"      "Scaling Factor Ablation"
run_experiment 3 "exp3.py"   "Attention Heatmaps"
run_experiment 4 "exp4.py"       "PE vs Learned Embeddings"
run_experiment 5 "exp5.py"     "Label Smoothing Ablation"

echo "=================================================="
echo "  All experiments done!"
echo "  Check individual logs in logs/"
echo "=================================================="