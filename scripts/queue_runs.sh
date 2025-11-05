#!/bin/bash
# Process multiple training runs sequentially
# Usage: bash scripts/queue_runs.sh run1 run2 run3 ...

# Check if any run names provided
if [ $# -eq 0 ]; then
    echo "Error: No run names provided"
    echo "Usage: bash scripts/queue_runs.sh run1 run2 run3 ..."
    exit 1
fi

# Store run names
RUN_NAMES=("$@")
TOTAL_RUNS=$#

echo "========================================"
echo "Queue Processing: $TOTAL_RUNS runs"
echo "========================================"
echo ""

# Phase 1: Validation
echo "Validating all runs before starting..."
echo ""

# Check for duplicates
declare -A seen_runs
for run_name in "${RUN_NAMES[@]}"; do
    if [ -n "${seen_runs[$run_name]}" ]; then
        echo "ERROR: Duplicate run name detected: $run_name"
        exit 1
    fi
    seen_runs[$run_name]=1
done

# Validate each run
for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$PWD/working/runs/${run_name}"

    echo "Validating: $run_name"

    # Check run directory exists
    if [ ! -d "$run_dir" ]; then
        echo "  ERROR: Run directory does not exist: $run_dir"
        echo "  Create the run first with: source scripts/run_init.sh $run_name"
        exit 1
    fi

    # Check config.py exists (run is configured)
    if [ ! -f "$run_dir/config.py" ]; then
        echo "  ERROR: config.py not found in $run_dir"
        echo "  This run has not been configured yet."
        exit 1
    fi

    # Check that run has NOT been executed yet (no checkpoints)
    if [ -d "$run_dir/base_checkpoints" ]; then
        echo "  ERROR: base_checkpoints/ directory already exists in $run_dir"
        echo "  This run has already been executed."
        exit 1
    fi

    # Check for run_failed marker
    if [ -f "$run_dir/run_failed" ]; then
        echo "  ERROR: run_failed marker exists in $run_dir"
        echo "  This run has previously failed. Remove the marker if you want to retry."
        exit 1
    fi

    echo "  ✓ Valid"
done

echo ""
echo "All runs validated successfully!"
echo ""

# Phase 2: Sequential execution
SUCCESS_COUNT=0
FAILED_COUNT=0
CURRENT=0

for run_name in "${RUN_NAMES[@]}"; do
    CURRENT=$((CURRENT + 1))
    run_dir="$PWD/working/runs/${run_name}"

    echo "========================================"
    echo "Processing run $CURRENT of $TOTAL_RUNS: $run_name"
    echo "========================================"

    # Set terminal title
    echo -en "\033]0;Run $CURRENT of $TOTAL_RUNS: $run_name\a"

    # Set environment and run training
    # Note: We need to run this in a subshell to avoid environment pollution
    (
        source scripts/run_set.sh "$run_name" && bash scripts/run_start.sh
    )
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Run completed successfully: $run_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "✗ Run failed with exit code $EXIT_CODE: $run_name"
        echo "Creating run_failed marker..."
        echo "Exit code: $EXIT_CODE" > "$run_dir/run_failed"
        echo "Timestamp: $(date)" >> "$run_dir/run_failed"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    echo ""
done

# Reset terminal title
echo -en "\033]0;Queue Complete\a"

# Summary
echo "========================================"
echo "Queue Processing Complete"
echo "========================================"
echo "Total runs:      $TOTAL_RUNS"
echo "Successful:      $SUCCESS_COUNT"
echo "Failed:          $FAILED_COUNT"
echo "========================================"

if [ $FAILED_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi
