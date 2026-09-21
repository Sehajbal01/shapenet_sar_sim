#! /bin/bash
#
# Render one srn_cars split across 6 GPUs: six generate_dataset.py processes in parallel, each
# pinned to one GPU and given one chunk of the object list.
#
#   ./distributed_generate_dataset.sh                 # cars_test, the default
#   ./distributed_generate_dataset.sh cars_train      # another split
#   ./distributed_generate_dataset.sh cars_test -test_run   # extra args go to generate_dataset.py
#
# Finished poses are skipped, so a chunk that dies is resumed by rerunning the whole script.
# Logs land one per chunk in logs/, since six processes interleaved on one terminal are unreadable.

set -u

SPLIT=${1:-cars_test}
shift 2>/dev/null || true          # anything further is forwarded to generate_dataset.py

GPUS=(0 1 2 3 4 5)
NUM_CHUNKS=${#GPUS[@]}
POWER_LIMIT_W=200
PYTHON=/workspace/berian/miniconda3/envs/sarrender/bin/python3.8
LOG_DIR=logs

GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

# Cap board power as test.sh's commented nvidia-smi line does. sudo wants a password here, so
# try the non-interactive form first, prompt only when there is a terminal to prompt on, and
# never block: under nohup a plain sudo would hang forever waiting for input.
if sudo -n nvidia-smi -i "$GPU_LIST" -pl "$POWER_LIMIT_W" >/dev/null 2>&1; then
    echo "power limit: ${POWER_LIMIT_W}W on GPUs $GPU_LIST"
elif [ -t 0 ]; then
    echo "power limit: ${POWER_LIMIT_W}W on GPUs $GPU_LIST (sudo needs your password)"
    sudo nvidia-smi -i "$GPU_LIST" -pl "$POWER_LIMIT_W" >/dev/null \
        && echo "  set" \
        || echo "  WARNING: failed -- the GPUs keep their current limit"
else
    echo "WARNING: no terminal for sudo, so the ${POWER_LIMIT_W}W limit was NOT applied."
    echo "         set it yourself first:  sudo nvidia-smi -i $GPU_LIST -pl $POWER_LIMIT_W"
fi

mkdir -p "$LOG_DIR"

# kill the whole group on Ctrl-C, so one interrupt stops all six and not just the wait
trap 'echo; echo "interrupted -- stopping all chunks"; kill 0; exit 130' INT TERM

echo "split $SPLIT: $NUM_CHUNKS chunks over GPUs $GPU_LIST"

pids=()
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    log="$LOG_DIR/${SPLIT}_chunk${i}of${NUM_CHUNKS}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" generate_dataset.py \
        -split "$SPLIT" -num_chunks "$NUM_CHUNKS" -chunk_id "$i" "$@" \
        > "$log" 2>&1 &
    pids+=($!)
    echo "  GPU $gpu -> chunk $i  pid ${pids[$i]}  log $log"
done

echo "watch one with:  tail -f $LOG_DIR/${SPLIT}_chunk0of${NUM_CHUNKS}_gpu${GPUS[0]}.log"

# wait on each chunk by pid, so the exit status of every one is reported rather than only the last
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "chunk $i (GPU ${GPUS[$i]}) finished"
    else
        echo "chunk $i (GPU ${GPUS[$i]}) FAILED -- see $LOG_DIR/${SPLIT}_chunk${i}of${NUM_CHUNKS}_gpu${GPUS[$i]}.log"
        status=1
    fi
done

echo "all chunks done for $SPLIT"
exit $status
