#! /bin/bash
#
# Render every srn_cars split across 6 GPUs: six workers in parallel, each pinned to one GPU. Each
# split is chunked on its own, and GPU i renders chunk i of cars_train, then chunk i of cars_val,
# then chunk i of cars_test. A worker moves on to its next split without waiting for the others.
#
#   ./distributed_generate_dataset.sh                 # all three splits
#   ./distributed_generate_dataset.sh -test_run       # args go to every generate_dataset.py call
#
# Finished poses are skipped, so a chunk that dies is resumed by rerunning the whole script.
# Logs land one per split and chunk in logs/, since six processes interleaved on one terminal are
# unreadable. Every STATUS_EVERY_S a status line gives images done and an ETA per split and overall,
# projected straight from images done, images left and time running.

set -u

# rendered in this order by every worker; all args are forwarded to generate_dataset.py
SPLITS=(cars_train cars_val cars_test)

GPUS=(0 1 2 3 4 5)
NUM_CHUNKS=${#GPUS[@]}
POWER_LIMIT_W=200
PYTHON=/workspace/berian/miniconda3/envs/sarrender/bin/python3.8
LOG_DIR=logs
STATUS_EVERY_S=600
# split start times and chunk end markers, for the status line; cleared every run
PROGRESS_DIR=$LOG_DIR/.progress

GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

# Cap board power as test.sh does, skipping sudo entirely when every GPU already has the limit.
# Otherwise sudo wants a password here, so try the non-interactive form first, prompt only when
# there is a terminal to prompt on, and never block: under nohup a plain sudo would hang forever
# waiting for input. An empty query (nvidia-smi failed) counts as not set.
if nvidia-smi -i "$GPU_LIST" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null \
        | awk -v w="$POWER_LIMIT_W" 'int($1) != w {bad=1} END {exit (bad || NR == 0)}'; then
    echo "power limit: already ${POWER_LIMIT_W}W on GPUs $GPU_LIST"
elif sudo -n nvidia-smi -i "$GPU_LIST" -pl "$POWER_LIMIT_W" >/dev/null 2>&1; then
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

rm -rf "$PROGRESS_DIR"
mkdir -p "$LOG_DIR" "$PROGRESS_DIR"

# kill the whole group on Ctrl-C, so one interrupt stops all six and not just the wait
trap 'echo; echo "interrupted -- stopping all chunks"; kill 0; exit 130' INT TERM

echo "splits ${SPLITS[*]}, in that order: $NUM_CHUNKS chunks each over GPUs $GPU_LIST"

log_path() {  # split, chunk index
    echo "$LOG_DIR/${1}_chunk${2}of${NUM_CHUNKS}_gpu${GPUS[$2]}.log"
}

# the images still to render per split, counted before any worker starts so that images left is
# simply this minus images done. One count per split, in parallel, each over all of its chunks
echo "counting the images left to render in each split..."
for split in "${SPLITS[@]}"; do
    "$PYTHON" generate_dataset.py -split "$split" -num_chunks "$NUM_CHUNKS" -count "$@" \
        > "$LOG_DIR/${split}_count.log" 2>&1 &
done
wait
declare -A TODO
for split in "${SPLITS[@]}"; do
    TODO[$split]=$(awk '/^count:/ {print $3}' "$LOG_DIR/${split}_count.log")
    if [ -z "${TODO[$split]}" ]; then
        echo "WARNING: counting $split failed, see $LOG_DIR/${split}_count.log -- no status lines this run"
        STATUS_EVERY_S=0
    else
        echo "  $split: ${TODO[$split]} images to render"
    fi
done

# one worker per GPU: its chunk of each split in turn. A split that crashes is reported and the
# worker goes on to the next one, since the splits are independent; the worker exits non-zero if
# any of its splits did
run_worker() {  # chunk index, then the args for generate_dataset.py
    local i=$1 gpu=${GPUS[$1]} split failed=0
    shift
    for split in "${SPLITS[@]}"; do
        # the first worker onto a split starts its clock
        [ -e "$PROGRESS_DIR/$split.start" ] || date +%s > "$PROGRESS_DIR/$split.start"
        echo "  GPU $gpu: starting $split chunk $i  log $(log_path "$split" "$i")"
        if CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" generate_dataset.py \
                -split "$split" -num_chunks "$NUM_CHUNKS" -chunk_id "$i" "$@" \
                > "$(log_path "$split" "$i")" 2>&1; then
            echo "  GPU $gpu: $split chunk $i finished"
        else
            echo "  GPU $gpu: $split chunk $i FAILED -- see $(log_path "$split" "$i")"
            failed=1
        fi
        touch "$PROGRESS_DIR/$split.chunk$i.end"
    done
    return $failed
}

fmt_duration() {  # seconds
    printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60))
}

# time left at the rate so far: elapsed * left / done
eta() {  # images done, images left, seconds elapsed
    if (($1 > 0)); then fmt_duration $(($3 * $2 / $1)); else echo "--"; fi
}

# images rendered this run, summed off the "(<n> images)" of each finished object in the split's
# logs. The logs are rewritten every run, so this counts nothing from earlier runs
images_done() {  # split
    local i
    for i in "${!GPUS[@]}"; do
        cat "$(log_path "$1" "$i")" 2>/dev/null
    done | grep -o '([0-9]* images)' | awk '{s += substr($1, 2)} END {print s + 0}'
}

print_status() {  # epoch seconds the workers started at
    local now split n_done left n_ended start all_done=0 all_left=0 all_todo=0
    now=$(date +%s)
    echo "status, $(fmt_duration $((now - $1))) in:"
    for split in "${SPLITS[@]}"; do
        n_done=$(images_done "$split")
        left=$((${TODO[$split]} - n_done))
        ((left < 0)) && left=0
        all_done=$((all_done + n_done)); all_left=$((all_left + left))
        all_todo=$((all_todo + ${TODO[$split]}))
        n_ended=$(ls "$PROGRESS_DIR" | grep -c "^$split\.chunk.*\.end$")
        if ((n_ended == NUM_CHUNKS)); then
            echo "  $split: finished, $n_done images"
        elif [ ! -e "$PROGRESS_DIR/$split.start" ]; then
            echo "  $split: not started, ${TODO[$split]} images"
        else
            start=$(cat "$PROGRESS_DIR/$split.start")
            echo "  $split: $n_done/${TODO[$split]} images, ETA $(eta "$n_done" "$left" $((now - start)))"
        fi
    done
    echo "  overall: $all_done/$all_todo images, ETA $(eta "$all_done" "$all_left" $((now - $1)))"
}

t_start=$(date +%s)
pids=()
for i in "${!GPUS[@]}"; do
    run_worker "$i" "$@" &
    pids+=($!)
done

echo "watch one with:  tail -f $(log_path "${SPLITS[0]}" 0)"

status_pid=
if ((STATUS_EVERY_S > 0)); then
    while sleep "$STATUS_EVERY_S"; do print_status "$t_start"; done &
    status_pid=$!
fi

# wait on each worker by pid, so the exit status of every one is reported rather than only the last
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "GPU ${GPUS[$i]} (chunk $i) done with all splits"
    else
        echo "GPU ${GPUS[$i]} (chunk $i) had a failed split -- see its logs above"
        status=1
    fi
done

[ -n "$status_pid" ] && kill "$status_pid" 2>/dev/null
echo "all chunks done for ${SPLITS[*]}"
exit $status
