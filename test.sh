#! /bin/bash

GPU_NUM=2
# cap the gpu's power draw, in watts -- only when it isn't already capped, to skip the sudo prompt
POWER_LIMIT=200
CUR_POWER_LIMIT=$(nvidia-smi -i $GPU_NUM --query-gpu=power.limit --format=csv,noheader,nounits)
if [ "${CUR_POWER_LIMIT%.*}" != "$POWER_LIMIT" ]; then
    sudo nvidia-smi -i $GPU_NUM -pl $POWER_LIMIT
fi

mkdir -p figures
# clear the whole folder, so nothing from an earlier run is mistaken for this one's output.
rm -f figures/*

# # dataset generation test run
# strip config.json's whole-line // comments first, as config.py does -- jq can't parse them
NUM_MODELS=$(ls "$(sed 's|^\s*//.*$||' config.json | jq -r .srn_cars_dir)/cars_test" | wc -l)
NUM_CHUNKS=$((NUM_MODELS))
CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
    generate_dataset.py -num_chunks $NUM_CHUNKS -chunk_id 5 -test_run -gif #-modalities forward_looking_sonar side_scan_sonar

# SAR paper figures
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     sar_paper_figures.py

# # SSS paper figures
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     sss_paper_figures.py

# # FLS paper figures
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     fls_paper_figures.py

