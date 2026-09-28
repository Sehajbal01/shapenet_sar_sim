#! /bin/bash

mkdir -p figures


# clear the whole folder, so nothing from an earlier run is mistaken for this one's output.
# This does take the stitched paper figures with it -- rerun sonar_paper_figures.py to remake them
rm -f figures/*

dataset generation test run
NUM_MODELS=$(ls "$(jq -r .srn_cars_dir config.json)/cars_test" | wc -l)
NUM_CHUNKS=$((NUM_MODELS))
CUDA_VISIBLE_DEVICES=0 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
    generate_dataset.py -num_chunks $NUM_CHUNKS -chunk_id 5 -test_run

# # the sonar and sar paper figures
# CUDA_VISIBLE_DEVICES=0 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     sonar_paper_figures.py
# CUDA_VISIBLE_DEVICES=0 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     paper_figures.py

# sudo nvidia-smi -i 1,3 -pl 200
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 sonar_paper_figures.py
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 paper_figures.py
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 debug_side_scan.py
