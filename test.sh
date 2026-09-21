#! /bin/bash

mkdir -p figures

# Dataset generator test run: every image the real run would make for one chunk, written into
# figures/ instead of into the dataset. num_chunks is the model count over 3, so a chunk is
# 3 models -- here chunk 5, all 251 poses of each, about an hour on one GPU.
# Drop -test_run and this same command writes into the dataset instead.
NUM_MODELS=$(ls /workspace/data/srncars/cars_test | wc -l)
NUM_CHUNKS=$((NUM_MODELS / 3))

# clear the whole folder, so nothing from an earlier run is mistaken for this one's output.
# This does take the stitched paper figures with it -- rerun sonar_paper_figures.py to remake them
rm -f figures/*

CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
    generate_dataset.py -num_chunks $NUM_CHUNKS -chunk_id 5 -test_run

# sudo nvidia-smi -i 1,3 -pl 200
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 sonar_paper_figures.py
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 paper_figures.py
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 debug_side_scan.py
