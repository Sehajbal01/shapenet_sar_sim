#! /bin/bash

mkdir -p figures
rm figures/*

sudo nvidia-smi -i 1,3 -pl 200
CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 sonar_paper_figures.py
CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 paper_figures.py
# CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 debug_side_scan.py
