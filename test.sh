#! /bin/bash

mkdir -p figures figures/tmp
rm figures/tmp/*

CUDA_VISIBLE_DEVICES=2 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 render_cvdomes.py
# CUDA_VISIBLE_DEVICES=0 /home/berian/miniconda3/envs/sarrender/bin/python render_images.py

rm -rf figures/tmp/
