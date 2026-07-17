#! /bin/bash

mkdir -p figures
rm figures/*

CUDA_VISIBLE_DEVICES=2 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 render_cvdomes.py
