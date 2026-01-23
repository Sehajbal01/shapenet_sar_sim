#! /bin/bash

mkdir -p figures figures/tmp figures/tmp_ray_tracer
rm figures/tmp/*.png
rm figures/tmp_ray_tracer/*.png

CUDA_VISIBLE_DEVICES=1 /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 render_images.py

# rm figures/tmp/*.png
# rm figures/tmp_ray_tracer/*.png
