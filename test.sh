#! /bin/bash

rm figures/tmp/*.png
rm figures/tmp_ray_tracer/*.png

CUDA_VISIBLE_DEVICES=4 python3.8 render_images.py

# rm figures/tmp/*.png
# rm figures/tmp_ray_tracer/*.png
