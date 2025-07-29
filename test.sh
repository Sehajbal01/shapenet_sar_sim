#! /bin/bash
rm figures/tmp/*.png
# CUDA_VISIBLE_DEVICES=3 python mesh.py
CUDA_VISIBLE_DEVICES=3 python render_images.py