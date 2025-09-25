#! /bin/bash

rm figures/tmp/*.png
rm figures/*.png

# CUDA_VISIBLE_DEVICES=3 python mesh.py
CUDA_VISIBLE_DEVICES=2 python3.8 render_images.py
# CUDA_VISIBLE_DEVICES=3 python utils.py

rm figures/tmp/*.png
