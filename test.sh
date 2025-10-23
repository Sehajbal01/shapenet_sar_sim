#! /bin/bash

rm figures/tmp/*.png

CUDA_VISIBLE_DEVICES=0 python3.8 render_images.py

rm figures/tmp/*.png
