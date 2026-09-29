#! /bin/bash

GPU_NUM=6
# cap the gpu's power draw, in watts
sudo nvidia-smi -i $GPU_NUM -pl 200

mkdir -p figures
# clear the whole folder, so nothing from an earlier run is mistaken for this one's output.
rm -f figures/*

# # dataset generation test run, range angle only
# strip config.json's whole-line // comments first, as config.py does -- jq can't parse them
NUM_MODELS=$(ls "$(sed 's|^\s*//.*$||' config.json | jq -r .srn_cars_dir)/cars_test" | wc -l)
NUM_CHUNKS=$((NUM_MODELS))
CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
    generate_dataset.py -num_chunks $NUM_CHUNKS -chunk_id 5 -test_run -gif 

# the sar paper figures, on config.json's sar_baseline obj_id/azimuth_deg/elevation_deg -- set to
# the test run's chunk 5 object at az 0 el 33, to look into its striations. Only the n_ray sweep is
# on in PAPER_EXPERIMENTS
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     paper_figures.py

# # the sonar paper figures
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 \
#     sonar_paper_figures.py

# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 sonar_paper_figures.py
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 paper_figures.py
# CUDA_VISIBLE_DEVICES=$GPU_NUM /workspace/berian/miniconda3/envs/sarrender/bin/python3.8 debug_side_scan.py
