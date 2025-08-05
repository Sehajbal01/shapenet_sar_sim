import os
import math

# Number of GPUs to use — adjust this if you have more or fewer
K = 8

# This is the part of the dataset we want to process (cars_train)
subdataset = 'cars_train'

# Path to where the original object folders are stored
source_base = os.path.join('/workspace/data/srncars', subdataset)

# Grab all the object folder names (each one is a 3D car model)
obj_ids = sorted(os.listdir(source_base))

# Divide the object IDs evenly into K chunks (one for each GPU)
chunk_size = math.ceil(len(obj_ids) / K)
chunks = [obj_ids[i * chunk_size:(i + 1) * chunk_size] for i in range(K)]

# For each chunk, create a .txt file listing all the object IDs it should handle
for k, chunk in enumerate(chunks):
    with open(f'ID_chunk_{k}.txt', 'w') as f:
        f.write('\n'.join(chunk))

# Now run dataset.py in parallel on each chunk, assigning each one to its own GPU
for k in range(K):
    # CUDA_VISIBLE_DEVICES={k} locks the job to GPU {k}
    # We pass in the corresponding ID_chunk_k.txt file to dataset.py
    os.system(f'CUDA_VISIBLE_DEVICES={k} python dataset.py ID_chunk_{k}.txt &')

# This waits for all background jobs to finish before exiting
os.system('wait')
