import os
import shutil


# Base paths
source_base = '/workspace/data/srncars/cars_train'
target_base = '/workspace/sehajdeepbal/create_dataset'
# Creating base directory and subfolders
os.makedirs(os.path.join(target_base, 'cars_train'), exist_ok=True)
os.makedirs(os.path.join(target_base, 'cars_val'), exist_ok=True)
os.makedirs(os.path.join(target_base, 'cars_test'), exist_ok=True)

# object folders
obj_ids = os.listdir(source_base)

for obj_id in obj_ids:
    src_obj_path = os.path.join(source_base, obj_id)
    tgt_obj_path = os.path.join(target_base, 'cars_train', obj_id)
    
    # Creating target object directory and 'shapenet_sar' subfolder
    shapenet_sar_path = os.path.join(tgt_obj_path, 'shapenet_sar')
    os.makedirs(shapenet_sar_path, exist_ok = True)
    # shapenet_sar_path = os.makedirs(os.path.join(tgt_obj_path, 'shapenet_sar'), exist_ok=True) ?
    # Sourcing image directory (contains .png files)
    img_dir = os.path.join(src_obj_path, 'images')
    if os.path.isdir(img_dir):
        for fname in os.listdir(img_dir):
            if fname.endswith('.png'):
                src_img = os.path.join(img_dir, fname)
                dst_img = os.path.join(shapenet_sar_path, fname)
                shutil.copy2(src_img, dst_img)
                
