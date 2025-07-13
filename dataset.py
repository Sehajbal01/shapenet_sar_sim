import os

# This is where we’ll build the new dataset structure
target_base = '/workspace/sehajdeepbal/create_dataset'

# Make sure all the main folders exist — even if we’re only filling cars_train for now
for split in ['cars_train', 'cars_val', 'cars_test']:
    os.makedirs(os.path.join(target_base, split), exist_ok=True)

# Loop through just the dataset we care about (cars_train)
for subdataset in ['cars_train']:
    # Define where the source data is and where we’re writing new output
    source_base = os.path.join('/workspace/data/srncars', subdataset)
    target_split_path = os.path.join(target_base, subdataset)

    # Grab all the object IDs (each is a folder)
    obj_ids = os.listdir(source_base)

    for obj_id in obj_ids:
        # Defining the full path to this object’s folder in both source and target
        src_obj_path = os.path.join(source_base, obj_id)
        tgt_obj_path = os.path.join(target_split_path, obj_id)

        # Inside the object folder, we’ll make a subfolder called "shapenet_sar"
        shapenet_sar_path = os.path.join(tgt_obj_path, 'shapenet_sar')
        os.makedirs(shapenet_sar_path, exist_ok=True)

        # Checking if this object has an "images" folder in the source
        img_dir = os.path.join(src_obj_path, 'images')
        if os.path.isdir(img_dir):
            # Loop through all the files in the images folder
            for fname in os.listdir(img_dir):
                if fname.endswith('.png'):
                    # Instead of copying the real image, we just create an empty placeholder with the same name (Not sure if this is what we want)
                    placeholder_path = os.path.join(shapenet_sar_path, fname)
                    open(placeholder_path, 'a').close()  # creates an empty .png file
