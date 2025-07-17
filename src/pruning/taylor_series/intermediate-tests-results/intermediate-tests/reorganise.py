import os
import shutil

# Define source and destination directories
source_root = "ModelNet40_12View/"
dest_root = "ModelNet40-12View/"

os.makedirs(dest_root, exist_ok=True)

for category in os.listdir(source_root):
    category_path = os.path.join(source_root, category)
    if not os.path.isdir(category_path):
        continue

    for split in ["train", "test"]:
        split_path = os.path.join(category_path, split)
        if not os.path.isdir(split_path):
            continue
        
        # Process view images directly under the split folder (no subfolders)
        for view_file in os.listdir(split_path):
            if view_file.endswith(".png"):
                # Example: "airplane_0627_shaded_0.png" gives "airplane_0627"
                obj_instance = view_file.split("_shaded")[0]
                view_index = view_file.split("_shaded_")[1].split(".")[0]
                new_filename = f"{obj_instance}_shaded_{view_index}.png"
                
                category_dest_path = os.path.join(dest_root, category, split, obj_instance)
                os.makedirs(category_dest_path, exist_ok=True)
                shutil.copy(os.path.join(split_path, view_file), os.path.join(category_dest_path, new_filename))
                    
print("Dataset reorganization complete.")