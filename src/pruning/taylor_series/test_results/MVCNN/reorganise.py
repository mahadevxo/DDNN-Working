import os
import shutil

# Define source and destination directories
source_root = "ModelNet40-12View"
dest_root = "modelnet40-12view_final"

# Ensure destination directory exists
os.makedirs(dest_root, exist_ok=True)

# Iterate over each category (bed, bench, etc.)
for category in os.listdir(source_root):
    category_path = os.path.join(source_root, category)
    if not os.path.isdir(category_path):
        continue
    
    # Iterate over train/test subdirectories
    for split in ["train", "test"]:
        split_path = os.path.join(category_path, split)
        if not os.path.isdir(split_path):
            continue
        
        # Iterate over object instances (bed_0516, bed_0526, etc.)
        for obj_instance in os.listdir(split_path):
            obj_path = os.path.join(split_path, obj_instance)
            if not os.path.isdir(obj_path):
                continue
            
            # Iterate over all view images
            for view_file in os.listdir(obj_path):
                if view_file.endswith(".png"):
                    view_index = view_file.split("_")[1].split(".")[0]  # Extract view index
                    new_filename = f"{obj_instance}_shaded_{view_index}.png"
                    
                    # Define destination path
                    category_dest_path = os.path.join(dest_root, category, split)
                    os.makedirs(category_dest_path, exist_ok=True)
                    
                    # Copy file
                    shutil.copy(os.path.join(obj_path, view_file), os.path.join(category_dest_path, new_filename))
                    
print("Dataset reorganization complete.")
os.system(f"rm -rf {source_root}")
print("Original dataset directory removed.")
os.system(f"mv {dest_root} {source_root}")
print("Dataset directory renamed.")