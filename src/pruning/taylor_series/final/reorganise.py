import os
import shutil

# Define source and destination directories
source_root = "ModelNet40_12View/"
dest_root = "ModelNet40-12View"

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
        
        # Check whether split_path contains directories (object instances)
        if any(os.path.isdir(os.path.join(split_path, item)) for item in os.listdir(split_path)):
            # Process when object instance folders exist
            for obj_instance in os.listdir(split_path):
                obj_path = os.path.join(split_path, obj_instance)
                if not os.path.isdir(obj_path):
                    continue
                
                # Iterate over all view images
                for view_file in os.listdir(obj_path):
                    if view_file.endswith(".png"):
                        view_index = view_file.split("_")[1].split(".")[0]  # Extract view index
                        new_filename = f"{obj_instance}_shaded_{view_index}.png"
                        
                        # Define destination path including object instance folder
                        category_dest_path = os.path.join(dest_root, category, split, obj_instance)
                        os.makedirs(category_dest_path, exist_ok=True)
                        
                        # Copy file
                        shutil.copy(os.path.join(obj_path, view_file), os.path.join(category_dest_path, new_filename))
        else:
            # Process when view images are directly under the split folder
            for view_file in os.listdir(split_path):
                if view_file.endswith(".png"):
                    # Extract object instance from file name
                    # Example: "airplane_0627_shaded_0.png" gives "airplane_0627"
                    obj_instance = view_file.split("_shaded")[0]
                    view_index = view_file.split("_shaded_")[1].split(".")[0]
                    new_filename = f"{obj_instance}_shaded_{view_index}.png"
                    
                    category_dest_path = os.path.join(dest_root, category, split, obj_instance)
                    os.makedirs(category_dest_path, exist_ok=True)
                    shutil.copy(os.path.join(split_path, view_file), os.path.join(category_dest_path, new_filename))
                    
print("Dataset reorganization complete.")