import os
import random
import shutil


source_root = "/home/mahadev/Desktop/modelnet40_images_new_12x"
destination_root = "test_set/"

os.makedirs(destination_root, exist_ok=True)

for class_folder in os.listdir(source_root):
    class_folder_path = os.path.join(source_root, class_folder)
    
    if not os.path.isdir(class_folder_path):
        continue

    test_folder = os.path.join(class_folder_path, "test")
    if not os.path.exists(test_folder):
        print(f"Test folder does not exist for class {class_folder}. Skipping.")
        continue

    all_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

    selected_files = random.sample(all_files, min(10, len(all_files)))

    #destination_class_folder = os.path.join(destination_root, class_folder)
    #os.makedirs(destination_class_folder, exist_ok=True)

    for file_name in selected_files:
        source_path = os.path.join(test_folder, file_name)
        #destination_path = os.path.join(destination_class_folder, file_name)
        shutil.copy(source_path, destination_root)

    print(f"Copied {len(selected_files)} files from {class_folder}/test to {destination_root}")

print("All files have been successfully copied!")