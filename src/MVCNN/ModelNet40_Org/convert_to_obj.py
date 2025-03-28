import os
import multiprocessing
import sys
import psutil
import subprocess
import trimesh
from tqdm import tqdm

def set_performance_cores():
    """Set process affinity to use only performance cores (P-Cores) on macOS."""
    try:
        subprocess.run(["taskpolicy", "-c", str(os.getpid())], check=True)
    except Exception as e:
        print(f"Could not set P-Cores: {e}")

def process_category(category, input_dir, output_dir):
    """Convert all OFF files in a category using P-Cores, with progress tracking."""
    cat_in_path = os.path.join(input_dir, category)
    cat_out_path = os.path.join(output_dir, category)

    for split in ["train", "test"]:
        split_in_path = os.path.join(cat_in_path, split)
        split_out_path = os.path.join(cat_out_path, split)

        if not os.path.isdir(split_in_path):
            continue

        os.makedirs(split_out_path, exist_ok=True)

        off_files = [f for f in os.listdir(split_in_path) if f.endswith(".off")]

        # tqdm progress bar (silent by default, updates only on completion)
        with tqdm(total=len(off_files), desc=f"Processing {category}/{split}", unit="file", disable=False) as pbar:
            for file in off_files:
                input_off = os.path.join(split_in_path, file)
                output_obj = os.path.join(split_out_path, file.replace(".off", ".obj"))

                try:
                    mesh = trimesh.load(input_off, file_type="off")
                    mesh.export(output_obj)
                except Exception as e:
                    print(f"Error converting {input_off}: {e}")

                pbar.update(1)  # Update progress bar

if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("fork", force=True)
        set_performance_cores()

    input_dir = "./ModelNet40"
    output_dir = "./ModelNet40_OBJ"
    os.makedirs(output_dir, exist_ok=True)

    categories = [entry.name for entry in os.scandir(input_dir) if entry.is_dir()]

    with multiprocessing.Pool(processes=psutil.cpu_count(logical=False)) as pool:
        pool.starmap(process_category, [(category, input_dir, output_dir) for category in categories])