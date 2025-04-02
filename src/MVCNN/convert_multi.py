import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import multiprocessing as mp
import platform
import subprocess
import torch  # Added for NVIDIA GPU detection

# Paths
MODELNET40_PATH = "ModelNet40"
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View"
VIEWS = 12  # 12 views per model
AZIMUTH_STEP = 360 / VIEWS  # 30-degree steps


def off_to_obj(off_path, obj_path):
    """Converts a .off file to .obj format, handling both standard and non-standard headers."""
    with open(off_path, "r") as f:
        lines = f.readlines()
    
    # Skip empty lines and handle potential comments
    line_index = 0
    while line_index < len(lines) and (not lines[line_index].strip() or lines[line_index].strip().startswith('#')):
        line_index += 1

    # Check if we have a valid file
    if line_index >= len(lines) or not lines[line_index].strip():
        raise ValueError(f"Empty or invalid OFF file: {off_path}")
    
    # Read the header line
    header = lines[line_index].strip()
    
    if header.startswith("OFF"):
        # Standard OFF format
        parts = header.split()
        if len(parts) == 1:
            # Next line contains vertex and face counts
            line_index += 1
            if line_index >= len(lines):
                raise ValueError(f"Incomplete OFF file: {off_path}")
            counts = list(map(int, lines[line_index].strip().split()))
        else:
            # Header contains the counts directly
            counts = list(map(int, parts[1:]))
    else:
        # Non-standard OFF format where the first line contains counts
        counts = list(map(int, header.split()))
    
    # Ensure valid count values
    if len(counts) < 2:
        raise ValueError(f"Invalid OFF header: {header}")
    
    num_vertices, num_faces = counts[0], counts[1]
    line_index += 1

    # Read vertices
    vertices = []
    for _ in range(num_vertices):
        while line_index < len(lines) and (not lines[line_index].strip() or lines[line_index].startswith('#')):
            line_index += 1  # Skip empty/comment lines

        if line_index >= len(lines):
            raise ValueError(f"Incomplete vertex data in OFF file: {off_path}")
        
        vertices.append(list(map(float, lines[line_index].strip().split())))
        line_index += 1
    
    # Read faces
    faces = []
    for _ in range(num_faces):
        while line_index < len(lines) and (not lines[line_index].strip() or lines[line_index].startswith('#')):
            line_index += 1  # Skip empty/comment lines

        if line_index >= len(lines):
            raise ValueError(f"Incomplete face data in OFF file: {off_path}")
        
        face_data = list(map(int, lines[line_index].strip().split()))
        
        if face_data[0] >= len(face_data) - 1:
            # Standard OFF format with vertex count as first number
            faces.append(face_data[1:])
        else:
            # Some OFF files omit the count, use as-is
            faces.append(face_data)

        line_index += 1

    # Write to OBJ format
    with open(obj_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            if len(face) >= 3:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")  # Convert 0-based to 1-based indexing

def convert_all_off_to_obj():
    successful = 0
    failed = 0
    for category in tqdm(os.listdir(MODELNET40_PATH), desc="Converting .off to .obj"):
        cat_path = os.path.join(MODELNET40_PATH, category)
        if not os.path.isdir(cat_path):
            continue
            
        obj_cat_path = os.path.join(MODELNET40_OBJ_PATH, category)
        os.makedirs(obj_cat_path, exist_ok=True)

        for split in ["train", "test"]:
            split_path = os.path.join(cat_path, split)
            if not os.path.exists(split_path):
                continue
                
            obj_split_path = os.path.join(obj_cat_path, split)
            os.makedirs(obj_split_path, exist_ok=True)

            for file in os.listdir(split_path):
                if file.endswith(".off"):
                    obj_file = os.path.join(obj_split_path, file.replace(".off", ".obj"))
                    try:
                        off_to_obj(os.path.join(split_path, file), obj_file)
                        successful += 1
                    except Exception as e:
                        failed += 1
                        print(f"Error converting {file}: {e}")
    
    print(f"Conversion complete: {successful} successful, {failed} failed")

def render_views(obj_path, save_dir, class_name, model_id):
    """Renders 12 views of a 3D model using Matplotlib and saves as PNG."""
    try:
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
        # Center and normalize the mesh
        vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        # Get the max dimension for proper scaling
        max_dim = max(vertices.max(axis=0) - vertices.min(axis=0))
        vertices = vertices / max_dim  # Normalize to [-0.5, 0.5] range
        
        faces = mesh.faces
        
        # Set matplotlib to use non-interactive backend
        plt.switch_backend('Agg')
        
        # Create figure with equal aspect ratio
        fig = plt.figure(figsize=(5, 5), dpi=100)  # Increased figure size for better quality
        ax = fig.add_subplot(111, projection='3d')
        
        # Create uniform limits for all axes to ensure correct proportions
        axis_limit = 0.6  # Slightly larger than the normalized object size
        
        # Prepare directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Pre-compute poly3d
        poly3d = [[vertices[vert] for vert in face] for face in faces]
        
        for i in range(VIEWS):
            ax.clear()
            
            # Add mesh collection
            mesh_collection = Poly3DCollection(poly3d, facecolors='lightgray', edgecolors='k', linewidths=0.1)
            ax.add_collection3d(mesh_collection)
            
            # Set uniform limits for all axes
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_zlim(-axis_limit, axis_limit)
            
            # Ensure equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=30, azim=i * AZIMUTH_STEP)
            ax.axis("off")
            
            # Save figure with tight layout
            plt.tight_layout()
            fig.savefig(f"{save_dir}/{class_name}_{model_id}_view_{i}.png", bbox_inches='tight', pad_inches=0)
            
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error processing {obj_path}: {e}")
        return False

def process_single_model(args):
    """Process a single model for multiprocessing."""
    class_name, split, model_name, base_paths = args
    model_id = model_name.split(".")[0]
    obj_path = os.path.join(base_paths['model_path'], class_name, split, model_name)
    output_dir = os.path.join(base_paths['output_path'], class_name, split, model_id)
    return render_views(obj_path, output_dir, class_name, model_id)

def detect_optimal_cores():
    """Detect the optimal number of cores for the current system."""
    total_cores = mp.cpu_count()

    # For Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        try:
            # Try to get performance cores
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error detecting performance cores: {e}")
            # Fallback to half of total cores
            
            total_cores = mp.cpu_count()
            return max(4, min(16, total_cores // 2)) 

    # For NVIDIA GPU systems, use a bit fewer cores than available
    # This prevents overloading the system while GPU is active
    return max(4, total_cores - 2)

def setup_gpu_environment():
    """Set up the environment for GPU usage if available."""
    # Check for NVIDIA GPU
    nvidia_gpu_available = False
    
    try:
        if torch.cuda.is_available():
            nvidia_gpu_available = True
            print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
            # Set environment variables for NVIDIA GPUs
            os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            os.environ["OPENBLAS_NUM_THREADS"] = "2"  # Reduce OpenBLAS threads
            os.environ["MKL_NUM_THREADS"] = "2"  # Reduce MKL threads
            os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # Reduce macOS BLAS threads
            os.environ["NUMEXPR_NUM_THREADS"] = "2"  # Reduce NumExpr threads
    except Exception as e:
        print(f"Error checking for NVIDIA GPU: {e}")
    
    # For Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Apple Silicon detected, optimizing for M-series chips")
    
    return nvidia_gpu_available

def collect_tasks():
    """Collect all rendering tasks."""
    tasks = []
    base_paths = {'model_path': MODELNET40_PATH, 'output_path': OUTPUT_PATH}
    
    # Count total files
    total_files = 0
    for class_name in sorted(os.listdir(MODELNET40_PATH)):
        class_path = os.path.join(MODELNET40_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for split in ["train", "test"]:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue
                
            for model_name in sorted(os.listdir(split_path)):
                if model_name.endswith(".obj"):
                    total_files += 1
                    tasks.append((class_name, split, model_name, base_paths))
    
    return tasks, total_files

def process_modelnet40():
    """Processes ModelNet40's .obj files and renders 12 views per model using multiprocessing."""
    # Set up GPU environment if available
    nvidia_gpu = setup_gpu_environment()

    # Get all tasks
    tasks, total_files = collect_tasks()

    print(f"Found {total_files} models to process")

    # Determine optimal multiprocessing approach
    if platform.system() == 'Windows':
        mp_context = mp.get_context('spawn')
    else:
        # 'fork' is more efficient on Linux/NVIDIA systems, 'spawn' safer on macOS
        mp_method = 'spawn' if platform.system() == 'Darwin' else 'fork'
        mp_context = mp.get_context(mp_method)

    # Number of processes - optimized for the system
    num_processes = detect_optimal_cores()

    print(f"Processing using {num_processes} cores with method: {mp_context.get_start_method()}")

    # Adjust chunk size based on system
    chunk_size = max(100, total_files // (num_processes * 4))

    # Use context manager for proper resource cleanup
    with mp_context.Pool(processes=min(8, num_processes)) as pool:
        # Process with tqdm for progress tracking
        results = list(tqdm(
            pool.imap(process_single_model, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="Rendering views",
            unit="model"
        ))

    # Report completion statistics
    successful = sum(r is True for r in results)
    print(f"Completed rendering {successful}/{total_files} models successfully")

if __name__ == "__main__":
    
    obj = input("Convert to .obj? Y/N: ")
    if obj.lower() in ["y", "yes"]:
        if not os.path.exists(MODELNET40_OBJ_PATH):
            os.makedirs(MODELNET40_OBJ_PATH, exist_ok=True)
        print("Converting .off files to .obj...")
        convert_all_off_to_obj()
        MODELNET40_PATH = MODELNET40_OBJ_PATH
        print("Converted .off files to .obj.")

    else:
        MODELNET40_PATH ='ModelNet40_OBJ'
        print("Skipping .obj conversion.")
    # Render views

    process_modelnet40()