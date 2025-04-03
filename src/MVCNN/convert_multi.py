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
import time
from concurrent.futures import ThreadPoolExecutor
# Modify imports to remove PyCUDA GL dependency
from PIL import Image
import torch.nn as nn

# Paths
MODELNET40_PATH = "ModelNet40"
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View"
VIEWS = 12  # 12 views per model
AZIMUTH_STEP = 360 / VIEWS  # 30-degree steps

# OpenGL configuration
WINDOW_SIZE = 500


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

def setup_gpu_environment():
    """Set up the environment for GPU usage if available."""
    # Check for NVIDIA GPU
    nvidia_gpu_available = False
    
    try:
        if torch.cuda.is_available():
            nvidia_gpu_available = True
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"NVIDIA GPU detected: {device_name} (Total GPUs: {device_count})")
            
            # For RTX 4090, optimize CUDA settings
            if "RTX" in device_name:
                print(f"RTX GPU detected: {device_name} - Optimizing settings")
                # Set environment variables for NVIDIA RTX GPUs
                os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable JIT caching
                os.environ["CUDA_AUTO_BOOST"] = "1"  # Enable auto boost
            
            # Set common CUDA environment variables
            os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Reduce OpenBLAS threads
            os.environ["MKL_NUM_THREADS"] = "4"  # Reduce MKL threads
            os.environ["NUMEXPR_NUM_THREADS"] = "4"  # Reduce NumExpr threads
            
            # Configure CUDA memory allocation
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 80% GPU memory
            
            # Print GPU memory info
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"GPU Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
    except Exception as e:
        print(f"Error configuring NVIDIA GPU: {e}")
    
    # For Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Apple Silicon detected, optimizing for M-series chips")
    
    return nvidia_gpu_available

class GpuMeshRasterizer(nn.Module):
    """PyTorch-based mesh rasterizer that uses GPU acceleration without OpenGL."""
    def __init__(self, image_size=WINDOW_SIZE):
        super().__init__()
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def to_device(self, tensor):
        """Helper to move tensors to the correct device."""
        if isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(self.device)
        return tensor.to(self.device)
        
    def render_view(self, vertices, faces, elevation=30, azimuth=0):
        """Render a view of the mesh using PyTorch operations."""
        # Convert inputs to tensors and move to GPU
        vertices_tensor = self.to_device(vertices).float()
        faces_tensor = self.to_device(faces).long()
        
        # Create rotation matrices for camera view
        elev_rad = torch.tensor(elevation * np.pi / 180.0).to(self.device)
        azim_rad = torch.tensor(azimuth * np.pi / 180.0).to(self.device)
        
        # Create rotation matrices
        rot_y = torch.tensor([
            [torch.cos(azim_rad), 0, torch.sin(azim_rad)],
            [0, 1, 0],
            [-torch.sin(azim_rad), 0, torch.cos(azim_rad)]
        ]).to(self.device)
        
        rot_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(elev_rad), -torch.sin(elev_rad)],
            [0, torch.sin(elev_rad), torch.cos(elev_rad)]
        ]).to(self.device)
        
        # Apply rotations
        vertices_rotated = torch.matmul(vertices_tensor, torch.matmul(rot_y, rot_x))
        
        # Move vertices slightly back for better perspective
        vertices_rotated[:, 2] += 2.0
        
        # Simple projection to image plane (z=1)
        vertices_2d = vertices_rotated[:, :2] / vertices_rotated[:, 2:3]
        
        # Scale to image coordinates
        vertices_2d = (vertices_2d + 1) * self.image_size / 2
        
        # Create an empty image
        image = torch.ones((self.image_size, self.image_size, 3), device=self.device) * 255
        
        # Get triangles for rasterization
        triangles = vertices_2d[faces_tensor]
        
        # Simple flat shading - prepare face normals in 3D space
        v0, v1, v2 = [vertices_rotated[faces_tensor[:, i]] for i in range(3)]
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
        
        # Lighting direction
        light_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        # Compute lighting (simple diffuse)
        diffuse = torch.matmul(face_normals, light_dir)
        diffuse = torch.clamp(diffuse, 0, 1) * 180 + 75  # Scale to reasonable gray values
        
        # Create colors for each face
        face_colors = torch.ones((len(faces_tensor), 3), device=self.device) * diffuse.view(-1, 1)
        
        # Rasterize triangles
        with torch.no_grad():
            # Sort faces by z-depth for basic z-buffering
            z_depths = vertices_rotated[faces_tensor, 2].mean(dim=1)
            _, indices = torch.sort(z_depths, descending=True)
            
            # Process faces from back to front
            for idx in indices:
                try:
                    tri = triangles[idx].long()
                    color = face_colors[idx]
                    
                    # Get bounding box
                    min_x = max(0, torch.min(tri[:, 0]).long())
                    max_x = min(self.image_size - 1, torch.max(tri[:, 0]).long() + 1)
                    min_y = max(0, torch.min(tri[:, 1]).long())
                    max_y = min(self.image_size - 1, torch.max(tri[:, 1]).long() + 1)
                    
                    # Skip if triangle is outside view
                    if max_x <= min_x or max_y <= min_y:
                        continue
                    
                    # Create grid of points
                    y, x = torch.meshgrid(
                        torch.arange(min_y, max_y, device=self.device),
                        torch.arange(min_x, max_x, device=self.device),
                        indexing='ij'
                    )
                    points = torch.stack([x.flatten(), y.flatten()], dim=1)
                    
                    # Check if points are inside triangle using barycentric coordinates
                    v0 = tri[0]
                    v1 = tri[1]
                    v2 = tri[2]
                    
                    area = 0.5 * torch.abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - 
                                           (v2[0] - v0[0]) * (v1[1] - v0[1]))
                    
                    # Skip degenerate triangles
                    if area < 1e-5:
                        continue
                        
                    # Calculate barycentric coordinates for each point
                    w0 = 0.5 * torch.abs((v1[0] - points[:, 0]) * (v2[1] - points[:, 1]) - 
                                        (v2[0] - points[:, 0]) * (v1[1] - points[:, 1])) / (area + 1e-8)
                    w1 = 0.5 * torch.abs((v2[0] - points[:, 0]) * (v0[1] - points[:, 1]) - 
                                        (v0[0] - points[:, 0]) * (v2[1] - points[:, 1])) / (area + 1e-8)
                    w2 = 1 - w0 - w1
                    
                    # Points inside triangle have all weights between 0 and 1
                    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (w0 <= 1) & (w1 <= 1) & (w2 <= 1)
                    
                    # Get pixel coordinates of points inside triangle
                    y_inside = points[mask, 1].long()
                    x_inside = points[mask, 0].long()
                    
                    # Draw the pixels
                    if len(y_inside) > 0:
                        image[y_inside, x_inside] = color
                except Exception as e:
                    continue  # Skip problematic triangles
                    
        # Convert to numpy array
        image_np = image.cpu().numpy().astype(np.uint8)
        return image_np

def gpu_render_views(obj_path, save_dir, class_name, model_id, renderer=None):
    """Renders 12 views of a 3D model using GPU acceleration."""
    try:
        mesh = trimesh.load(obj_path, process=False, force='mesh')

        if mesh.vertices.size == 0:
            raise ValueError(f"Mesh has no vertices: {obj_path}")
        if mesh.faces.size == 0:
            raise ValueError(f"Mesh has no faces: {obj_path}")

        # Center and normalize the mesh
        vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        max_dim = max(vertices.max(axis=0) - vertices.min(axis=0))
        vertices = vertices / (max_dim * 0.8)  # Normalize to fit in view
        
        faces = mesh.faces
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize renderer if not provided
        if renderer is None:
            renderer = GpuMeshRasterizer()
        
        # Render multiple views
        for i in range(VIEWS):
            azim = i * AZIMUTH_STEP
            image = renderer.render_view(vertices, faces, elevation=30, azimuth=azim)
            
            # Save the image
            output_path = f"{save_dir}/{class_name}_{model_id}_view_{i}.png"
            Image.fromarray(image).save(output_path)
            
        return True
    except Exception as e:
        print(f"Error processing {obj_path}: {e}")
        return False

def process_single_model_gpu(args):
    """Process a single model for multiprocessing with GPU rendering."""
    class_name, split, model_name, base_paths, renderer = args
    model_id = model_name.split(".")[0]
    obj_path = os.path.join(base_paths['model_path'], class_name, split, model_name)
    output_dir = os.path.join(base_paths['output_path'], class_name, split, model_id)
    return gpu_render_views(obj_path, output_dir, class_name, model_id, renderer)

def detect_optimal_cores():
    """Detect the optimal number of cores for the current system."""
    total_cores = mp.cpu_count()
    print(f"Total CPU cores detected: {total_cores}")
    
    # Check if NVIDIA GPU is present
    if torch.cuda.is_available():
        # For systems with powerful GPUs like RTX 4090,
        # we use fewer CPU cores for rendering as the GPU is doing most work
        device_name = torch.cuda.get_device_name(0)
        if "RTX" in device_name:
            print(f"Optimizing core usage for {device_name}")
            # For RTX GPUs, use fewer cores as the GPU is doing most of the work
            # This prevents CPU bottlenecks and memory contention
            return min(8, max(4, total_cores // 4))
    
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
            return max(4, min(16, total_cores // 2)) 

    # Default for other systems
    return max(4, total_cores - 4)

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

def batch_process_models(tasks, batch_size=50):
    """Process models in batches to optimize GPU memory usage."""
    try:
        # Create a single renderer to be reused
        renderer = GpuMeshRasterizer()
        renderer.to(torch.device('cuda'))
    except Exception as e:
        print(f"Error initializing GPU renderer: {e}")
        return 0, len(tasks)
    
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    
    successful = 0
    total = len(tasks)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_tasks = tasks[batch_start:batch_end]
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for task in batch_tasks:
                class_name, split, model_name, base_paths = task
                args = (class_name, split, model_name, base_paths, renderer)
                futures.append(executor.submit(process_single_model_gpu, args))
            
            # Collect results
            for future in futures:
                try:
                    if future.result():
                        successful += 1
                except Exception as e:
                    print(f"Error in batch processing: {e}")
        
        # Force CUDA synchronization and clear GPU memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return successful, total

def process_modelnet40():
    """Processes ModelNet40's .obj files and renders 12 views per model using GPU acceleration."""
    # Set up GPU environment
    nvidia_gpu = setup_gpu_environment()
    
    # Get all tasks
    tasks, total_files = collect_tasks()
    print(f"Found {total_files} models to process")
    
    start_time = time.time()
    
    if nvidia_gpu:
        print("Using GPU acceleration for rendering")
        successful, total = batch_process_models(tasks)
    else:
        print("No NVIDIA GPU detected. Using CPU rendering instead.")
        # Determine optimal multiprocessing approach
        if platform.system() == 'Windows':
            mp_context = mp.get_context('spawn')
        else:
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
            successful = sum(r is True for r in results)
            total = len(results)
    
    elapsed_time = time.time() - start_time
    
    # Report completion statistics
    print(f"Completed rendering {successful}/{total} models successfully")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per model: {elapsed_time/total:.4f} seconds")

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