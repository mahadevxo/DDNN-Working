import os
import numpy as np
import open3d as o3d
import trimesh
import torch
import time
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import platform

# Configuration
MODELNET40_PATH = "ModelNet40"
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View"
VIEWS = 12
AZIMUTH_STEP = 360 / VIEWS
WINDOW_SIZE = 500
USE_GPU = torch.cuda.is_available()

def off_to_obj(off_path, obj_path):
    """Optimized OFF to OBJ converter."""
    try:
        mesh = trimesh.load(off_path, file_type='off')
        mesh.export(obj_path)
        return True
    except Exception as e:
        print(f"Error converting {off_path}: {e}")
        return False

def batch_convert_off_to_obj(batch):
    """Process a batch of OFF to OBJ conversions."""
    results = []
    for off_path, obj_path in batch:
        results.append(off_to_obj(off_path, obj_path))
    return results

def convert_all_off_to_obj():
    """Convert all OFF files to OBJ using parallel processing."""
    conversion_tasks = []
    
    for category in os.listdir(MODELNET40_PATH):
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
                    off_path = os.path.join(split_path, file)
                    obj_path = os.path.join(obj_split_path, file.replace(".off", ".obj"))
                    conversion_tasks.append((off_path, obj_path))
    
    # Process in parallel batches
    batch_size = 100
    batches = [conversion_tasks[i:i+batch_size] for i in range(0, len(conversion_tasks), batch_size)]
    
    successful = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch_results in tqdm(executor.map(batch_convert_off_to_obj, batches), 
                                  total=len(batches), desc="Converting OFF to OBJ"):
            successful += sum(batch_results)
    
    print(f"Conversion complete: {successful}/{len(conversion_tasks)} successful")

def render_views_open3d(obj_path, save_dir, class_name, model_id):
    """Render views using Open3D visualization."""
    try:
        # Load the mesh with Open3D
        mesh = o3d.io.read_triangle_mesh(obj_path)
        
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError(f"Empty mesh: {obj_path}")
        
        # Center and normalize
        mesh.compute_vertex_normals()
        center = mesh.get_center()
        mesh.translate(-center)
        
        # Scale to unit size
        scale = np.max(np.abs(np.asarray(mesh.vertices)))
        mesh.scale(1/scale, np.zeros(3))
        
        # Set material for better rendering
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=WINDOW_SIZE, height=WINDOW_SIZE)
        vis.add_geometry(mesh)
        
        # Set rendering options
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([1.0, 1.0, 1.0])
        render_option.point_size = 1.0
        
        # Set view control
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Render views
        for i in range(VIEWS):
            # Set camera position for this view
            azim = i * AZIMUTH_STEP
            view_control.set_front([np.sin(np.radians(azim)), np.cos(np.radians(azim)), 0.5])
            view_control.set_up([0, 0, 1])
            view_control.set_zoom(0.8)
            
            # Update and render
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            image = vis.capture_screen_float_buffer(do_render=True)
            img_array = (np.asarray(image) * 255).astype(np.uint8)
            
            # Save image
            output_path = f"{save_dir}/{class_name}_{model_id}_view_{i}.png"
            Image.fromarray(img_array).save(output_path)
        
        # Clean up
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"Error rendering {obj_path}: {e}")
        return False

def process_model(args):
    """Process a single model (for parallel execution)."""
    class_name, split, model_name, base_paths = args
    model_id = model_name.split(".")[0]
    obj_path = os.path.join(base_paths['model_path'], class_name, split, model_name)
    output_dir = os.path.join(base_paths['output_path'], class_name, split, model_id)
    return render_views_open3d(obj_path, output_dir, class_name, model_id)

def process_modelnet40():
    """Process ModelNet40 models in parallel using Open3D."""
    # Collect all tasks
    tasks = []
    base_paths = {'model_path': MODELNET40_OBJ_PATH, 'output_path': OUTPUT_PATH}
    
    for class_name in sorted(os.listdir(MODELNET40_OBJ_PATH)):
        class_path = os.path.join(MODELNET40_OBJ_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for split in ["train", "test"]:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue
                
            for model_name in sorted(os.listdir(split_path)):
                if model_name.endswith(".obj"):
                    tasks.append((class_name, split, model_name, base_paths))
    
    print(f"Found {len(tasks)} models to process")
    
    # Determine optimal number of workers
    if platform.system() == 'Windows':
        # Windows has issues with too many processes
        num_workers = min(os.cpu_count() // 2, 8)
    else:
        num_workers = max(1, os.cpu_count() - 2)
    
    print(f"Processing with {num_workers} workers")
    
    # Process in parallel
    start_time = time.time()
    successful = 0
    total = len(tasks)
    
    # Process in batches to better manage memory
    batch_size = 100
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:min(i+batch_size, len(tasks))]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_model, batch), 
                total=len(batch),
                desc=f"Rendering batch {i//batch_size + 1}/{len(tasks)//batch_size + 1}"
            ))
        
        successful += sum(results)
        
    elapsed_time = time.time() - start_time
    print(f"Completed rendering {successful}/{total} models in {elapsed_time:.2f} seconds")
    print(f"Average time per model: {elapsed_time/total:.4f} seconds")

if __name__ == "__main__":
    # Make sure output directories exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(MODELNET40_OBJ_PATH, exist_ok=True)
    
    obj = input("Convert to .obj? Y/N: ")
    if obj.lower() in ["y", "yes"]:
        print("Converting .off files to .obj...")
        convert_all_off_to_obj()
        print("Conversion complete.")
    
    # Render views
    process_modelnet40()