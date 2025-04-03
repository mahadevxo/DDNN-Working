import torch
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import os
import imageio
from tqdm import tqdm

# Paths
MODELNET40_PATH = "ModelNet40"
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View_pytorch3d"
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
    

def render_views(obj_path, output_dir, device="cuda"):
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    mesh = load_objs_as_meshes([obj_path], device=device)
    
    # Define cameras (12 views, 30-degree increments, 30-degree incline)
    angles = torch.linspace(0, 360, steps=13)[:-1]  # Avoid duplicate first/last
    elev = torch.full_like(angles, 30.0)
    dist = torch.full_like(angles, 2.5)  # Adjust as needed
    R, T = look_at_view_transform(dist, elev, angles)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Renderer setup
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    
    # Render views and save images
    images = renderer(meshes_world=mesh)
    for i, img in enumerate(images):
        img = img[...,:3].cpu().numpy()  # Remove alpha channel
        imageio.imwrite(os.path.join(output_dir, f"view_{i:02d}.png"), (img * 255).astype('uint8'))

def start_render():
    for category in os.listdir(MODELNET40_OBJ_PATH):
        print("*"*20, category, "*"*20)
        
        cat_path = os.path.join(MODELNET40_OBJ_PATH, category)
        if not os.path.isdir(cat_path):
            continue
        print(f"Rendering category: {category}")
            
        for split in ["train", "test"]:
            split_path = os.path.join(cat_path, split)
            if not os.path.exists(split_path):
                continue
                
            for file in os.listdir(split_path):
                if file.endswith(".obj"):
                    obj_file = os.path.join(split_path, file)
                    out_path = os.path.join(OUTPUT_PATH, category, split, file.replace(".obj", ""))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    render_views(obj_file, out_path)
                    print(f"Rendered {obj_file} to {out_path}")

if __name__ == "__main__":
    check = input("Do you want to convert .off files to .obj? (y/n): ")
    if check.lower() in ['y', 'yes']:
        if not os.path.exists(MODELNET40_OBJ_PATH):
            os.makedirs(MODELNET40_OBJ_PATH)
        convert_all_off_to_obj()
    
    MODELNET40_PATH = MODELNET40_OBJ_PATH
    start_render()
    print("Rendering complete.")