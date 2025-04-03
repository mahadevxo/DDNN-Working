import os
import torch
import numpy as np
import imageio
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    look_at_view_transform,
    PointLights,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVPerspectiveCameras

# Input and output directories
INPUT_DIR = "ModelNet40_OBJ"
OUTPUT_DIR = "ModelNet40_12View"

# Camera settings
AZIMUTHS = np.linspace(0, 330, 12)  # 12 views (every 30 degrees)le
ELEVATION = 30  # Camera elevation in degrees
DISTANCE = 3  # Distance from object

# Initialize PyTorch3D renderer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_renderer(image_size=224):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,  # ✅ Use naive rasterization (fixes overflow issue)
        max_faces_per_bin=500000  # ✅ Increase bin limit to prevent overflow
    )
    
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    cameras = FoVPerspectiveCameras(device=device)  

    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,  
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, lights=lights, cameras=cameras)  
    )

renderer = create_renderer(image_size=1024)

def normalize_mesh(mesh):
    """ Normalize mesh to fit within a unit sphere and assign dummy textures. """
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)  
    verts = verts - center  
    scale = 1.0 / torch.max(torch.norm(verts, dim=1))  
    verts = verts * scale
    
    faces = mesh.faces_packed()
    
    # Ensure a consistent "up" direction (align with Z-axis)
    _, _, V = torch.svd(verts)  # SVD for orientation normalization
    rotation = V[:, [2, 1, 0]]  # Swap axes if necessary
    verts = verts @ rotation.T  # Apply rotation
    
    # Assign white color to vertices (since textures are missing)
    colors = torch.ones_like(verts).unsqueeze(0).to(device)  # Shape: (1, num_verts, 3)
    textures = TexturesVertex(verts_features=colors)

    return Meshes(verts=[verts], faces=[faces], textures=textures).to(device)

# Process each object in ModelNet40
for category in tqdm(os.listdir(INPUT_DIR), desc="Processing Categories"):
    for split in ["train", "test"]:
        category_path = os.path.join(INPUT_DIR, category, split)
        if not os.path.isdir(category_path):
            continue
        
        output_category_path = os.path.join(OUTPUT_DIR, category, split)
        os.makedirs(output_category_path, exist_ok=True)

        model_files = [f for f in os.listdir(category_path) if f.endswith(".obj")]

        for model_file in tqdm(model_files, desc=f"Rendering {category}/{split}", leave=False):
            model_path = os.path.join(category_path, model_file)
            output_model_path = os.path.join(output_category_path, model_file[:-4])  
            
            os.makedirs(output_model_path, exist_ok=True)

            # Load and normalize mesh
            try:
                mesh = load_objs_as_meshes([model_path], device=device)
                mesh = normalize_mesh(mesh)  # ✅ Now it has vertex colors
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue

            # Render 12 views
            for i, azimuth in enumerate(AZIMUTHS):
                R, T = look_at_view_transform(DISTANCE, -ELEVATION, azimuth, device=device)
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

                image = renderer(mesh.extend(len(R)), cameras=cameras)  
                image = image[0, ..., :3].detach().cpu().numpy()  

                output_file = os.path.join(output_model_path, f"view_{i}.png")
                imageio.imwrite(output_file, (image * 255).astype(np.uint8))

print("✅ Rendering complete! Check ModelNet40_12View/")