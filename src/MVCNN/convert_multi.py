import os
import torch
import numpy as np
import imageio
import trimesh
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    look_at_view_transform,
    PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVPerspectiveCameras

# Input and output directories
INPUT_DIR = "ModelNet40"
OUTPUT_DIR = "ModelNet40_12View"

# Camera settings
AZIMUTHS = np.linspace(0, 330, 12)  # 12 views (every 30 degrees)
ELEVATION = 30  # Camera elevation in degrees
DISTANCE = 2.5  # Distance from object

# Initialize PyTorch3D renderer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_renderer(image_size=224):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights)
    )

renderer = create_renderer()

def normalize_mesh(mesh):
    """ Normalize mesh to fit within a unit sphere. """
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)  # Compute centroid
    verts = verts - center  # Center the object
    scale = 1.0 / torch.max(torch.norm(verts, dim=1))  # Scale to unit size
    verts = verts * scale
    return Meshes(verts=[verts], faces=[mesh.faces_packed()]).to(device)

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
            output_model_path = os.path.join(output_category_path, model_file[:-4])  # Remove .obj
            
            os.makedirs(output_model_path, exist_ok=True)

            # Load and normalize mesh
            try:
                mesh = load_objs_as_meshes([model_path], device=device)
                mesh = normalize_mesh(mesh)
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue

            # Render 12 views
            for i, azimuth in enumerate(AZIMUTHS):
                R, T = look_at_view_transform(DISTANCE, ELEVATION, azimuth, device=device)
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

                image = renderer(mesh.extend(1), cameras=cameras)
                image = image[0, ..., :3].cpu().numpy()  # Extract RGB

                output_file = os.path.join(output_model_path, f"view_{i}.png")
                imageio.imwrite(output_file, (image * 255).astype(np.uint8))

print("✅ Rendering complete! Check ModelNet40_12View/")