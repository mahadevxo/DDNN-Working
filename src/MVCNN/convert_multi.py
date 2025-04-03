import os
import numpy as np
import moderngl
import pyassimp
import imageio
from PIL import Image
from pyassimp import load
from tqdm import tqdm



# Paths
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View"
VIEWS = 12  # 12 views per model
AZIMUTH_STEP = 360 / VIEWS  # 30-degree steps

# OpenGL Initialization with EGL (NO GLFW)
def create_context():
    try:
        ctx = moderngl.create_context(standalone=True, backend='egl')
        print("Successfully created EGL context!")
        return ctx
    except Exception as e:
        raise RuntimeError(f"Failed to create ModernGL context: {e}") from e

# Load OBJ file
def load_mesh(obj_path):
    with load(obj_path) as scene:
        if not scene.meshes:
            raise ValueError(f"No meshes found in {obj_path}")
        mesh = scene.meshes[0]
        vertices = np.array(mesh.vertices, dtype='f4')
        indices = np.array(mesh.faces, dtype='i4').flatten()
        
        # Get normals if they exist, otherwise compute them
        if hasattr(mesh, 'normals') and mesh.normals is not None and len(mesh.normals) > 0:
            normals = np.array(mesh.normals, dtype='f4')
        else:
            # Simple normal calculation - not as accurate as proper calculation but better than nothing
            normals = np.zeros_like(vertices)
            for i in range(0, len(indices), 3):
                if i+2 < len(indices):  # Make sure we have a complete triangle
                    v0 = vertices[indices[i]]
                    v1 = vertices[indices[i+1]]
                    v2 = vertices[indices[i+2]]
                    
                    # Calculate face normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    face_normal = np.cross(edge1, edge2)
                    
                    # Normalize
                    if np.linalg.norm(face_normal) > 0:
                        face_normal = face_normal / np.linalg.norm(face_normal)
                    
                    # Add to all vertices of this face
                    normals[indices[i]] += face_normal
                    normals[indices[i+1]] += face_normal
                    normals[indices[i+2]] += face_normal
            
            # Normalize all vertex normals
            for i in range(len(normals)):
                norm = np.linalg.norm(normals[i])
                if norm > 0:
                    normals[i] = normals[i] / norm
        
        return vertices, indices, normals

# Render a model from different views
def render_views(obj_path, output_dir, ctx):
    os.makedirs(output_dir, exist_ok=True)
    vertices, indices, normals = load_mesh(obj_path)

    # Center and scale the model
    vertices_min = np.min(vertices, axis=0)
    vertices_max = np.max(vertices, axis=0)
    vertices_center = (vertices_min + vertices_max) / 2
    vertices_scale = np.max(vertices_max - vertices_min) / 2
    vertices = (vertices - vertices_center) / vertices_scale

    vbo = ctx.buffer(vertices)
    nbo = ctx.buffer(normals)
    ibo = ctx.buffer(indices)

    program = ctx.program(
        vertex_shader="""
        #version 330
        in vec3 in_vert;
        in vec3 in_norm;
        out vec3 fragNormal;
        out vec3 fragPos;
        uniform mat4 modelview;
        uniform mat4 projection;

        void main() {
            vec4 worldPos = modelview * vec4(in_vert, 1.0);
            fragPos = worldPos.xyz;
            fragNormal = mat3(modelview) * in_norm;
            gl_Position = projection * worldPos;
        }
        """,
        fragment_shader="""
        #version 330
        in vec3 fragNormal;
        in vec3 fragPos;
        out vec4 fragColor;

        void main() {
            // Multiple light directions for better illumination
            vec3 lightDir1 = normalize(vec3(0.5, 1.0, 0.5));
            vec3 lightDir2 = normalize(vec3(-0.5, 0.7, 0.5));
            
            vec3 norm = normalize(fragNormal);
            
            // Calculate diffuse lighting from two sources
            float diff1 = max(dot(norm, lightDir1), 0.0);
            float diff2 = max(dot(norm, lightDir2), 0.0);
            
            // Ambient light component to ensure nothing is completely black
            float ambient = 0.2;
            
            // Combined lighting
            vec3 color = vec3(1.0) * (diff1 + diff2 * 0.5 + ambient);
            
            // Ensure nothing exceeds 1.0
            color = min(color, vec3(1.0));
            
            fragColor = vec4(color, 1.0);
        }
        """
    )

    # Create vertex array with both position and normal attributes
    vao = ctx.vertex_array(
        program, 
        [
            (vbo, '3f', 'in_vert'),
            (nbo, '3f', 'in_norm')
        ],
        ibo
    )
    
    # Set up the projection matrix (perspective)
    aspect_ratio = 1.0  # For our 800x800 viewport
    fov = 45.0  # Field of view in degrees
    near = 0.1
    far = 100.0
    
    # Build the projection matrix
    perspective = np.zeros((4, 4), dtype='f4')
    f = 1.0 / np.tan(np.radians(fov) / 2)
    perspective[0, 0] = f / aspect_ratio
    perspective[1, 1] = f
    perspective[2, 2] = (near + far) / (near - far)
    perspective[2, 3] = 2 * near * far / (near - far)
    perspective[3, 2] = -1.0
    
    # Send the projection matrix to the shader
    program['projection'].write(perspective.tobytes())
    
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((800, 800), 4)])

    # Enable depth testing
    ctx.enable(moderngl.DEPTH_TEST)
    
    for i, angle in enumerate(np.linspace(0, 360, num=13)[:-1]):
        # Create modelview matrix: rotate around Y axis and move back 3 units
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle)), 0],
            [0, 1, 0, 0],
            [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle)), -3],
            [0, 0, 0, 1]
        ], dtype='f4')

        program['modelview'].write(rotation_matrix.tobytes())
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear color and depth buffer
        vao.render()
        data = fbo.read(components=3)
        img = Image.frombytes('RGB', (800, 800), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip to correct orientation
        img.save(os.path.join(output_dir, f'view_{i:02d}.png'))

    # Clean up resources
    vao.release()
    vbo.release()
    nbo.release()
    ibo.release()

# Process all models
def start_render():
    ctx = create_context()  # No window needed
    categories = os.listdir(MODELNET40_OBJ_PATH)
    for category in tqdm(categories, desc="Processing categories"):
        cat_path = os.path.join(MODELNET40_OBJ_PATH, category)
        if not os.path.isdir(cat_path):
            continue

        for split in ["train", "test"]:
            split_path = os.path.join(cat_path, split)
            if not os.path.exists(split_path):
                continue

            obj_files = [file for file in os.listdir(split_path) if file.endswith(".obj")]
            for file in tqdm(obj_files, desc=f"Processing {category}/{split}", leave=False):
                obj_file = os.path.join(split_path, file)
                out_path = os.path.join(OUTPUT_PATH, category, split, file.replace(".obj", ""))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                render_views(obj_file, out_path, ctx)

if __name__ == "__main__":
    start_render()
    print("Rendering complete.")