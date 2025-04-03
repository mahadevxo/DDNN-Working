import os
import numpy as np
import moderngl
import pyassimp
import imageio
from PIL import Image
from pyassimp import load
from tqdm import tqdm
import glfw

# Paths
MODELNET40_OBJ_PATH = "ModelNet40_OBJ"
OUTPUT_PATH = "ModelNet40_12View"
VIEWS = 12  # 12 views per model
AZIMUTH_STEP = 360 / VIEWS  # 30-degree steps

# OpenGL Initialization with EGL
def create_context():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)  # Force EGL

    window = glfw.create_window(800, 800, "Offscreen", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    print("GLFW context API:", glfw.get_current_context())  # Debugging output

    ctx = moderngl.create_standalone_context()
    return ctx, window

# Load OBJ file
def load_mesh(obj_path):
    scene = load(obj_path)
    if not scene.meshes:
        raise ValueError(f"No meshes found in {obj_path}")
    mesh = scene.meshes[0]
    vertices = np.array(mesh.vertices, dtype='f4')
    indices = np.array(mesh.faces, dtype='i4').flatten()
    return vertices, indices

# Render a model from different views
def render_views(obj_path, output_dir, ctx):
    os.makedirs(output_dir, exist_ok=True)
    vertices, indices = load_mesh(obj_path)

    vbo = ctx.buffer(vertices)
    ibo = ctx.buffer(indices)

    program = ctx.program(
        vertex_shader="""
        #version 330
        in vec3 in_vert;
        uniform mat4 modelview;
        void main() {
            gl_Position = modelview * vec4(in_vert, 1.0);
        }
        """,
        fragment_shader="""
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        """
    )

    vao = ctx.simple_vertex_array(program, vbo, 'in_vert')
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((800, 800), 4)])

    for i, angle in enumerate(np.linspace(0, 360, num=13)[:-1]):
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle)), 0],
            [0, 1, 0, 0],
            [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle)), -3],
            [0, 0, 0, 1]
        ], dtype='f4')

        program['modelview'].write(rotation_matrix.tobytes())
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0)
        vao.render()
        data = fbo.read(components=3)
        img = Image.frombytes('RGB', (800, 800), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip to correct orientation
        img.save(os.path.join(output_dir, f'view_{i:02d}.png'))

    vao.release()
    vbo.release()
    ibo.release()

# Process all models
def start_render():
    ctx, window = create_context()
    for category in os.listdir(MODELNET40_OBJ_PATH):
        cat_path = os.path.join(MODELNET40_OBJ_PATH, category)
        if not os.path.isdir(cat_path):
            continue
        
        for split in ["train", "test"]:
            split_path = os.path.join(cat_path, split)
            if not os.path.exists(split_path):
                continue
                
            for file in os.listdir(split_path):
                if file.endswith(".obj"):
                    obj_file = os.path.join(split_path, file)
                    out_path = os.path.join(OUTPUT_PATH, category, split, file.replace(".obj", ""))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    render_views(obj_file, out_path, ctx)
    
    glfw.terminate()

if __name__ == "__main__":
    start_render()
    print("Rendering complete.")