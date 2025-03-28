import os
import numpy as np
import trimesh
import imageio
import pygfx as gfx
import random
import platform
import wgpu
from wgpu.gui.offscreen import WgpuCanvas
from tqdm import tqdm
from OpenGL import GL  # OpenGL for NVIDIA GPUs (CUDA)
from OpenGL.GL import shaders
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA
from OpenGL.GL import glFramebufferRenderbuffer
from OpenGL.EGL import eglDestroyContext

# Ensure Metal GPU Backend for Mac or CUDA for NVIDIA GPUs
if platform.system() == "Darwin":
    device = wgpu.utils.get_default_device()  # Metal for macOS
else:
    cuda.init()
    device = None  # Use CUDA directly for NVIDIA GPU

# Paths
INPUT_DIR = "./ModelNet40_OBJ"
OUTPUT_DIR = "./ModelNet40_12View"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Camera settings
INCLINE_ANGLE = np.radians(30)
NUM_VIEWS = 12
ANGLE_STEP = 360 / NUM_VIEWS
PROGRESS_BAR_LENGTH = 150  # Define progress bar length

def random_color():
    """Generates a random RGB color."""
    return (random.random(), random.random(), random.random())

def setup_scene(mesh):
    """Prepares the 3D scene with the object and realistic shading."""
    scene = gfx.Scene()
    
    # Convert to float32 for GPU compatibility
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # Generate a random color for the object
    color = random_color()

    # Create GPU geometry with shading
    geometry = gfx.Geometry(positions=vertices, indices=faces)
    material = gfx.MeshStandardMaterial(
        color=color,  # Set the object color to the random color
        metalness=0.3,          # Moderate metallic reflection
        roughness=0.7           # Adds shading detail
    )
    obj = gfx.Mesh(geometry, material)
    scene.add(obj)

    # Improved Lighting
    scene.add(gfx.AmbientLight((1, 1, 1), 0.3))  # Dim ambient light for contrast
    light = gfx.DirectionalLight((1, 1, 1), 2.0)  # Stronger directional light
    light.world.position = (2, 2, 5)  # Position for good shading effects
    scene.add(light)

    return scene, obj


def render_views(obj_path, save_dir):
    """Efficiently renders 12 views of an OBJ file using the GPU."""
    # Load and process mesh
    mesh = trimesh.load(obj_path, file_type="obj")

    # Compute optimal camera distance based on object bounding box
    bbox = mesh.bounds
    max_dim = np.max(bbox[1] - bbox[0])  # Get the longest axis
    camera_distance = max_dim * 2.5  # Adjust based on model size

    # Create rendering canvas and renderer
    canvas = WgpuCanvas() if platform.system() == "Darwin" else None  # Metal (macOS) or OpenGL for NVIDIA
    renderer = gfx.renderers.WgpuRenderer(canvas) if platform.system() == "Darwin" else None

    if platform.system() != "Darwin":
        # Use EGL for off-screen rendering on NVIDIA GPUs
        from OpenGL.EGL import eglGetDisplay, eglInitialize, eglChooseConfig, eglCreateContext, eglMakeCurrent, EGL_DEFAULT_DISPLAY
        from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
        import ctypes

        # Initialize EGL display
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        eglInitialize(display, None, None)

        # Choose an EGL config
        config_attribs = [
            0x3024, 8,  # Red size
            0x3023, 8,  # Green size
            0x3022, 8,  # Blue size
            0x3021, 8,  # Alpha size
            0x3038, 1,  # Renderable type (OpenGL ES 2.0)
            0x3031, 0   # None
        ]
        num_configs = ctypes.c_int()
        config = ctypes.POINTER(ctypes.c_void_p)()
        eglChooseConfig(display, config_attribs, ctypes.byref(config), 1, ctypes.byref(num_configs))

        # Create an EGL context with proper attributes
        context_attribs = [
            0x3098, 2,  # EGL_CONTEXT_CLIENT_VERSION = 2 (OpenGL ES 2.0)
            0x3038, 0   # EGL_NONE to terminate the list
        ]
        context = eglCreateContext(display, config, None, context_attribs)
        eglMakeCurrent(display, None, None, context)

        # Create and bind a framebuffer for off-screen rendering
        from OpenGL.GL import glGenFramebuffers, glBindFramebuffer, GL_FRAMEBUFFER
        framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

        # Create and bind a renderbuffer for the framebuffer
        from OpenGL.GL import glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage, GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, GL_RGBA8, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT

        # Generate and attach a color renderbuffer
        color_renderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, color_renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 512, 512)  # Use GL_RGBA8 for the color renderbuffer
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_renderbuffer)

        # Generate and attach a depth renderbuffer
        depth_renderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer)

        # Ensure the framebuffer is complete
        from OpenGL.GL import glCheckFramebufferStatus, GL_FRAMEBUFFER_COMPLETE
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

        # Clear the screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render a placeholder image (black image for now)
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        # Clean up EGL context after rendering
        eglMakeCurrent(display, None, None, None)
        eglDestroyContext(display, context)
        eglTerminate(display)

    # Prepare scene
    scene, obj = setup_scene(mesh)

    # Setup Camera
    camera = gfx.PerspectiveCamera(50)
    camera.local.position = (0, -camera_distance, camera_distance)
    camera.look_at((0, 0, 0))
    scene.add(camera)

    os.makedirs(save_dir, exist_ok=True)

    # Render all views
    for i in range(NUM_VIEWS):
        angle = np.radians(i * ANGLE_STEP)

        # Compute new camera position while keeping the object centered
        x = np.cos(angle) * camera_distance
        y = np.sin(angle) * camera_distance
        camera.local.position = (x, y, camera_distance)
        camera.look_at((0, 0, 0))  # Always look at the object's center

        if platform.system() == "Darwin":
            renderer.render(scene, camera)
            img = np.asarray(renderer.target.draw())
        else:
            # Use EGL for off-screen rendering on NVIDIA GPUs
            from OpenGL.EGL import eglGetDisplay, eglInitialize, eglChooseConfig, eglCreateContext, eglMakeCurrent, EGL_DEFAULT_DISPLAY
            from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
            import ctypes

            # Initialize EGL display
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
            eglInitialize(display, None, None)

            # Choose an EGL config
            config_attribs = [
                0x3024, 8,  # Red size
                0x3023, 8,  # Green size
                0x3022, 8,  # Blue size
                0x3021, 8,  # Alpha size
                0x3038, 1,  # Renderable type (OpenGL ES 2.0)
                0x3031, 0   # None
            ]
            num_configs = ctypes.c_int()
            config = ctypes.POINTER(ctypes.c_void_p)()
            eglChooseConfig(display, config_attribs, ctypes.byref(config), 1, ctypes.byref(num_configs))

            # Create an EGL context with proper attributes
            context_attribs = [
                0x3098, 2,  # EGL_CONTEXT_CLIENT_VERSION = 2 (OpenGL ES 2.0)
                0x3038, 0   # EGL_NONE to terminate the list
            ]
            context = eglCreateContext(display, config, None, context_attribs)
            eglMakeCurrent(display, None, None, context)

            # Create and bind a framebuffer for off-screen rendering
            from OpenGL.GL import glGenFramebuffers, glBindFramebuffer, GL_FRAMEBUFFER
            framebuffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

            # Create and bind a renderbuffer for the framebuffer
            from OpenGL.GL import glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage, GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, GL_RGBA8, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT

            # Generate and attach a color renderbuffer
            color_renderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, color_renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 512, 512)  # Use GL_RGBA8 for the color renderbuffer
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_renderbuffer)

            # Generate and attach a depth renderbuffer
            depth_renderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer)

            # Ensure the framebuffer is complete
            from OpenGL.GL import glCheckFramebufferStatus, GL_FRAMEBUFFER_COMPLETE
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Framebuffer is not complete")

            # Clear the screen
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render a placeholder image (black image for now)
            img = np.zeros((512, 512, 3), dtype=np.uint8)

            # Clean up EGL context after rendering
            eglMakeCurrent(display, None, None, None)
            eglDestroyContext(display, context)
            eglTerminate(display)

        img_path = os.path.join(save_dir, f"view_{i:02d}.png")
        imageio.imwrite(img_path, img)

def get_colored_bar(progress, total, length=50):
    """Generates a gradient progress bar from red to green."""
    bar = ""
    for i in range(length):
        ratio = i / (length - 1)
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        b = 0
        color_code = f"\033[38;2;{r};{g};{b}m█\033[0m"
        bar += color_code if i < (progress / total) * length else " "
    return bar


# Gather all OBJ files
obj_files = []
for category in os.listdir(INPUT_DIR):
    cat_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(cat_path):
        continue

    for split in ["train", "test"]:
        split_path = os.path.join(cat_path, split)
        if not os.path.isdir(split_path):
            continue

        for file in os.listdir(split_path):
            if file.endswith(".obj"):
                obj_files.append((category, split, file))

# Process files with progress bar
with tqdm(total=len(obj_files), ncols=PROGRESS_BAR_LENGTH, desc="Rendering Objects") as pbar:
    for category, split, file in obj_files:
        obj_file = os.path.join(INPUT_DIR, category, split, file)
        save_dir = os.path.join(OUTPUT_DIR, category, split, file.replace(".obj", ""))
        render_views(obj_file, save_dir)
        pbar.update(1)
print("Rendering completed.")