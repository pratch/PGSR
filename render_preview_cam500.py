import os
import sys
import math
import torch
import numpy as np
from PIL import Image
from plyfile import PlyData

# Mocking modules for diff-plane-rasterization compatibility
import types
pytorch3d_transforms = types.ModuleType("pytorch3d.transforms")
pytorch3d_transforms.matrix_to_quaternion = lambda x: torch.zeros((x.shape[0], 4), device=x.device)
pytorch3d_transforms.quaternion_to_matrix = lambda x: torch.eye(3, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
sys.modules["pytorch3d.transforms"] = pytorch3d_transforms
cv2_mock = types.ModuleType("cv2")
sys.modules["cv2"] = cv2_mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules", "diff-plane-rasterization"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov

# Helpers
def fibonacci_sphere_dirs(n_points: int) -> list:
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (2.0 * i) / max(1, n_points - 1)
        radius_at_y = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append([x, y, z])
    return points

def look_at(center, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        alternative_up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(alternative_up, z)
        x_norm = np.linalg.norm(x)
    x = x / x_norm
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    c2w = np.eye(4)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = center
    w2c = np.linalg.inv(c2w)
    return w2c, c2w

class PipelineParamsDummy:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def render_preview(ply_path, name, radius):
    print(f"\nRendering preview for {name} with radius {radius}...")
    ply = PlyData.read(ply_path)
    extra_f_names = [p.name for p in ply.elements[0].properties if p.name.startswith("f_rest_")]
    N_rest = len(extra_f_names)
    sh_degree = int(math.sqrt((N_rest + 3) / 3)) - 1
    
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    
    # Initialize knn_f
    gaussians._knn_f = torch.nn.Parameter(
        torch.randn((gaussians.get_xyz.shape[0], 6), dtype=torch.float, device="cuda").requires_grad_(True)
    )
    
    # Object center (median)
    v = ply['vertex'].data
    x_median = np.nanmedian(v['x'])
    y_median = np.nanmedian(v['y'])
    z_median = np.nanmedian(v['z'])
    object_center = np.array([x_median, y_median, z_median], dtype=np.float32)
    print(f"  Center: {object_center}")
    
    # Fov & camera setup
    width, height = 1280, 1280
    fx, fy = 1255.0, 1255.0
    fov_x = focal2fov(fx, width)
    fov_y = focal2fov(fy, height)
    
    directions = fibonacci_sphere_dirs(1000)
    # Get direction for cam_500 (1-based index 500 is index 499)
    direction = directions[499]
    
    camera_center = object_center + radius * np.array(direction)
    w2c, c2w = look_at(camera_center, target=object_center, up=np.array([0.0, 1.0, 0.0]))
    
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    
    view_cam = Camera(
        colmap_id=0,
        R=R, T=T,
        FoVx=fov_x, FoVy=fov_y,
        image_width=width, image_height=height,
        image_path="", image_name="preview",
        uid=0, preload_img=False
    )
    view_cam.Cx = 640.0
    view_cam.Cy = 640.0
    view_cam.Fx = fx
    view_cam.Fy = fy
    
    pipeline = PipelineParamsDummy()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        out = render(
            viewpoint_camera=view_cam,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background,
            return_plane=True,
            return_depth_normal=True
        )
        
    os.makedirs("output/previews", exist_ok=True)
    
    # Save RGB
    if "render" in out:
        rendering = out["render"].clamp(0.0, 1.0)
        rendering_np = (rendering.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        Image.fromarray(rendering_np).save(f"output/previews/{name}_rgb.png")
        print(f"  Saved RGB preview to output/previews/{name}_rgb.png")
        
    # Save visualized depth
    if "plane_depth" in out:
        plane_depth = out["plane_depth"].squeeze().cpu().numpy()
        plane_depth[plane_depth < 0.0] = 0.0
        if "rendered_alpha" in out:
            alpha = out["rendered_alpha"].squeeze().cpu().numpy()
            plane_depth[alpha < 1e-3] = 0.0
            
        non_zero_depth = plane_depth[plane_depth > 0]
        if len(non_zero_depth) > 0:
            d_min = non_zero_depth.min()
            d_max = non_zero_depth.max()
            depth_vis = np.zeros_like(plane_depth, dtype=np.uint8)
            mask = plane_depth > 0
            depth_vis[mask] = (255.0 * (1.0 - (plane_depth[mask] - d_min) / max(1e-5, d_max - d_min))).clip(0, 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(plane_depth, dtype=np.uint8)
            
        Image.fromarray(depth_vis).save(f"output/previews/{name}_depth.png")
        print(f"  Saved Depth preview to output/previews/{name}_depth.png")

if __name__ == "__main__":
    configs = [
        ("splats/butterfly.ply", "butterfly", 2.28),
        ("splats/cake.ply", "cake", 31.24),
        ("splats/caketall.ply", "caketall", 3.07),
        ("splats/fox.ply", "fox", 13.87),
        ("splats/tablebench.ply", "tablebench", 3.23),
    ]
    for ply_path, name, radius in configs:
        render_preview(ply_path, name, radius)
