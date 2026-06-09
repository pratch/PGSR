import sys
import os
import types
import torch

# Define quaternion_to_matrix matching PyTorch3D's convention
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = torch.div(2.0, (quaternions * quaternions).sum(-1))
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# Dynamic mocking of pytorch3d to bypass library dependency
pytorch3d = types.ModuleType("pytorch3d")
sys.modules["pytorch3d"] = pytorch3d

pytorch3d_transforms = types.ModuleType("pytorch3d.transforms")
pytorch3d_transforms.quaternion_to_matrix = quaternion_to_matrix
sys.modules["pytorch3d.transforms"] = pytorch3d_transforms

# Dynamic mocking of cv2 to bypass opencv dependency in scene.cameras
cv2_mock = types.ModuleType("cv2")
sys.modules["cv2"] = cv2_mock

# Remaining imports
import math
import argparse
import numpy as np
from plyfile import PlyData
from PIL import Image

# Add the local submodules and project root to python path to resolve modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules", "diff-plane-rasterization"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, getProjectionMatrixCenterShift

class PipelineParamsDummy:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def get_sh_degree_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    N_rest = len(extra_f_names)
    sh_degree = int(math.sqrt((N_rest + 3) / 3)) - 1
    return sh_degree

def parse_camera_txt(txt_path):
    params = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("render_width:"):
            params['width'] = int(line.split(":")[1].strip())
        elif line.startswith("render_height:"):
            params['height'] = int(line.split(":")[1].strip())
        elif line.startswith("intrinsic_3x3:"):
            # read next 3 lines
            intr = []
            for _ in range(3):
                idx += 1
                intr.append([float(x) for x in lines[idx].strip().split()])
            params['K'] = np.array(intr)
        elif line.startswith("extrinsic_world_to_camera_4x4:"):
            # read next 4 lines
            w2c = []
            for _ in range(4):
                idx += 1
                w2c.append([float(x) for x in lines[idx].strip().split()])
            params['W2C'] = np.array(w2c)
        idx += 1
    return params

def jet_colormap(x):
    # Standard Jet colormap implementation
    # Input x shape: (H, W), values in [0, 1]
    # Output: (H, W, 3), values in [0, 255]
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Render depth map from PGSR trained ply and camera view")
    parser.add_argument("--ply", type=str, required=True, help="Path to trained ply file")
    parser.add_argument("--camera", type=str, required=True, help="Path to camera view txt file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the rendered output")
    parser.add_argument("--median_depth", action="store_true", help="Use median depth (where transmittance crosses 0.5) instead of alpha-blended depth")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Parse camera txt file
    print(f"Parsing camera file: {args.camera}")
    cam_params = parse_camera_txt(args.camera)
    width = cam_params['width']
    height = cam_params['height']
    K = cam_params['K']
    w2c = cam_params['W2C']

    # Extrinsics
    R_w2c = w2c[:3, :3]
    T_w2c = w2c[:3, 3]

    # Camera class expects R transposed (cam2world rotation) and T (w2c translation)
    R = R_w2c.T
    T = T_w2c

    # Intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Compute FOVs
    fov_x = focal2fov(fx, width)
    fov_y = focal2fov(fy, height)

    # 2. Instantiate viewpoint camera
    print(f"Creating camera viewpoint: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    view_cam = Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=fov_x,
        FoVy=fov_y,
        image_width=width,
        image_height=height,
        image_path="",
        image_name="render_view",
        uid=0,
        preload_img=False
    )

    # Explicitly set camera parameters and update projection matrix to handle any principal point shift
    view_cam.Cx = cx
    view_cam.Cy = cy
    view_cam.Fx = fx
    view_cam.Fy = fy

    if not np.isclose(cx, width / 2.0) or not np.isclose(cy, height / 2.0):
        print("Handling principal point shift in projection matrix...")
        view_cam.projection_matrix = getProjectionMatrixCenterShift(
            znear=view_cam.znear, zfar=view_cam.zfar,
            cx=cx, cy=cy, fl_x=fx, fl_y=fy,
            w=width, h=height
        ).transpose(0, 1).cuda()
        view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)

    # 3. Load Gaussian model
    print("Determining SH degree of PLY...")
    sh_degree = get_sh_degree_from_ply(args.ply)
    print(f"SH degree: {sh_degree}")

    print("Loading Gaussian Model...")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(args.ply)

    # 4. Render
    print("Rendering...")
    pipeline = PipelineParamsDummy()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    with torch.no_grad():
        out = render(
            viewpoint_camera=view_cam,
            pc=gaussians,
            pipe=pipeline,
            bg_color=background,
            return_plane=True,
            return_depth_normal=True,
            use_median_depth=args.median_depth
        )

    # Extract depth
    plane_depth = out["plane_depth"].squeeze().cpu().numpy()
    
    # Save raw depth map (.npy)
    raw_depth_path = os.path.join(args.output_dir, "depth_raw.npy")
    np.save(raw_depth_path, plane_depth)
    print(f"Saved raw depth to {raw_depth_path}")

    # Colorize and save depth map visualization (.png)
    depth_min, depth_max = plane_depth.min(), plane_depth.max()
    print(f"Depth range: min={depth_min:.4f}, max={depth_max:.4f}")
    depth_normalized = (plane_depth - depth_min) / (depth_max - depth_min + 1e-20)
    depth_color = jet_colormap(depth_normalized)
    
    color_depth_path = os.path.join(args.output_dir, "depth_color.png")
    Image.fromarray(depth_color).save(color_depth_path)
    print(f"Saved colorized depth map to {color_depth_path}")

    # Process and save normal map visualization (.png)
    if "rendered_normal" in out:
        normal = out["rendered_normal"].permute(1, 2, 0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)
        normal = normal.cpu().numpy()
        normal_color = ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        normal_path = os.path.join(args.output_dir, "normal.png")
        Image.fromarray(normal_color).save(normal_path)
        print(f"Saved normal map to {normal_path}")

    # Save rendered RGB image for reference (.png)
    if "render" in out:
        rendering = out["render"].clamp(0.0, 1.0)
        rendering_np = (rendering.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        render_path = os.path.join(args.output_dir, "render_rgb.png")
        Image.fromarray(rendering_np).save(render_path)
        print(f"Saved rendered RGB image to {render_path}")

    print("Success!")

if __name__ == "__main__":
    main()
