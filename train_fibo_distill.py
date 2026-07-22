import os
import sys
import uuid
import math
import json
import random
import shutil
import types
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, desc=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.length = len(iterable) if hasattr(iterable, "__len__") else None
            self.iterator = iter(iterable)
            self.current = 0
            print(f"[{self.desc}] Started...")

        def __iter__(self):
            return self

        def __next__(self):
            try:
                item = next(self.iterator)
                self.current += 1
                if self.length and self.current % max(1, self.length // 20) == 0:
                    print(f"[{self.desc}] {self.current}/{self.length} ({(self.current/self.length)*100:.1f}%)")
                return item
            except StopIteration:
                print(f"[{self.desc}] Completed.")
                raise StopIteration

        def set_postfix(self, postfix_dict):
            if self.current % 100 == 0:
                postfix_str = ", ".join([f"{k}: {v}" for k, v in postfix_dict.items()])
                print(f"[{self.desc}] step {self.current}/{self.length or '?'} - {postfix_str}")

        def update(self, n=1):
            self.current += n

        def close(self):
            print(f"[{self.desc}] Closed.")

from PIL import Image
from plyfile import PlyData
from argparse import ArgumentParser, Namespace

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

# Dynamic mocking of pytorch3d and cv2 to bypass library dependencies
pytorch3d = types.ModuleType("pytorch3d")
sys.modules["pytorch3d"] = pytorch3d
pytorch3d_transforms = types.ModuleType("pytorch3d.transforms")
pytorch3d_transforms.quaternion_to_matrix = quaternion_to_matrix
sys.modules["pytorch3d.transforms"] = pytorch3d_transforms
cv2_mock = types.ModuleType("cv2")
sys.modules["cv2"] = cv2_mock

# Add the local submodules and project root to python path to resolve modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules", "diff-plane-rasterization"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import original PGSR modules
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from scene.app_model import AppModel
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrixCenterShift
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from utils.general_utils import safe_state
from utils.image_utils import psnr, erode
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(22)

# Camera helpers
def fibonacci_sphere_dirs(n_points: int) -> list:
    if n_points <= 0:
        return []
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

def azim_elev_from_dir(direction):
    x, y, z = direction
    y = max(-1.0, min(1.0, y))
    azim = math.degrees(math.atan2(x, z))
    elev = math.degrees(math.asin(y))
    return azim, elev

class PipelineParamsDummy:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

# Depth map generation helpers
def save_camera_txt(txt_path, K, w2c, c2w, width=1280, height=1280):
    lines = [
        "camera_name: main_view",
        "convention: world_to_camera",
        "matrix_layout: row_major_printed",
        "handedness: right_handed",
        "units: scene_units",
        f"source_width: {width}",
        f"source_height: {height}",
        f"render_width: {width}",
        f"render_height: {height}",
        "scale_factor: 1.000000000",
        "intrinsic_3x3:",
        f"{K[0,0]:.9f} {K[0,1]:.9f} {K[0,2]:.9f}",
        f"{K[1,0]:.9f} {K[1,1]:.9f} {K[1,2]:.9f}",
        f"{K[2,0]:.9f} {K[2,1]:.9f} {K[2,2]:.9f}",
        "intrinsic_scaled_3x3:",
        f"{K[0,0]:.9f} {K[0,1]:.9f} {K[0,2]:.9f}",
        f"{K[1,0]:.9f} {K[1,1]:.9f} {K[1,2]:.9f}",
        f"{K[2,0]:.9f} {K[2,1]:.9f} {K[2,2]:.9f}",
        "extrinsic_world_to_camera_4x4:",
        f"{w2c[0,0]:.9f} {w2c[0,1]:.9f} {w2c[0,2]:.9f} {w2c[0,3]:.9f}",
        f"{w2c[1,0]:.9f} {w2c[1,1]:.9f} {w2c[1,2]:.9f} {w2c[1,3]:.9f}",
        f"{w2c[2,0]:.9f} {w2c[2,1]:.9f} {w2c[2,2]:.9f} {w2c[2,3]:.9f}",
        f"{w2c[3,0]:.9f} {w2c[3,1]:.9f} {w2c[3,2]:.9f} {w2c[3,3]:.9f}",
        "extrinsic_camera_to_world_4x4:",
        f"{c2w[0,0]:.9f} {c2w[0,1]:.9f} {c2w[0,2]:.9f} {c2w[0,3]:.9f}",
        f"{c2w[1,0]:.9f} {c2w[1,1]:.9f} {c2w[1,2]:.9f} {c2w[1,3]:.9f}",
        f"{c2w[2,0]:.9f} {c2w[2,1]:.9f} {c2w[2,2]:.9f} {c2w[2,3]:.9f}",
        f"{c2w[3,0]:.9f} {c2w[3,1]:.9f} {c2w[3,2]:.9f} {c2w[3,3]:.9f}",
    ]
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def write_pfm(path, image, scale=1):
    image = np.flipud(image)  # PFM expects bottom-to-top layout
    with open(path, 'wb') as file:
        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            color = True
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1, or H x W dimensions.')

        file.write(b'PF\n' if color else b'Pf\n')
        file.write(f'{image.shape[1]} {image.shape[0]}\n'.encode())
        
        endian = image.dtype.byteorder
        if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
            scale = -abs(scale)
        else:
            scale = abs(scale)
        file.write(f'{scale}\n'.encode())
        
        image.tofile(file)

def jet_colormap(x):
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

def find_edge(depth_map, threshold, exponent=-1.0):
    is_foreground = depth_map > 0
    h, w = depth_map.shape
    
    def get_shifted(arr, dy, dx):
        res = np.zeros_like(arr)
        if dy > 0:
            res[dy:, :] = arr[:-dy, :]
        elif dy < 0:
            res[:dy, :] = arr[-dy:, :]
        if dx > 0:
            res[:, dx:] = arr[:, :-dx]
        elif dx < 0:
            res[:, :dx] = arr[:, -dx:]
        return res

    d_up = get_shifted(depth_map, 1, 0)
    d_down = get_shifted(depth_map, -1, 0)
    d_left = get_shifted(depth_map, 0, 1)
    d_right = get_shifted(depth_map, 0, -1)
    
    in_up = np.zeros_like(depth_map, dtype=bool)
    in_up[1:, :] = True
    in_down = np.zeros_like(depth_map, dtype=bool)
    in_down[:-1, :] = True
    in_left = np.zeros_like(depth_map, dtype=bool)
    in_left[:, 1:] = True
    in_right = np.zeros_like(depth_map, dtype=bool)
    in_right[:, :-1] = True

    def check_pass(d_neighbor, in_bounds):
        if exponent == 1.0:
            pass_cond = np.abs(depth_map - d_neighbor) < threshold
        elif exponent == -1.0:
            pass_cond = np.abs(depth_map - d_neighbor) < threshold * depth_map * depth_map
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                pass_cond = np.abs(np.power(depth_map, exponent) - np.power(d_neighbor, exponent)) < threshold
        return ~in_bounds | ((d_neighbor > 0) & pass_cond)

    pass_up = check_pass(d_up, in_up)
    pass_down = check_pass(d_down, in_down)
    pass_left = check_pass(d_left, in_left)
    pass_right = check_pass(d_right, in_right)

    edge_mask = is_foreground & (~pass_up | ~pass_down | ~pass_left | ~pass_right)
    return edge_mask

def dilate_mask(mask, iterations):
    if iterations <= 0:
        return mask
    res = mask.copy()
    for _ in range(iterations):
        u = np.zeros_like(res)
        d = np.zeros_like(res)
        l = np.zeros_like(res)
        r = np.zeros_like(res)
        
        u[1:, :] = res[:-1, :]
        d[:-1, :] = res[1:, :]
        l[:, 1:] = res[:, :-1]
        r[:, :-1] = res[:, 1:]
        
        ul = np.zeros_like(res)
        ur = np.zeros_like(res)
        dl = np.zeros_like(res)
        dr = np.zeros_like(res)
        
        ul[1:, 1:] = res[:-1, :-1]
        ur[1:, :-1] = res[:-1, 1:]
        dl[:-1, 1:] = res[1:, :-1]
        dr[:-1, :-1] = res[1:, 1:]
        
        res = res | u | d | l | r | ul | ur | dl | dr
    return res

def generate_distill_dataset(args, init_ply_path, dataset_path):
    print("------------------------------------------------------------------")
    print(f"Generating pseudo-GT distillation dataset at: {dataset_path}")
    print("------------------------------------------------------------------")
    os.makedirs(os.path.join(dataset_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test"), exist_ok=True)

    # Write a dummy points3d.ply to avoid loading errors in Blender reader
    dummy_xyz = np.zeros((1, 3))
    dummy_rgb = np.zeros((1, 3))
    from scene.dataset_readers import storePly
    storePly(os.path.join(dataset_path, "points3d.ply"), dummy_xyz, dummy_rgb)

    # Load initial Gaussian Model to render views
    print("Loading initial Gaussian model to render dataset...")
    ply = PlyData.read(init_ply_path)
    extra_f_names = [p.name for p in ply.elements[0].properties if p.name.startswith("f_rest_")]
    N_rest = len(extra_f_names)
    sh_degree = int(math.sqrt((N_rest + 3) / 3)) - 1
    print(f"SH degree of ply: {sh_degree}")
    
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(init_ply_path)

    # Compute object center
    v = ply['vertex'].data
    x_median = np.nanmedian(v['x'])
    y_median = np.nanmedian(v['y'])
    z_median = np.nanmedian(v['z'])
    object_center = np.array([x_median, y_median, z_median], dtype=np.float32)
    print(f"Object center (median xyz): {object_center}")

    # Generate Fibonacci sphere directions
    directions = fibonacci_sphere_dirs(args.n_cams)
    
    # Calculate FOV
    fov_x = focal2fov(args.fx, args.width)
    fov_y = focal2fov(args.fy, args.height)

    train_frames = []
    test_frames = []

    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsDummy()

    for idx, direction in enumerate(tqdm(directions, desc="Rendering views")):
        camera_center = object_center + args.radius * np.array(direction)
        w2c, c2w = look_at(camera_center, target=object_center, up=np.array([0.0, 1.0, 0.0]))
        
        # Build Camera object
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        view_cam = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fov_x,
            FoVy=fov_y,
            image_width=args.width,
            image_height=args.height,
            image_path="",
            image_name=f"r_{idx}",
            uid=idx,
            preload_img=False
        )
        view_cam.Cx = args.cx
        view_cam.Cy = args.cy
        view_cam.Fx = args.fx
        view_cam.Fy = args.fy

        if not np.isclose(args.cx, args.width / 2.0) or not np.isclose(args.cy, args.height / 2.0):
            view_cam.projection_matrix = getProjectionMatrixCenterShift(
                znear=view_cam.znear, zfar=view_cam.zfar,
                cx=args.cx, cy=args.cy, fl_x=args.fx, fl_y=args.fy,
                w=args.width, h=args.height
            ).transpose(0, 1).cuda()
            view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)

        # Render RGB & Alpha
        with torch.no_grad():
            out = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                return_plane=True,
                return_depth_normal=False
            )

        rendering = out["render"].clamp(0.0, 1.0)
        alpha = out["rendered_alpha"].clamp(0.0, 1.0)
        rgba = torch.cat([rendering, alpha], dim=0) # (4, H, W)
        rgba_np = (rgba.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)

        # Decide train vs test
        is_test = (idx % args.test_interval == 0)
        split = "test" if is_test else "train"
        
        # Save image
        img_name = f"r_{idx}"
        img_path = os.path.join(dataset_path, split, f"{img_name}.png")
        Image.fromarray(rgba_np).save(img_path)

        # c2w in OpenGL convention for Blender loader
        c2w_opengl = c2w.copy()
        c2w_opengl[:3, 1:3] *= -1

        frame_data = {
            "file_path": f"./{split}/{img_name}",
            "transform_matrix": c2w_opengl.tolist()
        }

        if is_test:
            test_frames.append(frame_data)
        else:
            train_frames.append(frame_data)

    # Save transforms.json
    with open(os.path.join(dataset_path, "transforms_train.json"), "w") as f:
        json.dump({
            "camera_angle_x": fov_x,
            "frames": train_frames
        }, f, indent=2)

    with open(os.path.join(dataset_path, "transforms_test.json"), "w") as f:
        json.dump({
            "camera_angle_x": fov_x,
            "frames": test_frames
        }, f, indent=2)

    print(f"Successfully generated {len(train_frames)} train frames and {len(test_frames)} test frames.")
    print("------------------------------------------------------------------")
    return object_center

def generate_depth_maps_for_checkpoint(args, gaussians, object_center, output_root, scene_name):
    print("------------------------------------------------------------------")
    print(f"Generating depth maps for {scene_name} at: {output_root}")
    print("------------------------------------------------------------------")
    
    # Generate Fibonacci camera directions
    directions = fibonacci_sphere_dirs(args.n_cams)
    
    fov_x = focal2fov(args.fx, args.width)
    fov_y = focal2fov(args.fy, args.height)
    
    pipeline = PipelineParamsDummy()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    manifest = {
        "scene": scene_name,
        "n_cams": args.n_cams,
        "camera_radius": args.radius,
        "runs": []
    }
    
    for cam_idx, direction in enumerate(tqdm(directions, desc="Generating depth maps"), start=1):
        cam_folder_name = f"cam_{cam_idx:02d}"
        out_dir = os.path.join(output_root, scene_name, cam_folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        camera_center = object_center + args.radius * np.array(direction)
        w2c, c2w = look_at(camera_center, target=object_center, up=np.array([0.0, 1.0, 0.0]))
        
        # Build Intrinsic matrix
        K = np.eye(3)
        K[0, 0] = args.fx
        K[1, 1] = args.fy
        K[0, 2] = args.cx
        K[1, 2] = args.cy
        
        # Save camera_main_view.txt
        txt_path = os.path.join(out_dir, "camera_main_view.txt")
        save_camera_txt(txt_path, K, w2c, c2w, width=args.width, height=args.height)
        
        # Instantiate camera viewpoint for rendering
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        view_cam = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fov_x,
            FoVy=fov_y,
            image_width=args.width,
            image_height=args.height,
            image_path="",
            image_name="render_view",
            uid=0,
            preload_img=False
        )

        view_cam.Cx = args.cx
        view_cam.Cy = args.cy
        view_cam.Fx = args.fx
        view_cam.Fy = args.fy

        if not np.isclose(args.cx, args.width / 2.0) or not np.isclose(args.cy, args.height / 2.0):
            view_cam.projection_matrix = getProjectionMatrixCenterShift(
                znear=view_cam.znear, zfar=view_cam.zfar,
                cx=args.cx, cy=args.cy, fl_x=args.fx, fl_y=args.fy,
                w=args.width, h=args.height
            ).transpose(0, 1).cuda()
            view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)

        # Render depth using ray-plane intersection the PGSR way
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

        # 1. Save standard RGB image
        if "render" in out:
            rendering = out["render"].clamp(0.0, 1.0)
            rendering_np = (rendering.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
            rgb_path = os.path.join(out_dir, "main_view_rgb.png")
            Image.fromarray(rendering_np).save(rgb_path)

        # 2. Save Alpha map
        if "rendered_alpha" in out:
            alpha = out["rendered_alpha"].squeeze().cpu().numpy()
            alpha_np = (alpha * 255.0).clip(0, 255).astype(np.uint8)
            alpha_path = os.path.join(out_dir, "main_view_alpha.png")
            Image.fromarray(alpha_np).save(alpha_path)
            
            # Save Alpha as PFM
            alpha_pfm_path = os.path.join(out_dir, "main_view_alpha.pfm")
            write_pfm(alpha_pfm_path, alpha.astype(np.float32))

        # Extract depth
        plane_depth = out["plane_depth"].squeeze().cpu().numpy()
        plane_depth[plane_depth < 0.0] = 0.0
        if "rendered_alpha" in out:
            alpha = out["rendered_alpha"].squeeze().cpu().numpy()
            plane_depth[alpha < 1e-3] = 0.0

        # Compute edge mask and dilate it
        edge_mask = find_edge(plane_depth, args.threshold, exponent=-1.0)
        dilated_edge = dilate_mask(edge_mask, args.dilate)
        
        # Apply mask
        depth_masked = plane_depth.copy()
        depth_masked[dilated_edge] = 0.0

        # Save dilated edge mask as PNG
        edge_mask_png = (dilated_edge * 255).astype(np.uint8)
        edge_mask_path = os.path.join(out_dir, "main_view_edge_mask.png")
        Image.fromarray(edge_mask_png).save(edge_mask_path)

        # 3. Save raw depth map (.pfm)
        depth_pfm_path = os.path.join(out_dir, "main_view_depth.pfm")
        write_pfm(depth_pfm_path, plane_depth.astype(np.float32))

        # 4. Save raw depth map masked (.pfm)
        depth_masked_pfm_path = os.path.join(out_dir, "main_view_depth_masked.pfm")
        write_pfm(depth_masked_pfm_path, depth_masked.astype(np.float32))

        # 5. Save visualized depth map (.png)
        depth_min, depth_max = plane_depth.min(), plane_depth.max()
        depth_normalized = (plane_depth - depth_min) / (depth_max - depth_min + 1e-20)
        depth_color = jet_colormap(depth_normalized)
        color_depth_path = os.path.join(out_dir, "main_view_depth.png")
        Image.fromarray(depth_color).save(color_depth_path)

        manifest["runs"].append({
            "cam_index": cam_idx,
            "direction": [float(d) for d in direction],
            "azim_deg": float(azim_elev_from_dir(direction)[0]),
            "elev_deg": float(azim_elev_from_dir(direction)[1]),
            "status": "ok",
            "returncode": 0
        })

    # Save manifest file
    manifest_path = os.path.join(output_root, scene_name, "fibo_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("------------------------------------------------------------------")

def prepare_output_and_logger(args):    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(min(5, len(scene.getTrainCameras())), min(30, len(scene.getTrainCameras())), 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def main():
    parser = ArgumentParser(description="PGSR Distillation / Refinement from Fibonacci Sphere")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Custom/overriding args
    parser.add_argument('--ply', type=str, required=True, help="Path to initial/source ply file")
    parser.add_argument('--n-cams', type=int, default=1000, help="Number of views to render")
    parser.add_argument('--radius', type=float, default=3.0, help="Camera distance from target")
    parser.add_argument('--fx', type=float, default=1255.0, help="Focal length fx")
    parser.add_argument('--fy', type=float, default=1255.0, help="Focal length fy")
    parser.add_argument('--cx', type=float, default=640.0, help="Principal point cx")
    parser.add_argument('--cy', type=float, default=640.0, help="Principal point cy")
    parser.add_argument('--width', type=int, default=1280, help="Render width")
    parser.add_argument('--height', type=int, default=1280, help="Render height")
    
    parser.add_argument('--test_interval', type=int, default=20, help="Interval for reserving test views")
    parser.add_argument('--reg_from_iter', type=int, default=0, help="Start regularization from this iteration")
    
    parser.add_argument('--threshold', type=float, default=0.01, help="Depth edge mask threshold")
    parser.add_argument('--dilate', type=int, default=5, help="Depth edge mask dilation iterations")
    parser.add_argument('--median_depth', action='store_true', help="Use median depth instead of expected depth")
    
    parser.add_argument('--quick_trial', action='store_true', help="Run a quick trial with 50 cams and 200 iterations")
    parser.add_argument('--only_30k', action='store_true', help="Only save and evaluate at iteration 30,000")

    # Parse and override defaults
    args = parser.parse_args()
    
    # User's specifications: start regularization from iter 0
    args.single_view_weight_from_iter = args.reg_from_iter
    args.multi_view_weight_from_iter = args.reg_from_iter

    if args.quick_trial:
        args.n_cams = 50
        args.iterations = 200
        args.densify_until_iter = 150
        args.densify_from_iter = 50
        save_iters = [100, 200]
        test_iters = [100, 200]
        print("Running quick trial: n_cams=50, iterations=200")
    else:
        args.iterations = 30000
        if args.only_30k:
            save_iters = [30000]
            test_iters = [30000]
        else:
            save_iters = [10000, 20000, 30000]
            test_iters = [10000, 20000, 30000]

    if not args.model_path:
        args.model_path = os.path.join("./output", f"beermug_distill_{datetime.now().strftime('%b%d_%H%M%S')}")
        
    print("Preparing model output directory...")
    tb_writer = prepare_output_and_logger(args)

    # Save reproduction shell script to model path
    os.makedirs(args.model_path, exist_ok=True)
    reproduce_script_path = os.path.join(args.model_path, "reproduce.sh")
    with open(reproduce_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        cmd_args = " ".join([f'"{a}"' if " " in a or "*" in a or "$" in a else a for a in sys.argv])
        f.write(f"python {cmd_args}\n")
    os.chmod(reproduce_script_path, 0o755)

    # 1. Render and save pseudo-GT dataset
    dataset_path = os.path.join(args.model_path, "distill_dataset")
    object_center = generate_distill_dataset(args, args.ply, dataset_path)
    args.source_path = dataset_path

    # Extract configs for Scene/Gaussians/Optimization
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    
    # Disable image preloading to prevent CUDA Out Of Memory with 1000 views
    dataset.preload_img = False
    
    # Overwrite iterations list
    opt.iterations = args.iterations
    opt.single_view_weight_from_iter = args.reg_from_iter
    opt.multi_view_weight_from_iter = args.reg_from_iter

    # 2. Load model & scene
    print("Initializing Scene...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    print(f"Loading initial model weights from: {args.ply}")
    gaussians.load_ply(args.ply)

    # Check and initialize knn_f (since standard 3DGS PLY doesn't have it)
    if gaussians._knn_f.numel() == 0 or gaussians._knn_f.shape[0] != gaussians.get_xyz.shape[0]:
        print("Initializing knn_f features...")
        gaussians._knn_f = torch.nn.Parameter(
            torch.randn((gaussians.get_xyz.shape[0], 6), dtype=torch.float, device="cuda").requires_grad_(True)
        )

    # Initialize max_radii2D and max_weight to match the loaded model size
    print("Initializing max_radii2D and max_weight...")
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    gaussians.max_weight = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    print("Setting up optimizers and learning rate schedulers...")
    gaussians.training_setup(opt)

    app_model = AppModel(num_images=max(1600, args.n_cams))
    app_model.train()
    app_model.cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training progress")
    
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    print("------------------------------------------------------------------")
    print("Starting PGSR training and regularization...")
    print("------------------------------------------------------------------")

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        
        # Update learning rate
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # Render
        # Always return plane & depth normal since we start regularization from iter 0
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
            return_plane=True, return_depth_normal=True
        )
        
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # L1 and SSIM photo losses
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # Scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()

        # Single-view normal loss (regularization)
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0, 1).detach() ** 2
            if not opt.wo_image_weight:
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += normal_loss

        # Multi-view geometric & photo consistency losses (regularization)
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                # Import generator function
                from train import gen_virtul_cam
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=min(1.5, dataset.multi_view_max_dis), deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(
                    nearest_cam, gaussians, pipe, bg, app_model=app_model,
                    return_plane=True, return_depth_normal=False
                )

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[3, :3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                
                R_mat = torch.tensor(nearest_cam.R).float().cuda()
                T_vec = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam - T_vec) @ R_mat.transpose(-1, -2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3, :3] + viewpoint_cam.world_view_transform[3, :3]
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                     pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()
                
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    
                    if not use_virtul_cam:
                        with torch.no_grad():
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            pixels = pixels.reshape(-1, 2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H_g, W_g = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_g - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_g - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

                        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)
                        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]
                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1), 
                                         ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / ref_local_d[..., None, None]
                        
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W_g - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H_g - 1) - 1.0
                        
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        # Backpropagation and optimization step
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "Single": f"{ema_single_view_for_log:.5f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.5f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            # Log reports
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), test_iters, scene, render, (pipe, background), app_model)
            
            # Save Checkpoint & Generate Depth Maps at iterations
            if iteration in save_iters:
                print(f"\n[ITER {iteration}] Saving Checkpoint and model PLY")
                scene.save(iteration)
                app_model.save_weights(scene.model_path, iteration)
                
                # Generate depth maps for all 1000 views at this checkpoint
                ckpt_output_dir = os.path.join(args.model_path, f"depths_iter_{iteration}")
                generate_depth_maps_for_checkpoint(
                    args=args,
                    gaussians=gaussians,
                    object_center=object_center,
                    output_root=ckpt_output_dir,
                    scene_name=os.path.splitext(os.path.basename(args.ply))[0]
                )

            # Densification & Pruning
            if iteration < opt.densify_until_iter:
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # Multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # Reset Opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer Step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()
    print("\nTraining complete. All depth maps generated successfully.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    safe_state(False)
    main()
