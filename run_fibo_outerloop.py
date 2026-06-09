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

# Add the project root to python path to resolve modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, getProjectionMatrixCenterShift, normal_from_depth_image

def load_simple_yaml(file_path):
    defaults = {}
    scenes = []
    current_section = None
    current_scene = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.split("#")[0].strip()
            if not line_clean:
                continue
            if line_clean.startswith("scenes:"):
                current_section = "scenes"
                continue
            elif line_clean.startswith("defaults:"):
                current_section = "defaults"
                continue
            if line_clean.startswith("- "):
                line_clean = line_clean[2:].strip()
                if current_section == "scenes":
                    current_scene = {}
                    scenes.append(current_scene)
            if ":" in line_clean:
                parts = line_clean.split(":", 1)
                k = parts[0].strip()
                v_str = parts[1].strip()
                if v_str.lower() == "true":
                    v = True
                elif v_str.lower() == "false":
                    v = False
                else:
                    try:
                        if "." in v_str:
                            v = float(v_str)
                        else:
                            v = int(v_str)
                    except ValueError:
                        v = v_str
                if current_section == "defaults":
                    defaults[k] = v
                elif current_section == "scenes" and current_scene is not None:
                    current_scene[k] = v
    return defaults, scenes

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

def fibonacci_hemisphere_dirs(n_points, upper=True, y_down=False):
    if n_points <= 0:
        return []

    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    i = 0
    max_iter = 20 * n_points
    while len(points) < n_points and i < max_iter:
        y = 1.0 - (2.0 * i) / max(1, 2 * n_points - 1)

        if y_down:
            keep = (y <= 0.0) if upper else (y >= 0.0)
        else:
            keep = (y >= 0.0) if upper else (y <= 0.0)

        if keep:
            radius_at_y = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * i
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            points.append([x, y, z])

        i += 1

    return points[:n_points]

def azim_elev_from_dir(direction):
    x, y, z = direction
    y = max(-1.0, min(1.0, y))
    azim = math.degrees(math.atan2(x, z))
    elev = math.degrees(math.asin(y))
    return azim, elev

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

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps from Fibonacci views using PGSR")
    parser.add_argument("--ply", type=str, required=True, help="Path to trained ply file")
    parser.add_argument("--output-root", type=str, required=True, help="Output root directory")
    parser.add_argument("--n-cams", type=int, default=30, help="Number of Fibonacci views")
    parser.add_argument("--upper-hemisphere", action="store_true", default=True, help="Use upper hemisphere")
    parser.add_argument("--lower-hemisphere", action="store_true", help="Use lower hemisphere")
    parser.add_argument("--opencv-y-down", action="store_true", default=True, help="Use OpenCV y-down convention (upper hemisphere has y <= 0)")
    
    # Camera intrinsic overrides
    parser.add_argument("--radius", type=float, default=3.0, help="Camera distance from target")
    parser.add_argument("--fx", type=float, default=1255.0, help="Focal length fx")
    parser.add_argument("--fy", type=float, default=1255.0, help="Focal length fy")
    parser.add_argument("--cx", type=float, default=640.0, help="Principal point cx")
    parser.add_argument("--cy", type=float, default=640.0, help="Principal point cy")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=1280, help="Render height")
    
    # Masking and YAML config arguments
    parser.add_argument("--config", type=str, default="../splat_merge/configs.yaml", help="Path to configs.yaml")
    parser.add_argument("--scene", type=str, default="pgsrlego", help="Scene name in configs.yaml")
    parser.add_argument("--threshold", type=float, default=0.05, help="Default threshold if not in configs.yaml")
    parser.add_argument("--dilate", type=int, default=3, help="Default dilate iterations if not in configs.yaml")
    
    # Dummy args to allow compatibility with previous shell configs
    parser.add_argument("--scenes", type=str, nargs="*", help="Scene name(s) (unused)")
    parser.add_argument("--preview-only", action="store_true", help="Unused dummy argument")
    args = parser.parse_args()

    # Determine threshold and dilate iterations from configs.yaml if available
    threshold = args.threshold
    dilate_iterations = args.dilate
    if os.path.exists(args.config):
        print(f"Loading YAML config from {args.config}...")
        try:
            defaults, scenes = load_simple_yaml(args.config)
            scene = next((s for s in scenes if s.get("name") == args.scene), None)
            if scene is not None:
                threshold = scene.get("threshold", defaults.get("threshold", threshold))
                dilate_iterations = scene.get("dilate", defaults.get("dilate", dilate_iterations))
                print(f"Loaded config for scene '{args.scene}': threshold={threshold}, dilate={dilate_iterations}")
            else:
                threshold = defaults.get("threshold", threshold)
                dilate_iterations = defaults.get("dilate", dilate_iterations)
                print(f"Scene '{args.scene}' not found in config. Using defaults: threshold={threshold}, dilate={dilate_iterations}")
        except Exception as e:
            print(f"Error parsing YAML config: {e}. Using CLI defaults.")
    else:
        print(f"Config path {args.config} not found. Using CLI defaults: threshold={threshold}, dilate={dilate_iterations}")

    # Determine upper/lower hemisphere
    upper = not args.lower_hemisphere

    # Generate Fibonacci camera directions
    print(f"Generating {args.n_cams} Fibonacci views on the {'upper' if upper else 'lower'} hemisphere...")
    directions = fibonacci_hemisphere_dirs(
        n_points=args.n_cams,
        upper=upper,
        y_down=args.opencv_y_down
    )

    # 1. Load Gaussian model once
    print("Determining SH degree of PLY...")
    sh_degree = get_sh_degree_from_ply(args.ply)
    print(f"SH degree: {sh_degree}")

    print("Loading Gaussian Model...")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(args.ply)

    print("Reading splats to compute exact object center median...")
    ply = PlyData.read(args.ply)
    v = ply['vertex'].data
    x_median = np.median(v['x'])
    y_median = np.median(v['y'])
    z_median = np.median(v['z'])
    object_center = np.array([x_median, y_median, z_median], dtype=np.float32)
    print(f"Object center (median xyz): {object_center}")

    pipeline = PipelineParamsDummy()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # 2. Loop through each view
    for cam_idx, direction in enumerate(directions, start=1):
        cam_folder_name = f"cam_{cam_idx:02d}"
        out_dir = os.path.join(args.output_root, cam_folder_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Processing View {cam_idx:02d}/{args.n_cams} ({cam_folder_name}) ---")
        
        # Compute extrinsics (direction is look-at direction vector)
        camera_center = object_center + args.radius * np.array(direction)
        
        # In world coordinates, standard up is [0, 1, 0] (y is up)
        w2c, c2w = look_at(camera_center, target=object_center, up=np.array([0., 1., 0.]))

        # Build Intrinsic matrix
        K = np.eye(3)
        K[0, 0] = args.fx
        K[1, 1] = args.fy
        K[0, 2] = args.cx
        K[1, 2] = args.cy

        # Save camera_main_view.txt
        txt_path = os.path.join(out_dir, "camera_main_view.txt")
        save_camera_txt(txt_path, K, w2c, c2w, width=args.width, height=args.height)
        print(f"Saved camera view file to {txt_path}")

        # Instantiate camera viewpoint for rendering
        fov_x = focal2fov(args.fx, args.width)
        fov_y = focal2fov(args.fy, args.height)
        
        # Camera class expects R transposed (cam2world rotation) and T (w2c translation)
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

        # Render
        with torch.no_grad():
            out = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                return_plane=True,
                return_depth_normal=True
            )

        # 1. Save standard RGB image
        if "render" in out:
            rendering = out["render"].clamp(0.0, 1.0)
            rendering_np = (rendering.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
            rgb_path = os.path.join(out_dir, "main_view_rgb.png")
            Image.fromarray(rendering_np).save(rgb_path)
            print(f"Saved RGB to {rgb_path}")

        # 2. Save Alpha map
        if "rendered_alpha" in out:
            alpha = out["rendered_alpha"].squeeze().cpu().numpy()
            alpha_np = (alpha * 255.0).clip(0, 255).astype(np.uint8)
            alpha_path = os.path.join(out_dir, "main_view_alpha.png")
            Image.fromarray(alpha_np).save(alpha_path)
            print(f"Saved Alpha map to {alpha_path}")
            
            # Save Alpha as PFM
            alpha_pfm_path = os.path.join(out_dir, "main_view_alpha.pfm")
            write_pfm(alpha_pfm_path, alpha.astype(np.float32))
            print(f"Saved Alpha PFM to {alpha_pfm_path}")

        # Extract depth
        plane_depth = out["plane_depth"].squeeze().cpu().numpy()
        plane_depth[plane_depth < 0.0] = 0.0
        if "rendered_alpha" in out:
            alpha = out["rendered_alpha"].squeeze().cpu().numpy()
            plane_depth[alpha < 1e-3] = 0.0

        # Compute edge mask and dilate it
        edge_mask = find_edge(plane_depth, threshold, exponent=-1.0)
        dilated_edge = dilate_mask(edge_mask, dilate_iterations)
        
        # Apply mask (setTo 0.0f where edge mask is active)
        depth_masked = plane_depth.copy()
        depth_masked[dilated_edge] = 0.0

        # Save dilated edge mask as PNG
        edge_mask_png = (dilated_edge * 255).astype(np.uint8)
        edge_mask_path = os.path.join(out_dir, "main_view_edge_mask.png")
        Image.fromarray(edge_mask_png).save(edge_mask_path)
        print(f"Saved edge mask to {edge_mask_path}")

        # 3. Save raw depth map (.pfm)
        depth_pfm_path = os.path.join(out_dir, "main_view_depth.pfm")
        write_pfm(depth_pfm_path, plane_depth.astype(np.float32))
        print(f"Saved raw depth PFM to {depth_pfm_path}")

        # 4. Save raw depth map masked (.pfm)
        depth_masked_pfm_path = os.path.join(out_dir, "main_view_depth_masked.pfm")
        write_pfm(depth_masked_pfm_path, depth_masked.astype(np.float32))
        print(f"Saved raw depth masked PFM to {depth_masked_pfm_path}")

        # 5. Save visualized depth map (.png)
        depth_min, depth_max = plane_depth.min(), plane_depth.max()
        depth_normalized = (plane_depth - depth_min) / (depth_max - depth_min + 1e-20)
        depth_color = jet_colormap(depth_normalized)
        color_depth_path = os.path.join(out_dir, "main_view_depth.png")
        Image.fromarray(depth_color).save(color_depth_path)
        print(f"Saved colorized depth map to {color_depth_path}")

    print("\nAll views processed successfully!")

if __name__ == "__main__":
    main()
