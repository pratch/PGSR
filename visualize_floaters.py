import sys
import os
import types
import torch

# ==============================================================================
# Mocks to satisfy project-internal imports without installing external packages
# ==============================================================================
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

pytorch3d = types.ModuleType("pytorch3d")
sys.modules["pytorch3d"] = pytorch3d

pytorch3d_transforms = types.ModuleType("pytorch3d.transforms")
pytorch3d_transforms.quaternion_to_matrix = quaternion_to_matrix
sys.modules["pytorch3d.transforms"] = pytorch3d_transforms

cv2_mock = types.ModuleType("cv2")
sys.modules["cv2"] = cv2_mock

# Add project root to python path to resolve local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# Remaining Imports
# ==============================================================================
import math
import argparse
import numpy as np
from plyfile import PlyData
from PIL import Image

from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, getProjectionMatrixCenterShift
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

class PipelineParamsDummy:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

# ==============================================================================
# Colormap Helpers (Vectorized NumPy implementation)
# ==============================================================================
def color_ramp(x, colors):
    """
    Linearly interpolates a normalized 2D array x (values in [0, 1]) 
    through a list of RGB colors to create a colorized map.
    """
    x = np.clip(x, 0.0, 1.0)
    num_colors = len(colors)
    scaled_x = x * (num_colors - 1)
    idx = np.floor(scaled_x).astype(int)
    idx = np.clip(idx, 0, num_colors - 2)
    frac = scaled_x - idx
    frac = frac[..., None]
    
    colors_np = np.array(colors)
    c0 = colors_np[idx]
    c1 = colors_np[idx + 1]
    
    out = (1.0 - frac) * c0 + frac * c1
    return (out * 255.0).astype(np.uint8)

def get_colormap(name):
    # Premium color maps designed to look beautiful and professional
    if name == "magma":
        colors = [
            [0.00, 0.00, 0.03], # Black
            [0.09, 0.04, 0.22], # Dark Purple
            [0.31, 0.07, 0.49], # Purple
            [0.58, 0.15, 0.40], # Magenta
            [0.82, 0.26, 0.25], # Orange-Red
            [0.97, 0.50, 0.20], # Orange
            [0.99, 0.77, 0.35], # Yellow-Orange
            [0.98, 0.98, 0.74]  # Pale Yellow
        ]
    elif name == "plasma":
        colors = [
            [0.05, 0.03, 0.53], # Dark Blue
            [0.34, 0.01, 0.67], # Purple
            [0.57, 0.09, 0.68], # Violet
            [0.76, 0.23, 0.54], # Pink-Red
            [0.90, 0.41, 0.39], # Salmon
            [0.98, 0.64, 0.23], # Orange
            [0.94, 0.89, 0.13]  # Bright Yellow
        ]
    elif name == "inferno":
        colors = [
            [0.00, 0.00, 0.00], # Black
            [0.08, 0.02, 0.18], # Deep Purple
            [0.26, 0.03, 0.40], # Purple
            [0.47, 0.08, 0.47], # Magenta
            [0.70, 0.18, 0.40], # Hot Pink
            [0.89, 0.36, 0.27], # Orange
            [0.98, 0.61, 0.17], # Bright Orange
            [0.98, 0.88, 0.44]  # Pale Yellow
        ]
    elif name == "rdbu":
        # Diverging Red-White-Blue
        colors = [
            [0.02, 0.19, 0.44], # Dark Blue (Negative: GS further than Opaque)
            [0.26, 0.57, 0.77], # Light Blue
            [0.90, 0.90, 0.90], # Greyish White (Near zero difference)
            [0.96, 0.56, 0.42], # Light Red
            [0.67, 0.00, 0.15]  # Dark Red (Positive: GS closer/protruding due to floaters)
        ]
    elif name == "jet":
        colors = [
            [0.0, 0.0, 0.5], # Blue
            [0.0, 0.5, 1.0], # Cyan
            [0.5, 1.0, 0.5], # Green
            [1.0, 1.0, 0.0], # Yellow
            [1.0, 0.0, 0.0]  # Red
        ]
    else:  # Default to viridis-like
        colors = [
            [0.27, 0.00, 0.33],
            [0.23, 0.28, 0.54],
            [0.13, 0.57, 0.55],
            [0.37, 0.81, 0.42],
            [0.99, 0.91, 0.13]
        ]
    return colors

# ==============================================================================
# Helper to parse camera view parameters
# ==============================================================================
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
            intr = []
            for _ in range(3):
                idx += 1
                intr.append([float(x) for x in lines[idx].strip().split()])
            params['K'] = np.array(intr)
        elif line.startswith("extrinsic_world_to_camera_4x4:"):
            w2c = []
            for _ in range(4):
                idx += 1
                w2c.append([float(x) for x in lines[idx].strip().split()])
            params['W2C'] = np.array(w2c)
        idx += 1
    return params

def get_sh_degree_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    N_rest = len(extra_f_names)
    sh_degree = int(math.sqrt((N_rest + 3) / 3)) - 1
    return sh_degree

# ==============================================================================
# Custom render function that takes all_map directly
# ==============================================================================
def render_custom(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, input_all_map: torch.Tensor, scaling_modifier = 1.0):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    shs = pc.get_features
    colors_precomp = None

    raster_settings = PlaneGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=True,
        debug=pipe.debug
    )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    return out_all_map

# ==============================================================================
# HTML Dashboard Generator
# ==============================================================================
def generate_html_report(output_dir, threshold, view_name="View 17"):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PGSR Floater Analysis Dashboard - {view_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            padding: 2.5rem;
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }}
        
        h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        
        .container {{
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
            align-items: start;
        }}
        
        @media (max-width: 1024px) {{
            .container {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        
        .panel h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            border-left: 4px solid var(--accent-primary);
            padding-left: 0.75rem;
        }}
        
        /* Interactive Slider Styles */
        .slider-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .controls-row {{
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}
        
        .control-group label {{
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        select {{
            background: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem;
            font-size: 0.9rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }}
        
        select:focus {{
            border-color: var(--accent-primary);
        }}
        
        .slider-wrapper {{
            position: relative;
            width: 100%;
            max-width: 600px;
            aspect-ratio: 1;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-color);
        }}
        
        .slider-img {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
            user-select: none;
        }}
        
        #overlay-img {{
            z-index: 2;
        }}
        
        .slider-range-input {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: ew-resize;
            z-index: 10;
        }}
        
        .slider-handle-line {{
            position: absolute;
            top: 0;
            height: 100%;
            width: 2px;
            background: #fff;
            z-index: 5;
            pointer-events: none;
            box-shadow: 0 0 10px rgba(255,255,255,0.8);
        }}
        
        .slider-handle-button {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            background: var(--accent-primary);
            border: 2px solid #fff;
            color: #fff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            z-index: 6;
            pointer-events: none;
            transition: background 0.2s;
        }}
        
        .slider-wrapper:hover .slider-handle-button {{
            background: #2563eb;
        }}
        
        /* Grid and thumbnails */
        .maps-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1rem;
        }}
        
        .map-card {{
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
        }}
        
        .map-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-primary);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        }}
        
        .map-card.active {{
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }}
        
        .map-card img {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            display: block;
        }}
        
        .map-info {{
            padding: 0.8rem;
        }}
        
        .map-title {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        
        .map-desc {{
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.3;
        }}
        
        /* Information Section */
        .info-panel {{
            margin-top: 1.5rem;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.9rem;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: var(--text-muted);
        }}
        
        .metric-value {{
            font-weight: 600;
        }}
        
        .pill {{
            padding: 0.1rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-primary);
        }}
    </style>
</head>
<body>
    <header>
        <h1>PGSR Floater & Depth Analysis Dashboard</h1>
        <div class="subtitle">Visualizing semi-transparent floating Gaussians and their effect on rendered depth for <strong>{view_name}</strong></div>
    </header>
    
    <div class="container">
        <!-- Interactive Split Slider -->
        <div class="panel">
            <h2>Interactive Comparison</h2>
            <div class="slider-section">
                <div class="controls-row">
                    <div class="control-group">
                        <label for="base-select">Base Image (Left)</label>
                        <select id="base-select" onchange="changeImages()">
                            <option value="rgb.png" selected>Standard RGB</option>
                            <option value="depth_standard.png">Standard Depth</option>
                            <option value="depth_opaque.png">Opaque-only Depth</option>
                            <option value="depth_diff_signed.png">Signed Depth Discrepancy</option>
                            <option value="floater_weight.png">Floating Gaussian Weight</option>
                            <option value="rgb_floater_overlay.png">RGB with Floater Overlay</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="overlay-select">Overlay Image (Right)</label>
                        <select id="overlay-select" onchange="changeImages()">
                            <option value="rgb.png">Standard RGB</option>
                            <option value="depth_standard.png">Standard Depth</option>
                            <option value="depth_opaque.png">Opaque-only Depth</option>
                            <option value="depth_diff_signed.png">Signed Depth Discrepancy</option>
                            <option value="floater_weight.png" selected>Floating Gaussian Weight</option>
                            <option value="rgb_floater_overlay.png">RGB with Floater Overlay</option>
                        </select>
                    </div>
                </div>
                
                <div class="slider-wrapper">
                    <img src="rgb.png" class="slider-img" id="base-img" alt="Base Image">
                    <img src="floater_weight.png" class="slider-img" id="overlay-img" alt="Overlay Image" style="clip-path: inset(0 50% 0 0);">
                    <div class="slider-handle-line" id="handle-line" style="left: 50%;"></div>
                    <div class="slider-handle-button" id="handle-btn" style="left: 50%;">↔</div>
                    <input type="range" min="0" max="100" value="50" class="slider-range-input" id="range-slider" oninput="slide(this.value)">
                </div>
            </div>
            
            <div class="info-panel">
                <div class="metric-row">
                    <span class="metric-label">Target Viewport</span>
                    <span class="metric-value">{view_name} (0-indexed view 17, cam_18)</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Floater Opacity Threshold</span>
                    <span class="metric-value"><span class="pill">&lt; {threshold}</span></span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Protrusion Insight</span>
                    <span class="metric-value" style="color: #60a5fa; max-width: 70%; text-align: right;">
                        Positive depth difference (Red) indicates standard depth is pulled closer than the opaque surface.
                    </span>
                </div>
            </div>
        </div>
        
        <!-- Grid of All Rendered Maps -->
        <div class="panel">
            <h2>Rendered Output Maps</h2>
            <div class="maps-grid">
                <div class="map-card" onclick="selectCard('rgb.png')">
                    <img src="rgb.png" alt="RGB">
                    <div class="map-info">
                        <div class="map-title">Standard RGB</div>
                        <div class="map-desc">Full 3D Gaussian Splatting rendering of the scene from this camera viewpoint.</div>
                    </div>
                </div>
                
                <div class="map-card" onclick="selectCard('depth_standard.png')">
                    <img src="depth_standard.png" alt="Standard Depth">
                    <div class="map-info">
                        <div class="map-title">Standard Depth Map</div>
                        <div class="map-desc">Depth calculated from all Gaussians along the ray, showing typical protrusions.</div>
                    </div>
                </div>
                
                <div class="map-card" onclick="selectCard('depth_opaque.png')">
                    <img src="depth_opaque.png" alt="Opaque Depth">
                    <div class="map-info">
                        <div class="map-title">Opaque-only Depth Map</div>
                        <div class="map-desc">Depth calculated ignoring semi-transparent floaters (opacity &lt; {threshold}).</div>
                    </div>
                </div>
                
                <div class="map-card" onclick="selectCard('floater_weight.png')">
                    <img src="floater_weight.png" alt="Floater Weight">
                    <div class="map-info">
                        <div class="map-title">Floating Gaussian Weight</div>
                        <div class="map-desc">Accumulated blending weight of floaters. Brighter areas indicate high floater influence.</div>
                    </div>
                </div>
                
                <div class="map-card" onclick="selectCard('depth_diff_signed.png')">
                    <img src="depth_diff_signed.png" alt="Depth Difference">
                    <div class="map-info">
                        <div class="map-title">Signed Depth Discrepancy</div>
                        <div class="map-desc">Red indicates standard depth is protruding towards the camera compared to the opaque surface.</div>
                    </div>
                </div>
                
                <div class="map-card" onclick="selectCard('rgb_floater_overlay.png')">
                    <img src="rgb_floater_overlay.png" alt="Overlay">
                    <div class="map-info">
                        <div class="map-title">RGB Floater Overlay</div>
                        <div class="map-desc">Floater weight map overlaid onto the RGB render to locate where floaters appear visually.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function slide(val) {{
            const overlay = document.getElementById('overlay-img');
            const line = document.getElementById('handle-line');
            const btn = document.getElementById('handle-btn');
            
            overlay.style.clipPath = `inset(0 ${{100 - val}}% 0 0)`;
            line.style.left = `${{val}}%`;
            btn.style.left = `${{val}}%`;
        }}
        
        function changeImages() {{
            const baseSelect = document.getElementById('base-select');
            const overlaySelect = document.getElementById('overlay-select');
            
            document.getElementById('base-img').src = baseSelect.value;
            document.getElementById('overlay-img').src = overlaySelect.value;
        }}
        
        function selectCard(imgName) {{
            // Find which select to update - let's set it as the overlay
            const overlaySelect = document.getElementById('overlay-select');
            overlaySelect.value = imgName;
            changeImages();
            
            // Highlight card
            const cards = document.querySelectorAll('.map-card');
            cards.forEach(card => {{
                const img = card.querySelector('img');
                if (img.getAttribute('src') === imgName) {{
                    card.classList.add('active');
                }} else {{
                    card.classList.remove('active');
                }}
            }});
        }}
        
        // Initialize highlight
        selectCard('floater_weight.png');
    </script>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated interactive dashboard: {html_path}")

def generate_central_html_report(output_dir, cams_list, threshold):
    cams_json = str(cams_list)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PGSR Floater Analysis Central Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --panel-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --columns: 4;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            padding: 2.5rem;
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        
        .header-left {{
            max-width: 60%;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1rem;
        }}
        
        .panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }}
        
        /* Controls Row */
        .controls-panel {{
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            align-items: center;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .control-group label {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        select, input[type="range"] {{
            background: #1e293b;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }}
        
        select:focus {{
            border-color: var(--accent-primary);
        }}
        
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        
        .slider-val {{
            font-weight: 600;
            min-width: 1.5rem;
            text-align: center;
        }}
        
        /* Grid Layout */
        .grid {{
            display: grid;
            grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
            gap: 1.5rem;
        }}
        
        .grid-item {{
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }}
        
        .grid-item:hover {{
            transform: translateY(-4px);
            border-color: var(--accent-primary);
            box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
        }}
        
        .img-container {{
            width: 100%;
            aspect-ratio: 1;
            position: relative;
            cursor: zoom-in;
            background: #000;
            overflow: hidden;
        }}
        
        .grid-img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: transform 0.3s ease;
        }}
        
        .grid-item:hover .grid-img {{
            transform: scale(1.02);
        }}
        
        .card-info {{
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid var(--border-color);
            background: rgba(30, 41, 59, 0.4);
        }}
        
        .card-title {{
            font-size: 0.95rem;
            font-weight: 600;
        }}
        
        .btn-view {{
            background: var(--accent-primary);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.2s;
        }}
        
        .btn-view:hover {{
            background: #2563eb;
        }}
        
        /* Lightbox modal */
        .lightbox {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            z-index: 100;
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        
        .lightbox-content-wrapper {{
            position: relative;
            width: 85vw;
            height: 80vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .lightbox-img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .lightbox-close {{
            position: absolute;
            top: 2rem;
            right: 2rem;
            background: rgba(255,255,255,0.1);
            border: none;
            color: #fff;
            font-size: 1.5rem;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}
        
        .lightbox-close:hover {{
            background: rgba(255,255,255,0.25);
        }}
        
        .lightbox-nav {{
            position: absolute;
            background: rgba(255,255,255,0.1);
            border: none;
            color: #fff;
            font-size: 1.8rem;
            width: 55px;
            height: 55px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s, transform 0.1s;
        }}
        
        .lightbox-nav:hover {{
            background: rgba(255,255,255,0.25);
        }}
        
        .lightbox-nav:active {{
            transform: scale(0.95);
        }}
        
        .lightbox-prev {{
            left: 2rem;
        }}
        
        .lightbox-next {{
            right: 2rem;
        }}
        
        .lightbox-caption {{
            color: var(--text-main);
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 1.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
        
        .lightbox-subcaption {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        
        /* Stats Pill */
        .pill {{
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-primary);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <h1>PGSR Multi-View Floater Analysis</h1>
            <div class="subtitle">Central visualizer for all viewpoints showing semi-transparent floater distribution and depth deviations (Threshold &lt; {threshold})</div>
        </div>
        <div>
            <span class="pill">{len(cams_list)} Views Loaded</span>
        </div>
    </header>
    
    <div class="panel controls-panel">
        <div class="control-group">
            <label for="map-select">Visualize Map Type</label>
            <select id="map-select" onchange="updateMap(this.value)">
                <option value="rgb">Standard RGB</option>
                <option value="depth_std">Standard Depth Map</option>
                <option value="depth_opaque">Opaque-only Depth Map</option>
                <option value="floater_weight" selected>Floating Gaussian Weight</option>
                <option value="depth_diff_signed">Signed Depth Discrepancy (Red = Protrusion)</option>
                <option value="overlay">RGB Floater Overlay</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="col-range">Grid Columns</label>
            <div class="slider-container">
                <input type="range" id="col-range" min="2" max="6" value="4" oninput="updateColumns(this.value)">
                <span class="slider-val" id="col-val">4</span>
            </div>
        </div>
    </div>
    
    <div class="grid" id="grid-container">
        <!-- GRID_ITEMS_PLACEHOLDER -->
    </div>
    
    <!-- Lightbox Modal -->
    <div class="lightbox" id="lightbox-modal">
        <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
        <button class="lightbox-nav lightbox-prev" onclick="navigateLightbox(-1)">&lsaquo;</button>
        <button class="lightbox-nav lightbox-next" onclick="navigateLightbox(1)">&rsaquo;</button>
        <div class="lightbox-content-wrapper">
            <img src="" class="lightbox-img" id="lightbox-image" alt="Lightbox Visual">
        </div>
        <div class="lightbox-caption" id="lightbox-title">Camera Name</div>
        <div class="lightbox-subcaption" id="lightbox-desc">Map Type</div>
    </div>
    
    <script>
        const cams = {cams_json};
        let currentCamIdx = 0;
        
        const mapFiles = {{
            'rgb': 'rgb.png',
            'depth_std': 'depth_standard.png',
            'depth_opaque': 'depth_opaque.png',
            'floater_weight': 'floater_weight.png',
            'depth_diff_signed': 'depth_diff_signed.png',
            'overlay': 'rgb_floater_overlay.png'
        }};
        
        const mapNames = {{
            'rgb': 'Standard RGB Render',
            'depth_std': 'Standard Depth Map (Plasma)',
            'depth_opaque': 'Opaque-only Depth Map (Plasma)',
            'floater_weight': 'Floating Gaussian Weight Map (Magma)',
            'depth_diff_signed': 'Signed Depth Discrepancy (Red = Protrusion, Blue = Recession)',
            'overlay': 'RGB with Floater Weight Overlay'
        }};
        
        function updateColumns(val) {{
            document.getElementById('grid-container').style.setProperty('--columns', val);
            document.getElementById('col-val').innerText = val;
        }}
        
        function updateMap(mapKey) {{
            const fileName = mapFiles[mapKey];
            const imgs = document.querySelectorAll('.grid-img');
            imgs.forEach(img => {{
                const cam = img.dataset.cam;
                img.src = `${{cam}}/${{fileName}}`;
            }});
            
            // If lightbox is open, update its image too
            const lightbox = document.getElementById('lightbox-modal');
            if (lightbox.style.display === 'flex') {{
                openLightbox(currentCamIdx);
            }}
        }}
        
        function openLightbox(idx) {{
            currentCamIdx = idx;
            const camName = cams[idx];
            const mapKey = document.getElementById('map-select').value;
            const fileName = mapFiles[mapKey];
            
            const imagePath = `${{camName}}/${{fileName}}`;
            
            document.getElementById('lightbox-image').src = imagePath;
            document.getElementById('lightbox-title').innerText = camName.toUpperCase();
            document.getElementById('lightbox-desc').innerText = mapNames[mapKey];
            document.getElementById('lightbox-modal').style.display = 'flex';
        }}
        
        function closeLightbox() {{
            document.getElementById('lightbox-modal').style.display = 'none';
        }}
        
        function navigateLightbox(dir) {{
            let nextIdx = currentCamIdx + dir;
            if (nextIdx < 0) nextIdx = cams.length - 1;
            if (nextIdx >= cams.length) nextIdx = 0;
            openLightbox(nextIdx);
        }}
        
        // Key bindings for lightbox navigation
        document.addEventListener('keydown', function(e) {{
            const lightbox = document.getElementById('lightbox-modal');
            if (lightbox.style.display === 'flex') {{
                if (e.key === 'ArrowLeft') {{
                    navigateLightbox(-1);
                }} else if (e.key === 'ArrowRight') {{
                    navigateLightbox(1);
                }} else if (e.key === 'Escape') {{
                    closeLightbox();
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    grid_items = []
    for idx, cam in enumerate(cams_list):
        grid_items.append(f"""
        <div class="grid-item">
            <div class="img-container" onclick="openLightbox({idx})">
                <img src="{cam}/floater_weight.png" class="grid-img" data-cam="{cam}" alt="{cam}">
            </div>
            <div class="card-info">
                <span class="card-title">{cam}</span>
                <a href="{cam}/index.html" class="btn-view" target="_blank">Detail View</a>
            </div>
        </div>""")
        
    grid_html = "\n".join(grid_items)
    html_content = html_content.replace("<!-- GRID_ITEMS_PLACEHOLDER -->", grid_html)
    
    central_html_path = os.path.join(output_dir, "index.html")
    with open(central_html_path, "w") as f:
        f.write(html_content)
    print(f"Generated central dashboard: {central_html_path}")

# ==============================================================================
# Main Process
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize semi-transparent floaters and depth discrepancies")
    parser.add_argument("--ply", type=str, default="splats/pgsrlego.ply", help="Path to trained ply file")
    parser.add_argument("--camera", type=str, default="output_multicams_tsdf/pgsrlego/cam_18/camera_main_view.txt", help="Path to camera view txt file")
    parser.add_argument("--camera_dir", type=str, default=None, help="Path to directory containing camera folders (cam_xx/camera_main_view.txt)")
    parser.add_argument("--opacity_threshold", "-t", type=float, default=0.2, help="Opacity threshold below which a Gaussian is considered a floater")
    parser.add_argument("--output_dir", "-o", type=str, default="output_vis_floater", help="Output directory to save images")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory initialized at: {args.output_dir}")

    # Determine cameras to process
    cams_to_process = []
    if args.camera_dir is not None:
        if not os.path.exists(args.camera_dir):
            print(f"Error: Camera directory {args.camera_dir} does not exist!")
            return
        # Find subdirectories that contain camera_main_view.txt
        for d in os.listdir(args.camera_dir):
            cam_path = os.path.join(args.camera_dir, d, "camera_main_view.txt")
            if os.path.isfile(cam_path):
                cams_to_process.append((cam_path, d))
        cams_to_process.sort(key=lambda x: x[1])
        print(f"Found {len(cams_to_process)} camera viewpoints in directory: {args.camera_dir}")
    else:
        if not os.path.exists(args.camera):
            print(f"Error: Camera file {args.camera} does not exist!")
            return
        cam_dir_name = os.path.basename(os.path.dirname(args.camera))
        if not cam_dir_name:
            cam_dir_name = "cam_view"
        cams_to_process.append((args.camera, cam_dir_name))

    if len(cams_to_process) == 0:
        print("Error: No camera views to process.")
        return

    # 1. Load Gaussian model (do this once)
    print(f"Loading PLY file to determine SH degree...")
    if not os.path.exists(args.ply):
        print(f"Error: PLY file {args.ply} does not exist!")
        return
    sh_degree = get_sh_degree_from_ply(args.ply)
    print(f"SH degree: {sh_degree}")

    print("Loading Gaussian Model parameters into GPU...")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(args.ply)

    pipeline = PipelineParamsDummy()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # 2. Loop through all viewpoints
    for cam_idx, (cam_path, cam_dir_name) in enumerate(cams_to_process):
        # Determine camera-specific output directory
        if args.camera_dir is not None:
            cam_output_dir = os.path.join(args.output_dir, cam_dir_name)
        else:
            cam_output_dir = args.output_dir
        
        os.makedirs(cam_output_dir, exist_ok=True)
        print(f"\n[{cam_idx+1}/{len(cams_to_process)}] Processing camera {cam_dir_name} -> {cam_output_dir}")

        # Parse camera parameters
        cam_params = parse_camera_txt(cam_path)
        width = cam_params['width']
        height = cam_params['height']
        K = cam_params['K']
        w2c = cam_params['W2C']

        # Set up extrinsics/intrinsics for viewpoint
        R_w2c = w2c[:3, :3]
        T_w2c = w2c[:3, 3]
        R = R_w2c.T
        T = T_w2c
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Compute field of view
        fov_x = focal2fov(fx, width)
        fov_y = focal2fov(fy, height)

        # Instantiate viewpoint camera
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
        view_cam.Cx = cx
        view_cam.Cy = cy
        view_cam.Fx = fx
        view_cam.Fy = fy

        if not np.isclose(cx, width / 2.0) or not np.isclose(cy, height / 2.0):
            view_cam.projection_matrix = getProjectionMatrixCenterShift(
                znear=view_cam.znear, zfar=view_cam.zfar,
                cx=cx, cy=cy, fl_x=fx, fl_y=fy,
                w=width, h=height
            ).transpose(0, 1).cuda()
            view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)

        # Render Pass 1: Standard Rendering
        with torch.no_grad():
            out = render(
                viewpoint_camera=view_cam,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                return_plane=True,
                return_depth_normal=True
            )

        # Extract RGB and depth maps
        rgb_torch = out["render"].clamp(0.0, 1.0)
        rgb_np = (rgb_torch.permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        
        depth_standard_np = out["plane_depth"].squeeze().cpu().numpy()
        depth_standard_np[depth_standard_np < 0.0] = 0.0
        if "rendered_alpha" in out:
            alpha_np = out["rendered_alpha"].squeeze().cpu().numpy()
            depth_standard_np[alpha_np < 1e-3] = 0.0

        # Render Pass 2: Custom Floater Analysis
        means3D = gaussians.get_xyz
        pts_in_cam = means3D @ view_cam.world_view_transform[:3,:3] + view_cam.world_view_transform[3,:3]
        depth_z = pts_in_cam[:, 2]  # (N,)
        
        opacity = gaussians.get_opacity  # (N, 1)
        
        # Construct input_all_map:
        # Channel 0: depth_z (Gaussian depth along camera axis)
        # Channel 1: floaters (1.0 if opacity < threshold else 0.0)
        # Channel 2: depth_z of opaque (depth_z if opacity >= threshold else 0.0)
        # Channel 3: opaque (1.0 if opacity >= threshold else 0.0)
        # Channel 4: depth_z of floaters (depth_z if opacity < threshold else 0.0)
        input_all_map = torch.zeros((means3D.shape[0], 5), dtype=torch.float32, device="cuda")
        input_all_map[:, 0] = depth_z
        input_all_map[:, 1] = (opacity[:, 0] < args.opacity_threshold).float()
        input_all_map[:, 2] = depth_z * (opacity[:, 0] >= args.opacity_threshold).float()
        input_all_map[:, 3] = (opacity[:, 0] >= args.opacity_threshold).float()
        input_all_map[:, 4] = depth_z * (opacity[:, 0] < args.opacity_threshold).float()

        with torch.no_grad():
            vis_map_torch = render_custom(
                viewpoint_camera=view_cam,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                input_all_map=input_all_map
            )
        
        vis_map = vis_map_torch.cpu().numpy()  # (5, H, W)
        
        vis_depth_all = vis_map[0]
        vis_weight_translucent = vis_map[1]
        vis_depth_opaque_accum = vis_map[2]
        vis_weight_opaque = vis_map[3]
        vis_depth_translucent_accum = vis_map[4]

        # Compute derived maps
        depth_opaque = np.zeros_like(vis_depth_opaque_accum)
        mask_opaque = vis_weight_opaque > 1e-4
        depth_opaque[mask_opaque] = vis_depth_opaque_accum[mask_opaque] / vis_weight_opaque[mask_opaque]
        depth_opaque[~mask_opaque] = 0.0

        depth_translucent = np.zeros_like(vis_depth_translucent_accum)
        mask_translucent = vis_weight_translucent > 1e-4
        depth_translucent[mask_translucent] = vis_depth_translucent_accum[mask_translucent] / vis_weight_translucent[mask_translucent]
        depth_translucent[~mask_translucent] = 0.0

        # Depth discrepancy: Opaque depth - Standard Depth
        depth_diff = np.zeros_like(depth_opaque)
        valid_mask = (vis_weight_opaque > 1e-2) & (depth_standard_np > 0)
        depth_diff[valid_mask] = depth_opaque[valid_mask] - depth_standard_np[valid_mask]

        # Save visualizations
        rgb_path = os.path.join(cam_output_dir, "rgb.png")
        Image.fromarray(rgb_np).save(rgb_path)

        valid_depths = depth_standard_np[depth_standard_np > 0]
        d_min = valid_depths.min() if len(valid_depths) > 0 else 0.0
        d_max = valid_depths.max() if len(valid_depths) > 0 else 1.0

        def colorize_depth(depth_map, colormap_name, min_val, max_val):
            norm_map = (depth_map - min_val) / (max_val - min_val + 1e-8)
            norm_map[depth_map == 0] = 0.0
            colors = get_colormap(colormap_name)
            colorized = color_ramp(norm_map, colors)
            colorized[depth_map == 0] = 0
            return colorized

        # Save standard depth
        depth_std_color = colorize_depth(depth_standard_np, "plasma", d_min, d_max)
        depth_std_path = os.path.join(cam_output_dir, "depth_standard.png")
        Image.fromarray(depth_std_color).save(depth_std_path)

        # Save opaque-only depth
        depth_opaque_color = colorize_depth(depth_opaque, "plasma", d_min, d_max)
        depth_opaque_path = os.path.join(cam_output_dir, "depth_opaque.png")
        Image.fromarray(depth_opaque_color).save(depth_opaque_path)

        # Save floater weight map
        floater_weight_color = color_ramp(vis_weight_translucent, get_colormap("magma"))
        if "rendered_alpha" in out:
            floater_weight_color[alpha_np < 1e-3] = 0
        floater_weight_path = os.path.join(cam_output_dir, "floater_weight.png")
        Image.fromarray(floater_weight_color).save(floater_weight_path)

        # Save depth difference map (protrusion)
        max_diff = 0.15 * (d_max - d_min)
        protrude_map = np.clip(depth_diff, 0.0, max_diff)
        protrude_color = colorize_depth(protrude_map, "inferno", 0.0, max_diff)
        protrude_path = os.path.join(cam_output_dir, "depth_diff_protrude.png")
        Image.fromarray(protrude_color).save(protrude_path)

        # Save signed difference (Diverging Red-Blue)
        signed_diff_norm = (depth_diff + max_diff) / (2 * max_diff + 1e-8)
        signed_diff_norm[~valid_mask] = 0.5
        signed_color = color_ramp(signed_diff_norm, get_colormap("rdbu"))
        signed_color[~valid_mask] = 0
        signed_path = os.path.join(cam_output_dir, "depth_diff_signed.png")
        Image.fromarray(signed_color).save(signed_path)

        # Save RGB overlay of floater weight
        overlay_np = (rgb_np * 0.5 + floater_weight_color * 0.5).astype(np.uint8)
        overlay_path = os.path.join(cam_output_dir, "rgb_floater_overlay.png")
        Image.fromarray(overlay_np).save(overlay_path)

        # Save raw arrays
        np.save(os.path.join(cam_output_dir, "floater_weight.npy"), vis_weight_translucent)
        np.save(os.path.join(cam_output_dir, "depth_diff.npy"), depth_diff)

        # Generate interactive report
        generate_html_report(cam_output_dir, args.opacity_threshold, cam_dir_name)

    print(f"\nAll {len(cams_to_process)} viewpoints processed successfully!")

    # Generate central report for batch processing
    if args.camera_dir is not None:
        generate_central_html_report(args.output_dir, [c[1] for c in cams_to_process], args.opacity_threshold)

if __name__ == "__main__":
    main()
