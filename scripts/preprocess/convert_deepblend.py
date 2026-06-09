"""
Convert DeepBlending dataset with known camera poses (NVM format) to COLMAP format.
This script reads existing camera poses from NVM files instead of running COLMAP SfM.
"""

import os
import logging
import numpy as np
import json
from argparse import ArgumentParser
import shutil
from pathlib import Path
import sys

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]
sys.path.append(dir_path.__str__())

from database import COLMAPDatabase
from read_write_model import rotmat2qvec, write_model


def read_nvm_file(nvm_path):
    """Read camera poses from NVM file."""
    cameras = {}
    with open(nvm_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if line.startswith('NVM_V3'):
            line_idx += 1
            continue
        if line == '' or line.startswith('#'):
            line_idx += 1
            continue
        
        # Read number of cameras
        num_cameras = int(line)
        line_idx += 1
        
        # Read camera parameters
        for i in range(num_cameras):
            parts = lines[line_idx].strip().split()
            image_name = parts[0]
            
            # Remove 'images/' prefix if present (NVM format includes path)
            if image_name.startswith('images/'):
                image_name = image_name.replace('images/', '', 1)
            
            focal_length = float(parts[1])
            
            # Read quaternion (w, x, y, z) and translation
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            
            # Radial distortion
            radial_distortion = float(parts[9])
            
            cameras[image_name] = {
                'focal': focal_length,
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
                'radial': radial_distortion
            }
            line_idx += 1
        break
    
    return cameras


def create_colmap_model_from_nvm(nvm_cameras, image_dir, sparse_dir):
    """Create COLMAP model from NVM camera data."""
    import collections
    
    # Get image dimensions from first image
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    from PIL import Image
    first_img_path = os.path.join(image_dir, image_files[0])
    with Image.open(first_img_path) as img:
        width, height = img.size
    
    # Create COLMAP cameras dictionary
    Camera = collections.namedtuple(
        "Camera", ["id", "model", "width", "height", "params"])
    
    cameras_colmap = {}
    camera_id = 1
    
    # Create COLMAP images dictionary
    Image_colmap = collections.namedtuple(
        "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    
    images_colmap = {}
    
    for idx, image_name in enumerate(sorted(nvm_cameras.keys()), 1):
        cam_data = nvm_cameras[image_name]
        
        # For NVM, focal length is normalized. Convert to pixels
        # Assuming focal length is given as ratio to image width
        focal_pixels = cam_data['focal'] * max(width, height)
        
        # Create camera if not exists (assuming all cameras have same intrinsics)
        if camera_id not in cameras_colmap:
            # Use PINHOLE model: fx, fy, cx, cy
            params = np.array([focal_pixels, focal_pixels, width/2, height/2])
            cameras_colmap[camera_id] = Camera(
                id=camera_id,
                model='PINHOLE',
                width=width,
                height=height,
                params=params
            )
        
        # NVM uses camera-to-world quaternion, COLMAP uses world-to-camera
        # Convert from NVM to COLMAP convention
        qvec = cam_data['qvec']
        tvec = cam_data['tvec']
        
        # Create image entry
        images_colmap[idx] = Image_colmap(
            id=idx,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=np.zeros((0, 2)),  # No 2D points
            point3D_ids=np.full(0, -1, dtype=np.int64)  # No 3D points
        )
    
    # DeepBlending NVM files don't contain 3D points, so we create random initialization points
    # This is similar to how PGSR handles synthetic datasets
    logging.info("Creating random initialization point cloud (100k points)")
    Point3D = collections.namedtuple(
        "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
    
    points3D = {}
    num_pts = 100_000
    np.random.seed(42)  # For reproducibility
    # Generate random points in a reasonable space around origin
    for i in range(num_pts):
        xyz = np.random.random(3) * 6.0 - 3.0  # Points in [-3, 3] cube
        rgb = np.random.randint(0, 255, 3, dtype=np.uint8)
        points3D[i + 1] = Point3D(
            id=i + 1,
            xyz=xyz,
            rgb=rgb,
            error=0.0,
            image_ids=np.array([], dtype=np.int32),
            point2D_idxs=np.array([], dtype=np.int32)
        )
    
    # Write COLMAP model
    os.makedirs(sparse_dir, exist_ok=True)
    write_model(cameras_colmap, images_colmap, points3D, sparse_dir, ext='.bin')
    write_model(cameras_colmap, images_colmap, points3D, sparse_dir, ext='.txt')
    
    return cameras_colmap, images_colmap


def create_database(db_path, cameras_colmap, images_colmap):
    """Create COLMAP database with fixed poses."""
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    
    # Add cameras
    for camera_id, camera in cameras_colmap.items():
        model_id = 1  # PINHOLE
        db.add_camera(model_id, camera.width, camera.height, camera.params, camera_id=camera_id)
    
    # Add images
    for image_id, image in images_colmap.items():
        db.add_image(image.name, image.camera_id, image_id=image_id)
    
    db.commit()
    db.close()


def convert_deepblending(args):
    """Main conversion function."""
    colmap_command = '"{}"'.format(args.colmap_executable) if args.colmap_executable else "colmap"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.data_path, exist_ok=True)
    
    # Find all scenes in source data path (look for *_IBR_Inputs_Outputs directories)
    all_items = sorted(os.listdir(args.source_data_path))
    scene_dirs = [item for item in all_items if item.endswith('_IBR_Inputs_Outputs') and 
                  os.path.isdir(os.path.join(args.source_data_path, item))]
    
    # Extract scene names (remove _IBR_Inputs_Outputs suffix)
    scenes = [scene_dir.replace('_IBR_Inputs_Outputs', '') for scene_dir in scene_dirs]
    
    logging.info(f"Found {len(scenes)} scenes in {args.source_data_path}")
    logging.info(f"Scenes: {', '.join(scenes)}")
    
    processed_count = 0
    for scene in scenes:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing scene: {scene}")
        logging.info(f"{'='*60}")
        
        # Find NVM file and images in source dataset
        source_scene_dir = os.path.join(args.source_data_path, f"{scene}_IBR_Inputs_Outputs")
        nvm_dir = os.path.join(source_scene_dir, "input_camera_poses_as_nvm")
        
        # Try scene.nvm first (DeepBlending convention)
        nvm_path = os.path.join(nvm_dir, "scene.nvm")
        if not os.path.exists(nvm_path):
            # Fallback to cameras.nvm
            nvm_path = os.path.join(nvm_dir, "cameras.nvm")
        
        if not os.path.exists(nvm_path):
            logging.error(f"NVM file not found in {nvm_dir}")
            continue
        
        source_images_dir = os.path.join(nvm_dir, "images")
        if not os.path.exists(source_images_dir):
            logging.error(f"Images directory not found: {source_images_dir}")
            continue
        
        # Check if images exist
        image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not image_files:
            logging.warning(f"No images found in {source_images_dir}, skipping")
            continue
        
        logging.info(f"Found {len(image_files)} images in source")
        logging.info(f"Reading NVM from: {nvm_path}")
        
        # Create output scene directory
        scene_path = os.path.join(args.data_path, scene)
        os.makedirs(scene_path, exist_ok=True)
        
        # Copy images to output/input directory
        input_dir = os.path.join(scene_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        logging.info(f"Copying images to {input_dir}")
        for img_file in image_files:
            src = os.path.join(source_images_dir, img_file)
            dst = os.path.join(input_dir, img_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        logging.info(f"Processing scene: {scene}")
        logging.info(f"Reading NVM from: {nvm_path}")
        
        # Read NVM cameras
        nvm_cameras = read_nvm_file(nvm_path)
        logging.info(f"Read {len(nvm_cameras)} cameras from NVM file")
        
        # Create sparse directory
        sparse_dir = os.path.join(scene_path, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        
        # Convert NVM to COLMAP format
        cameras_colmap, images_colmap = create_colmap_model_from_nvm(
            nvm_cameras, input_dir, sparse_dir)
        
        # Create database
        db_path = os.path.join(scene_path, "database.db")
        create_database(db_path, cameras_colmap, images_colmap)
        
        logging.info("Created COLMAP model with fixed poses")
        
        # Run image undistortion only (no SfM)
        img_undist_cmd = (
            f"{colmap_command} image_undistorter "
            f"--image_path {input_dir} "
            f"--input_path {sparse_dir} "
            f"--output_path {scene_path} "
            f"--output_type COLMAP"
        )
        
        logging.info("Running image undistortion...")
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Image undistortion failed with code {exit_code}")
            continue
        
        processed_count += 1
        logging.info(f"Successfully processed {scene}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Conversion complete! Processed {processed_count}/{len(scenes)} scenes")
    logging.info(f"{'='*60}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = ArgumentParser(description="Convert DeepBlending dataset to COLMAP format")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to processed dataset (e.g., /path/to/deepblending_pgsr)')
    parser.add_argument('--source_data_path', type=str, required=True,
                        help='Path to original DeepBlending dataset with NVM files')
    parser.add_argument("--colmap_executable", default="", type=str,
                        help='Path to COLMAP executable')
    args = parser.parse_args()
    
    convert_deepblending(args)
    print("Done.")
