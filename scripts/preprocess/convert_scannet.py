"""
Convert ScanNet dataset with existing COLMAP reconstruction to PGSR format.
This script reads existing COLMAP data from ScanNet and organizes it for PGSR.
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
from read_write_model import read_model, write_model


def create_database(db_path, cameras, images):
    """Create COLMAP database with existing camera parameters."""
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    
    # Add cameras
    for camera_id, camera in cameras.items():
        # Map COLMAP model names to model IDs
        model_name_to_id = {
            'SIMPLE_PINHOLE': 0,
            'PINHOLE': 1,
            'SIMPLE_RADIAL': 2,
            'RADIAL': 3,
            'OPENCV': 4,
            'OPENCV_FISHEYE': 5,
            'FULL_OPENCV': 6,
            'FOV': 7,
            'SIMPLE_RADIAL_FISHEYE': 8,
            'RADIAL_FISHEYE': 9,
            'THIN_PRISM_FISHEYE': 10
        }
        model_id = model_name_to_id.get(camera.model, 1)  # Default to PINHOLE
        db.add_camera(model_id, camera.width, camera.height, camera.params, camera_id=camera_id)
    
    # Add images
    for image_id, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=image_id)
    
    db.commit()
    db.close()


def convert_scannet(args):
    """Main conversion function for ScanNet dataset."""
    # Find all scene directories in source data path
    all_items = sorted(os.listdir(args.source_data_path))
    scene_dirs = [item for item in all_items if os.path.isdir(os.path.join(args.source_data_path, item))]
    
    # Filter to scenes that have the dslr/colmap directory structure
    scenes = []
    for scene_dir in scene_dirs:
        dslr_path = os.path.join(args.source_data_path, scene_dir, "dslr")
        colmap_path = os.path.join(dslr_path, "colmap")
        if os.path.exists(colmap_path):
            scenes.append(scene_dir)
    
    logging.info(f"Found {len(scenes)} ScanNet scenes in {args.source_data_path}")
    if len(scenes) > 0:
        logging.info(f"Scenes: {', '.join(scenes[:5])}" + (f" ... and {len(scenes)-5} more" if len(scenes) > 5 else ""))
    
    processed_count = 0
    for scene in scenes:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing scene: {scene}")
        logging.info(f"{'='*60}")
        
        # Source paths
        source_scene_dir = os.path.join(args.source_data_path, scene)
        source_dslr_dir = os.path.join(source_scene_dir, "dslr")
        source_colmap_dir = os.path.join(source_dslr_dir, "colmap")
        source_images_dir = os.path.join(source_dslr_dir, "resized_undistorted_images")
        
        # Check if all required directories exist
        if not os.path.exists(source_colmap_dir):
            logging.error(f"COLMAP directory not found: {source_colmap_dir}")
            continue
        
        if not os.path.exists(source_images_dir):
            logging.error(f"Images directory not found: {source_images_dir}")
            continue
        
        # Check if images exist
        image_files = sorted([f for f in os.listdir(source_images_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if not image_files:
            logging.warning(f"No images found in {source_images_dir}, skipping")
            continue
        
        logging.info(f"Found {len(image_files)} images in source")
        
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
        
        # Read existing COLMAP reconstruction
        logging.info(f"Reading COLMAP reconstruction from {source_colmap_dir}")
        try:
            cameras, images, points3D = read_model(source_colmap_dir, ext='.txt')
            logging.info(f"Read {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")
        except Exception as e:
            logging.error(f"Failed to read COLMAP model: {e}")
            continue
        
        # Convert camera models to PINHOLE for PGSR compatibility
        # ScanNet images are already undistorted, so we can safely use PINHOLE model
        for cam_id, camera in cameras.items():
            if camera.model == 'OPENCV_FISHEYE':
                # Extract fx, fy, cx, cy from OPENCV_FISHEYE parameters
                # OPENCV_FISHEYE params: fx, fy, cx, cy, k1, k2, k3, k4
                fx, fy, cx, cy = camera.params[:4]
                # Convert to PINHOLE model (params: fx, fy, cx, cy)
                cameras[cam_id] = camera._replace(
                    model='PINHOLE',
                    params=np.array([fx, fy, cx, cy])
                )
                logging.info(f"Converted camera {cam_id} from OPENCV_FISHEYE to PINHOLE")
        
        # Create sparse directory and write COLMAP model
        sparse_dir = os.path.join(scene_path, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)
        
        logging.info(f"Writing COLMAP model to {sparse_dir}")
        write_model(cameras, images, points3D, sparse_dir, ext='.bin')
        write_model(cameras, images, points3D, sparse_dir, ext='.txt')
        
        # Create database
        db_path = os.path.join(scene_path, "database.db")
        create_database(db_path, cameras, images)
        
        logging.info("Created COLMAP database")
        
        # Since ScanNet provides pre-undistorted images, we don't need to run image undistortion
        # Copy images to images directory (physical copy, not symlink)
        images_output_dir = os.path.join(scene_path, "images")
        if os.path.exists(images_output_dir):
            shutil.rmtree(images_output_dir)
        shutil.copytree(input_dir, images_output_dir)
        logging.info(f"Copied images to {images_output_dir}")
        
        # Create sparse/0 symlink in main directory for compatibility
        sparse_link = os.path.join(scene_path, "sparse")
        if not os.path.exists(os.path.join(sparse_link, "0")):
            os.makedirs(sparse_link, exist_ok=True)
        
        processed_count += 1
        logging.info(f"Successfully processed {scene}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Conversion complete! Processed {processed_count}/{len(scenes)} scenes")
    logging.info(f"{'='*60}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = ArgumentParser(description="Convert ScanNet dataset to PGSR format")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to output processed dataset (e.g., /path/to/scannet_pgsr)')
    parser.add_argument('--source_data_path', type=str, required=True,
                        help='Path to original ScanNet dataset with COLMAP reconstructions (e.g., /path/to/scannet_6scenes/data)')
    args = parser.parse_args()
    
    convert_scannet(args)
    print("Done.")

