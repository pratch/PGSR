# copy images from /ist-nas/users/pratchp/projects/datasets/deepblending/<scene>_IBR_Inputs_Outputs/input_camera_poses_as_nvm/images/*
# to /ist-nas/users/pratchp/projects/datasets/deepblend_pgsr/<scene>/input/

import os
import shutil

def copy_deepblend_images(scene_name):
    src_dir = f"/projects/datasets/deepblending/{scene_name}_IBR_Inputs_Outputs/input_camera_poses_as_nvm/images"
    dst_dir = f"/projects/datasets/deepblending_pgsr/{scene_name}/input"
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        shutil.copy2(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")


if __name__ == "__main__":
    data_path="/projects/datasets/deepblending"
    zip_names = [f[:-4] for f in os.listdir(data_path) if f.endswith('.zip')]
    scenes = [name.replace('_IBR_Inputs_Outputs', '') for name in zip_names]

    print("Scenes to process:", scenes)
    for scene in scenes:
        copy_deepblend_images(scene)   
         