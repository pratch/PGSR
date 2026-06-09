import os

# list all output_* directories

# output_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('output_')]
output_dirs = ['output_deepblending']

# get all subfolder names inside each output_* directory
for output_dir in output_dirs:
    subfolders = [f.name for f in os.scandir(output_dir) if f.is_dir()]
    print(f"Subfolders in {output_dir}: {subfolders}")

# create new dir for mesh and GS
out_all_dir = 'output_all_deepblending'

if not os.path.exists(out_all_dir):
    os.makedirs(out_all_dir)

# copy mesh and GS files from each output_* dir to output_all dir
for output_dir in output_dirs:
    scenes = [f.name for f in os.scandir(output_dir) if f.is_dir()]
    dataset_name = output_dir.replace('output_', '')
    for scene in scenes:
        scene_path = os.path.join(output_dir, scene)
        if dataset_name in ['dtu', 'mip360']:
            scene_path = os.path.join(scene_path, 'test')
        mesh_src = os.path.join(scene_path, 'mesh', 'tsdf_fusion_post.ply')
        gs_src = os.path.join(scene_path, 'point_cloud', 'iteration_30000','point_cloud.ply')

        # replace _ with - in scene name
        scene = scene.replace('_', '-')

        mesh_dst_name = f'mesh_{dataset_name}_{scene}.ply'
        gs_dst_name = f'gs_{dataset_name}_{scene}.ply'

        mesh_dst = os.path.join(out_all_dir, mesh_dst_name)
        gs_dst = os.path.join(out_all_dir, gs_dst_name)

        if os.path.exists(mesh_src):
            os.system(f'cp "{mesh_src}" "{mesh_dst}"')
            print(f'Copied {mesh_src} to {mesh_dst}')
        else:
            print(f'No mesh file found at {mesh_src}')

        if os.path.exists(gs_src):
            os.system(f'cp "{gs_src}" "{gs_dst}"')
            print(f'Copied {gs_src} to {gs_dst}')
        else:
            print(f'No GS file found at {gs_src}')