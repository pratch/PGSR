import os
    
# scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
# factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
# data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
# data_base_path='mip360'



data_base_path='/projects/datasets/scannet_6scenes_pgsr'
out_base_path='output_scannet'
gpu_id=0

scenes = [d for d in os.listdir(data_base_path) if os.path.isdir(os.path.join(data_base_path, d))]
# factors = ['4'] * len(scenes)
# data_devices = ['cuda'] * len(scenes)
print("Scenes to process:", scenes)
# exit()

for id, scene in enumerate(scenes):

    if scene != "1366d5ae89":
        continue

    print(f"Processing scene: {scene}")

    common_args = f"--use_depth_filter --max_depth 10.0 --voxel_size 0.01" # --skip_train
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene} {common_args}' # --use_depth_filter --num_cluster 1 --max_depth 10.0 --voxel_size 0.01
    print(cmd)
    os.system(cmd)

    

    # common_args = f"--max_depth 10.0 --voxel_size 0.01"
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene} {common_args}' 
    # print(cmd)
    # os.system(cmd)

