import os
    
# scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
# factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
# data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
# data_base_path='mip360'



data_base_path='/projects/datasets/deepblending_pgsr_colmapsucceed'
out_base_path='output_deepblending'
gpu_id=0

scenes = [d for d in os.listdir(data_base_path) if os.path.isdir(os.path.join(data_base_path, d))]
# factors = ['4'] * len(scenes)
# data_devices = ['cuda'] * len(scenes)
print("Scenes to process:", scenes)

for id, scene in enumerate(scenes):

    # force DrJohnson scene
    scene="Bedroom"

    print(f"Processing scene: {scene}")

    cmd = f'rm -rf {out_base_path}/{scene}/*'
    print(cmd)
    os.system(cmd)

    # common_args = f"-r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval"
    common_args = f"--max_abs_split_points 0 --opacity_cull_threshold 0.05" 
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    # common_args = f"--max_depth 10.0 --voxel_size 0.01"
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene} {common_args}' 
    # print(cmd)
    # os.system(cmd)

    break
    
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    # print(cmd)
    # os.system(cmd)
