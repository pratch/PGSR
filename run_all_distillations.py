import subprocess
import time
import os
import sys

configs = [
    {"name": "butterfly", "ply": "splats/butterfly.ply", "radius": 2.28},
    {"name": "cake", "ply": "splats/cake.ply", "radius": 31.24},
    {"name": "caketall", "ply": "splats/caketall.ply", "radius": 3.07},
    {"name": "fox", "ply": "splats/fox.ply", "radius": 13.87},
    {"name": "tablebench", "ply": "splats/tablebench.ply", "radius": 3.23},
]

# Create output directories
os.makedirs("output/logs", exist_ok=True)

for config in configs:
    name = config["name"]
    ply = config["ply"]
    radius = config["radius"]
    output_dir = f"output/{name}_distill_run"
    log_file = f"output/logs/{name}_distill.log"
    
    print(f"\n==========================================")
    print(f"Starting self-distillation for {name}...")
    print(f"Initial PLY: {ply}")
    print(f"Camera Radius: {radius}")
    print(f"Output Directory: {output_dir}")
    print(f"Log Location: {log_file}")
    print(f"==========================================")
    
    # Run sequential command
    cmd = [
        "python", "train_fibo_distill.py",
        "--ply", ply,
        "--n-cams", "1000",
        "--radius", str(radius),
        "--multi_view_max_dis", "15.0",
        "--model_path", output_dir,
        "--only_30k"
    ]
    
    with open(log_file, "w") as out:
        process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
        
        # Monitor progress in stdout
        last_step = -1
        while process.poll() is None:
            time.sleep(10)
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    # Look for step progress in logs
                    for line in reversed(lines):
                        if "[Training progress] step" in line:
                            parts = line.split("step")
                            if len(parts) > 1:
                                step_info = parts[1].split("/")[0].strip()
                                try:
                                    step = int(step_info)
                                    if step != last_step:
                                        print(f"[{name}] Iteration progress: {step}/30000")
                                        last_step = step
                                except ValueError:
                                    pass
                            break
                        elif "[Rendering views]" in line and "Completed" not in line:
                            print(f"[{name}] Dataset Generation: {line.strip()}")
                            break
                        elif "[Generating depth maps]" in line and "Completed" not in line:
                            print(f"[{name}] Final Depth Rendering: {line.strip()}")
                            break
        
        # Check return code
        rc = process.returncode
        if rc == 0:
            print(f"[{name}] Distillation training completed successfully!")
        else:
            print(f"[{name}] Distillation training failed with exit code: {rc}. Check details in {log_file}")
            # Exit to prevent wasting GPU resources on later runs if earlier fails
            sys.exit(rc)
