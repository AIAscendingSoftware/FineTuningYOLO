'''import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")'''

import torch
import GPUtil

def check_multiple_gpus():
    # Check if CUDA is available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPUs available:")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            gpu = GPUtil.getGPUs()[i]
            print(f"Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        return num_gpus
    else:
        print("No GPUs are available, running on CPU.")
        return 0

# Call the function
num_gpus = check_multiple_gpus()
