import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU details:")
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Allocate a tensor on the GPU to test memory allocation
    tensor = torch.randn(1000, 1000, device='cuda')
    print(f"Tensor size: {tensor.size()}")
    tensor = tensor.to('cuda')
    print(f"Tensor device: {tensor.device}")

    # Print details for each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")

    # Clear the cache and check memory again
    torch.cuda.empty_cache()
    for i in range(num_gpus):
        print(f"After emptying cache - GPU {i}:")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
else:
    print("CUDA is not available. No GPU found.")