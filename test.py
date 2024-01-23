import torch

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# Create a tensor on the GPU
tensor_on_gpu = torch.rand(3, 3).to(device)

# Perform some operations on the GPU
result = tensor_on_gpu + 1

# Transfer the result back to the CPU if needed
result_on_cpu = result.cpu()

print("Original Tensor on GPU:")
print(tensor_on_gpu)

print("Result on GPU:")
print(result)

print("Result on CPU:")
print(result_on_cpu)
