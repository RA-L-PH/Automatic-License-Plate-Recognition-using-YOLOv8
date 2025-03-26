import torch
import torchvision

# Check if PyTorch and torchvision are properly installed
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA is available with version: {torch.version.cuda}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA is not available")