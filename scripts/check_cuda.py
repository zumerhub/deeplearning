# import torch

# # path = r"C:\samcodebase\PFG_GROUND_RGB\yolov8m-seg.pt"

# try:
#     # checkpoint = torch.load(path, map_location='cpu', weights_only=False)
#     print("✓ File is HEALTHY")
#     print(f"  Type : {type(checkpoint)}")
#     print(f"  Keys : {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
# except Exception as e:
#     print(f"✗ File is CORRUPT: {e}")

import torch
print(torch.cuda.is_available() and torch.cuda.device_count() > 0   )



# # cuda installation my system compatibility:
# pip uninstall torch torchvision torchaudio -y
# pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
