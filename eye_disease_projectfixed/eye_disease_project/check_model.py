import sys
try:
    import torch
    print('Torch version:', torch.__version__)
except Exception as e:
    print('Failed to import torch:', e)

try:
    ckpt = torch.load('models/best.pth', map_location='cpu')
    if isinstance(ckpt, dict):
        print('Loaded checkpoint keys:', list(ckpt.keys()))
        print('classes in ckpt:', ckpt.get('classes'))
    else:
        print('Checkpoint loaded, type:', type(ckpt))
except Exception as e:
    print('Error loading checkpoint:', e)
print('Python version:', sys.version)
print('Executable:', sys.executable)