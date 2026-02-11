
import sys
import torch
try:
    import torchvision
    print(f"torchvision version: {torchvision.__version__}")
except ImportError:
    print("torchvision not found")

sys.path.insert(0, '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src')
import open_clip
print("open_clip version:", open_clip.__version__)
print("Available models:")
models = open_clip.list_models()
# Filter for biomedclip related
biomed_models = [m for m in models if 'biomed' in m.lower() or 'pubmed' in m.lower()]
print("Biomed/PubMed related models:", biomed_models)
