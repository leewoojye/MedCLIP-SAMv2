print("Start loading BiomedCLIP...")
from transformers import AutoModel, AutoProcessor
print("Imported transformers")
biomed_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
try:
    print(f"Loading {biomed_model_name}...")
    model = AutoModel.from_pretrained(biomed_model_name, trust_remote_code=True)
    print("Loaded model to CPU")
    model.to("cuda")
    print("Moved to CUDA")
except Exception as e:
    print(f"Error loading model: {e}")
print("Done.")
