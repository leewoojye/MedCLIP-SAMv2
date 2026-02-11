
import transformers
print(f"Transformers version: {transformers.__version__}")

try:
    from transformers import CLIPVisionEmbeddings
    print("CLIPVisionEmbeddings imported successfully from top level.")
except ImportError:
    print("CLIPVisionEmbeddings NOT found at top level.")

try:
    from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
    print("CLIPVisionEmbeddings imported successfully from models.clip.modeling_clip.")
except ImportError:
    print("CLIPVisionEmbeddings NOT found in models.clip.modeling_clip.")
