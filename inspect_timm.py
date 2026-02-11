import timm
import inspect
from timm.models.vision_transformer import Attention

print(f"Timm version: {timm.__version__}")
print("Source of Attention.forward:")
print(inspect.getsource(Attention.forward))
