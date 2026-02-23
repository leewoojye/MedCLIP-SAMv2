import torch
from transformers import AutoTokenizer, AutoModel

device = "cpu"
base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True).to(device)
if not hasattr(model.config.text_config, "is_decoder"):
    model.config.text_config.is_decoder = False
model.eval()

phrases = ["healthy brain"]
inputs = tokenizer(phrases, padding=True, return_tensors="pt")
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

import torch.nn as nn

class MyEmb(nn.Module):
    def __init__(self, name, layer):
        super().__init__()
        self.name = name
        self.layer = layer
    def forward(self, x):
        print(f"{self.name} called with shape {x.shape} min {x.min()} max {x.max()}")
        return self.layer(x)

model.text_model.embeddings.token_type_embedding = MyEmb("Token Type Embedding", model.text_model.embeddings.token_type_embedding)
model.text_model.embeddings.token_embedding = MyEmb("Token Embedding", model.text_model.embeddings.token_embedding)

try:
    with torch.no_grad():
        model.get_text_features(**inputs)
except Exception as e:
    import traceback
    traceback.print_exc()
