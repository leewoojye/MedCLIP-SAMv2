import os
import torch
import open_clip
import json
import re
from open_clip.hf_model import HFTextEncoder
from transformers import AutoModel, AutoProcessor, AutoTokenizer

STATE_DICT_PATTERNS = [
    # Vision
    (r"visual\.head.proj.(\w+)", "visual_projection.{0}"),
    (r"visual\.trunk\.norm\.(\w+)", "vision_model.post_layernorm.{0}"),
    (r"visual\.trunk.patch_embed\.proj\.(\w+)", "vision_model.embeddings.patch_embedding.{0}"),
    (r"visual\.trunk\.blocks\.(\w+)\.norm2\.(\w+)", "vision_model.encoder.layers.{0}.layer_norm2.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.norm1\.(\w+)", "vision_model.encoder.layers.{0}.layer_norm1.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.attn\.proj\.(\w+)", "vision_model.encoder.layers.{0}.self_attn.out_proj.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc1\.(\w+)", "vision_model.encoder.layers.{0}.mlp.fc1.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc2\.(\w+)", "vision_model.encoder.layers.{0}.mlp.fc2.{1}"),
    # Text
    (r"text\.transformer\.embeddings\.token_type_embeddings.(\w+)", "text_model.embeddings.token_type_embedding.{0}"),
    (r"text\.transformer\.embeddings\.word_embeddings.(\w+)", "text_model.embeddings.token_embedding.{0}"),
    (r"text\.transformer\.embeddings\.position_embeddings.(\w+)", "text_model.embeddings.position_embedding.{0}"),
    (r"text\.transformer\.embeddings\.LayerNorm\.(\w+)", "text_model.embeddings.layer_norm.{0}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.key.(\w+)", "text_model.encoder.layers.{0}.self_attn.k_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.query.(\w+)", "text_model.encoder.layers.{0}.self_attn.q_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.value.(\w+)", "text_model.encoder.layers.{0}.self_attn.v_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.dense.(\w+)", "text_model.encoder.layers.{0}.self_attn.out_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).intermediate.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm2.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc2.{1}"),
    (r"text\.proj\.0\.(\w+)", "text_projection.fc1.{0}"),
    (r"text\.proj\.2\.(\w+)", "text_projection.fc2.{0}"),
]

def convert_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        found = False
        if match := re.match(r"visual\.trunk\.blocks\.(\w+)\.attn\.qkv\.(\w+)", k):
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                new_state_dict[new_k] = proj_v
                found = True
        elif k == "visual.trunk.cls_token":
            new_k = "vision_model.embeddings.class_embedding"
            new_state_dict[new_k] = v.squeeze(0).squeeze(0)
            found = True
        elif k == "visual.trunk.pos_embed":
            new_k = "vision_model.embeddings.position_embedding.weight"
            new_state_dict[new_k] = v.squeeze(0)
            found = True
        else:
            for pattern, replacement in STATE_DICT_PATTERNS:
                if match := re.match(pattern, k):
                    new_k = replacement.format(*match.groups())
                    new_state_dict[new_k] = v
                    found = True
                    break
        if not found:
            new_state_dict[k] = v
    return new_state_dict

if __name__ == "__main__":
    checkpoint_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v16/checkpoints/epoch_10.pt"
    output_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model_dpo_v16.bin"

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Clean prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."): k = k[6:]
        if k.startswith("module."): k = k[7:]
        new_state_dict[k] = v
    state_dict = new_state_dict

    # Create dummy model to get standard state dict structure
    from open_clip import create_model_from_pretrained
    openclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # Load weights into openclip model
    incompatible = openclip_model.load_state_dict(state_dict, strict=False)
    print("MISSING KEYS:", incompatible.missing_keys)
    print("UNEXPECTED KEYS:", incompatible.unexpected_keys)
    
    # Convert to HF format
    hf_state_dict = convert_state_dict(openclip_model.state_dict())
    
    print(f"Saving converted weights to {output_path}")
    torch.save(hf_state_dict, output_path)
    print("Done!")
