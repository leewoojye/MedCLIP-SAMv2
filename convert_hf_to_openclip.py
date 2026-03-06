"""
HuggingFace BiomedCLIP → OpenCLIP 형식 역변환 스크립트
bin -> pt 파일 변환 용도

Usage:
    python convert_hf_to_openclip.py \
        --input saliency_maps/model/pytorch_model.bin \
        --output saliency_maps/model/openclip_model.pt
"""
import os
import re
import torch
import argparse

# HF → OpenCLIP 역매핑 패턴
REVERSE_PATTERNS = [
    # Vision
    (r"visual_projection\.(.*)", "visual.head.proj.{0}"),
    (r"vision_model\.post_layernorm\.(.*)", "visual.trunk.norm.{0}"),
    (r"vision_model\.embeddings\.patch_embedding\.(.*)", "visual.trunk.patch_embed.proj.{0}"),
    (r"vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.(.*)", "visual.trunk.blocks.{0}.norm2.{1}"),
    (r"vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.(.*)", "visual.trunk.blocks.{0}.norm1.{1}"),
    (r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(.*)", "visual.trunk.blocks.{0}.attn.proj.{1}"),
    (r"vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(.*)", "visual.trunk.blocks.{0}.mlp.fc1.{1}"),
    (r"vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(.*)", "visual.trunk.blocks.{0}.mlp.fc2.{1}"),
    # Text
    (r"text_model\.embeddings\.token_type_embedding\.(.*)", "text.transformer.embeddings.token_type_embeddings.{0}"),
    (r"text_model\.embeddings\.token_embedding\.(.*)", "text.transformer.embeddings.word_embeddings.{0}"),
    (r"text_model\.embeddings\.position_embedding\.(.*)", "text.transformer.embeddings.position_embeddings.{0}"),
    (r"text_model\.embeddings\.layer_norm\.(.*)", "text.transformer.embeddings.LayerNorm.{0}"),
    (r"text_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(.*)", "text.transformer.encoder.layer.{0}.attention.self.key.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(.*)", "text.transformer.encoder.layer.{0}.attention.self.query.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(.*)", "text.transformer.encoder.layer.{0}.attention.self.value.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(.*)", "text.transformer.encoder.layer.{0}.attention.output.dense.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.layer_norm1\.(.*)", "text.transformer.encoder.layer.{0}.attention.output.LayerNorm.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(.*)", "text.transformer.encoder.layer.{0}.intermediate.dense.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.layer_norm2\.(.*)", "text.transformer.encoder.layer.{0}.output.LayerNorm.{1}"),
    (r"text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(.*)", "text.transformer.encoder.layer.{0}.output.dense.{1}"),
    (r"text_projection\.fc1\.(.*)", "text.proj.0.{0}"),
    (r"text_projection\.fc2\.(.*)", "text.proj.2.{0}"),
]


def reverse_convert_state_dict(hf_state_dict):
    """HuggingFace state_dict → OpenCLIP state_dict"""
    new_state_dict = {}

    # QKV를 합치기 위한 임시 저장소
    qkv_parts = {}  # layer_idx -> {"q": tensor, "k": tensor, "v": tensor, "type": "weight"|"bias"}

    for k, v in hf_state_dict.items():
        found = False

        # Vision QKV: HF는 분리, OpenCLIP은 합산
        m = re.match(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)", k)
        if m:
            layer_idx, proj, param_type = m.groups()
            key = (layer_idx, param_type)
            if key not in qkv_parts:
                qkv_parts[key] = {}
            proj_short = proj[0]  # q_proj -> q, k_proj -> k, v_proj -> v
            qkv_parts[key][proj_short] = v
            found = True

        # cls_token: unsqueeze
        elif k == "vision_model.embeddings.class_embedding":
            new_state_dict["visual.trunk.cls_token"] = v.unsqueeze(0).unsqueeze(0)
            found = True

        # pos_embed: unsqueeze
        elif k == "vision_model.embeddings.position_embedding.weight":
            new_state_dict["visual.trunk.pos_embed"] = v.unsqueeze(0)
            found = True

        else:
            for pattern, replacement in REVERSE_PATTERNS:
                m2 = re.match(pattern, k)
                if m2:
                    new_k = replacement.format(*m2.groups())
                    new_state_dict[new_k] = v
                    found = True
                    break

        if not found:
            new_state_dict[k] = v

    # QKV 합치기
    for (layer_idx, param_type), parts in qkv_parts.items():
        if "q" in parts and "k" in parts and "v" in parts:
            combined = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            new_k = f"visual.trunk.blocks.{layer_idx}.attn.qkv.{param_type}"
            new_state_dict[new_k] = combined

    return new_state_dict


def main():
    parser = argparse.ArgumentParser("HF → OpenCLIP 역변환")
    parser.add_argument("--input", type=str, default="saliency_maps/model/pytorch_model.bin",
                        help="HuggingFace pytorch_model.bin 경로")
    parser.add_argument("--output", type=str, default="saliency_maps/model/openclip_model.pt",
                        help="출력 OpenCLIP .pt 파일 경로")
    parser.add_argument("--wrap-state-dict", action="store_true",
                        help="{'state_dict': ...} 형태로 감싸서 저장 (DPO 학습 호환)")
    args = parser.parse_args()

    print(f"Loading HF weights from: {args.input}")
    hf_state_dict = torch.load(args.input, map_location="cpu")

    # 이미 state_dict로 감싸져 있는 경우 처리
    if isinstance(hf_state_dict, dict) and "state_dict" in hf_state_dict:
        hf_state_dict = hf_state_dict["state_dict"]

    print(f"Converting {len(hf_state_dict)} keys...")
    openclip_state_dict = reverse_convert_state_dict(hf_state_dict)
    print(f"Result: {len(openclip_state_dict)} keys")

    # 검증: OpenCLIP 모델에 로드해보기
    try:
        from open_clip import create_model_from_pretrained
        model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        incompatible = model.load_state_dict(openclip_state_dict, strict=False)
        print(f"Missing keys: {len(incompatible.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
        if incompatible.missing_keys:
            print("  Missing:", incompatible.missing_keys[:5], "...")
        if incompatible.unexpected_keys:
            print("  Unexpected:", incompatible.unexpected_keys[:5], "...")
    except Exception as e:
        print(f"Verification skipped: {e}")

    if args.wrap_state_dict:
        save_obj = {"state_dict": openclip_state_dict}
    else:
        save_obj = openclip_state_dict

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(save_obj, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
