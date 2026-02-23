import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def main():
    device = "cpu"
    base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    
    print(f"Loading Model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    if not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()

    phrases = [
        "healthy brain",
        "healthy breast",
        "tumor brain",
        "tumor breast"
    ]

    print(f"Extracting text embeddings for:\n  - {phrases[0]}\n  - {phrases[1]}\n  - {phrases[2]}\n  - {phrases[3]}\n")
    inputs = tokenizer(phrases, padding=True, return_tensors="pt")
    
    # Patch for vocab size mismatch that causes CUDA device-side asserts
    vocab_size = model.config.text_config.vocab_size
    if inputs["input_ids"].max() >= vocab_size:
        print(f"Warning: Found tokens exceeding model vocab size {vocab_size}. Clamping.")
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
        
    # Explicitly set token_type_ids to 0 to bypass the model's corrupted internal buffer
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    # Explicitly set position_ids to bypass another corrupted internal buffer in BiomedCLIP checkpoint
    seq_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # 텍스트 임베딩 정규화 (선택적이지만 코사인 유사도를 위해 방향 벡터로 맞춤)
        # CLIP 논문에서는 normalize한 벡터를 사용합니다.
        text_features = F.normalize(text_features, p=2, dim=-1)

    v_hb = text_features[0]  # healthy brain
    v_hbr = text_features[1] # healthy breast
    v_tb = text_features[2]  # tumor brain
    v_tbr = text_features[3] # tumor breast

    # 식 1: healthy brain - healthy breast vs tumor brain - tumor breast
    diff1_A = v_hb - v_hbr
    diff1_B = v_tb - v_tbr
    sim1 = F.cosine_similarity(diff1_A.unsqueeze(0), diff1_B.unsqueeze(0)).item()

    # 식 2: tumor brain - healthy brain vs tumor breast - healthy breast
    diff2_A = v_tb - v_hb
    diff2_B = v_tbr - v_hbr
    sim2 = F.cosine_similarity(diff2_A.unsqueeze(0), diff2_B.unsqueeze(0)).item()

    print("="*65)
    print("                  🧬 BioMedCLIP 벡터 유추 실험 🧬")
    print("="*65)
    
    print("\n[실험 1] 장기(Organ)별 의미적 차이 벡터")
    print("  Vector A: V('healthy brain') - V('healthy breast')")
    print("  Vector B: V('tumor brain')   - V('tumor breast')")
    print(f"  ➡️  Cosine Similarity: {sim1:.4f}")
    
    print("\n[실험 2] 질병 상태(Pathological Shift)별 의미적 차이 벡터")
    print("  Vector A: V('tumor brain')   - V('healthy brain')")
    print("  Vector B: V('tumor breast')  - V('healthy breast')")
    print(f"  ➡️  Cosine Similarity: {sim2:.4f}")
    print("\n" + "="*65)

if __name__ == "__main__":
    main()
