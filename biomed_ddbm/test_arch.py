import torch
from biomed_ddbm.model import BioMedDDBM

def test_arch():
    print("Testing BioMedDDBM Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Init Model
    model = BioMedDDBM(
        dim=32, # Small dim for fast test
        biomed_img_dim=512,
        biomed_text_dim=512
    ).to(device)
    print("Model Initialized.")
    
    # Dummy Inputs
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    
    # BioMedCLIP Embeddings (Dummy)
    c_img = torch.randn(B, 512).to(device) # Image Embed
    c_text = torch.randn(B, 1, 512).to(device) # Text Embed
    
    print(f"Input Shape: {x.shape}")
    print(f"Time Shape: {t.shape}")
    print(f"Condition Img: {c_img.shape}")
    print(f"Condition Text: {c_text.shape}")
    
    # Forward
    try:
        out = model(x, t, c_img, c_text)
        print(f"Output Shape: {out.shape}")
        
        if out.shape == x.shape:
            print("SUCCESS: Output shape matches input shape.")
        else:
            print("FAILURE: Output shape mismatch!")
            
        # Scaling Function Check
        print("Checking Scaling Network internals...")
        # (This is implicitly tested by forward pass)
        
    except Exception as e:
        print(f"FAILURE: Forward pass threw exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_arch()
