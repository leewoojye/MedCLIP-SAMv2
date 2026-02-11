import torch
import torch.nn as nn
from open_clip.model import CustomTextCLIP
from open_clip import create_model_and_transforms

def load_and_convert(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    # Load raw state dict from the checkpoint file
    # Ensure it handles both raw state dict and dict-with-metadata (epoch, state_dict, etc.)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        print("Extracting 'state_dict' from checkpoint...")
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean keys: remove "model." or "module." prefixes if they exist
    # The error showed "visual.trunk..." which implies keys are already correct relative to OpenCLIP model
    # BUT the target model is CustomTextCLIP which expects "visual.trunk..." and "text.transformer..."
    # The Missing Keys listed are exactly the keys of the model we are trying to load INTO.
    # This means the state_dict we are loading DOES NOT have these keys, or they are named differently.
    
    print(f"Total keys in state_dict: {len(state_dict)}")
    print("Sample keys from state_dict:")
    for k in list(state_dict.keys())[:5]:
        print(f"  {k}")
        
    # The error message says: Missing key(s) in state_dict: "logit_scale", "visual.trunk.cls_token", ...
    # And Unexpected keys: "state_dict", "optimizer", ...
    # Wait, "Unexpected key(s) in state_dict: 'state_dict'" ???
    # This means `state_dict` variable PASSED to `load_state_dict` still contains the wrapper keys!
    # It implies the variable `state_dict` was NOT correctly unwrapped.
    # IN FACT, `state_dict` variable in `convert.py` line 135 `openclip_model.load_state_dict(state_dict)` 
    # must be the raw dictionary.
    
    # If the file loaded was `saliency_maps/model/epoch_10.pt`, it likely contains `{"epoch": ..., "state_dict": {...}}`
    # The `convert.py` code did:
    # state_dict = torch.load(...)
    # for key in list(state_dict.keys()):
    #    if(key.startswith("model.")): ...
    
    # BUT it didn't extract `state_dict["state_dict"]` if it exists!
    # The unexpected key "state_dict" confirms this.
    
    pass

if __name__ == "__main__":
    load_and_convert("saliency_maps/model/epoch_10.pt")
