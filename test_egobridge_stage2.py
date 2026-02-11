
import torch
import sys
import os
from PIL import Image
import torchvision.transforms as T

# Add src to path
sys.path.insert(0, '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src')

from open_clip_train.data import EgoBridgeDataset
from open_clip.loss import EgoBridgeLoss

def test_dataset_stage2():
    print("Testing Stage 2 Dataset Initialization...")
    class Args:
        train_data = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
        egobridge_mode = 'stage2'
        distributed = False
        batch_size = 2
        workers = 0
    args = Args()
    
    preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    mask_transform = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
    
    # Mode='stage2'
    dataset = EgoBridgeDataset(root_dir=args.train_data, mode='stage2', transform=preprocess, mask_transform=mask_transform)
    print(f"Dataset Initialized. Length: {len(dataset)}")
    
    # Check item structure: (anchor_img, anchor_mask, neg_img)
    item = dataset[0]
    anchor_img, anchor_mask, neg_img = item
    print(f"Item 0 loaded.")
    print(f"Anchor Img: {anchor_img.shape}, Mask: {anchor_mask.shape}, Neg Img: {neg_img.shape}")
    return item

def test_loss_stage2(item):
    print("Testing Stage 2 Loss Calculation...")
    anchor_img, anchor_mask, neg_img = item
    
    # Simulate Features [B, N, D]
    B = 2
    N = 196 # 14x14
    D = 64
    
    # In train.py, for stage 2, images are concatenated [anchor; neg] -> [2B, C, H, W]
    # And passed to model.
    # Output features will be [2B, N, D]
    # Then split into feat1 (anchor) and feat2 (neg)
    
    feat1 = torch.randn(B, N, D) # Anchor features
    feat2 = torch.randn(B, N, D) # Neg features
    
    # Loss expects batch masks for anchor
    batch_mask_anchor = torch.stack([anchor_mask, anchor_mask])
    
    print("Instantiating Loss (Stage 2)...")
    loss_fn = EgoBridgeLoss(mode='stage2', contrastive_lambda=1.0)
    
    print("Forward Pass...")
    # mask2 is None for stage2
    loss_out = loss_fn(feat1, feat2, batch_mask_anchor, mask2=None, output_dict=True)
    
    print("Loss Output Keys:", loss_out.keys())
    print("Contrastive Loss Value:", loss_out['loss'].item())

def test_stage2():
    try:
        item = test_dataset_stage2()
        test_loss_stage2(item)
        print("Stage 2 Verification Passed!")
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stage2()
