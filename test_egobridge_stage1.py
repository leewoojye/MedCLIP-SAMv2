import torch
import sys
import os
from PIL import Image

# Add src to path
sys.path.insert(0, '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src')

from open_clip_train.data import EgoBridgeDataset
from open_clip.loss import EgoBridgeLoss
import torchvision.transforms as T


def test_dataset():
    print("Testing Dataset Initialization...")
    class Args:
        train_data = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
        egobridge_mode = 'stage1'
        distributed = False
        batch_size = 2
        workers = 0
    args = Args()
    
    preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    mask_transform = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
    
    dataset = EgoBridgeDataset(root_dir=args.train_data, mode='stage1', transform=preprocess, mask_transform=mask_transform)
    print(f"Dataset Initialized. Length: {len(dataset)}")
    
    item = dataset[0]
    print("Item 0 loaded.")
    return item

def test_loss(item):
    print("Testing Loss Calculation...")
    anchor_img, anchor_mask, pos_img, pos_mask = item
    
    # Simulate Features [B, N, D]
    B = 2
    N = 196 # 14x14
    D = 64 # Reduce dim for speed
    
    feat1 = torch.randn(B, N, D)
    feat2 = torch.randn(B, N, D)
    
    batch_mask1 = torch.stack([anchor_mask, anchor_mask])
    batch_mask2 = torch.stack([pos_mask, pos_mask])
    
    print("Instantiating Loss...")
    loss_fn = EgoBridgeLoss(mode='stage1', sinkhorn_eps=0.1, sinkhorn_max_iter=5, contrastive_lambda=1.0)
    
    print("Forward Pass...")
    loss_out = loss_fn(feat1, feat2, batch_mask1, batch_mask2, output_dict=True)
    
    print("Loss Output Keys:", loss_out.keys())
    print("Sinkhorn Loss Value:", loss_out['sinkhorn_loss'].item())

def test_stage1():
    try:
        item = test_dataset()
        test_loss(item)
        print("Stage 1 Verification Passed!")
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stage1()
