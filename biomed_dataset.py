
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class EgoBridgeDataset(Dataset):
    def __init__(self, root_dir, stage="stage1", transform=None, mask_transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., Dataset_BUSI_with_GT)
            stage (str): "stage1" (Pos-Pos) or "stage2" (Pos-Neg)
            transform (callable, optional): Transform to be applied to the image.
            mask_transform (callable, optional): Transform to be applied to the mask.
        """
        self.root_dir = root_dir
        self.stage = stage
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Define classes
        # Positive (Tumor): benign, malignant
        # Negative (Normal): normal
        self.pos_dirs = ['benign', 'malignant']
        self.neg_dirs = ['normal']
        
        self.pos_images = []
        self.neg_images = []
        
        print(f"Scanning dataset in {root_dir}...")
        
        # Load Positive Images
        for subdir in self.pos_dirs:
            dir_path = os.path.join(root_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} not found.")
                continue
                
            for filename in os.listdir(dir_path):
                # Filter for images (not masks)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in filename:
                    img_path = os.path.join(dir_path, filename)
                    # Try to find corresponding mask
                    # Pattern: name.png -> name_mask.png
                    # Case 1: normal suffix
                    name_base = os.path.splitext(filename)[0]
                    mask_name = f"{name_base}_mask.png"
                    mask_path = os.path.join(dir_path, mask_name)
                    
                    if os.path.exists(mask_path):
                         self.pos_images.append({'image': img_path, 'mask': mask_path})
                    else:
                         # Try finding with space handling or other patterns if strict match fails?
                         # For now, skip if mask not found (BUSI usually has masks)
                         pass

        # Load Negative Images
        for subdir in self.neg_dirs:
            dir_path = os.path.join(root_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} not found.")
                continue
            
            for filename in os.listdir(dir_path):
                 if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in filename:
                    img_path = os.path.join(dir_path, filename)
                    # Normal images might have masks (empty or full black), or we might not need them for Stage 2 negs?
                    # Stage 2 Neg is "background".
                    self.neg_images.append(img_path)

        print(f"Found {len(self.pos_images)} positive (tumor) images.")
        print(f"Found {len(self.neg_images)} negative (normal) images.")

        if len(self.pos_images) == 0:
            raise ValueError("No positive images found! Check dataset path.")

        # Default Transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
        if self.mask_transform is None:
            # Masks should be resized but NOT normalized usually (0/1 or class indices)
            # Or resized and converted to tensor
            self.mask_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor() 
            ])

    def __len__(self):
        # We iterate over positive images primarily
        return len(self.pos_images)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')
    
    def _load_mask(self, path):
        return Image.open(path).convert('L') # Grayscale

    def __getitem__(self, idx):
        # Primary sample (Positive)
        item1 = self.pos_images[idx]
        img1 = self._load_image(item1['image'])
        mask1 = self._load_mask(item1['mask'])
        
        img1_tensor = self.transform(img1)
        mask1_tensor = self.mask_transform(mask1)
        # Ensure mask is 0-1 (binary)
        mask1_tensor = (mask1_tensor > 0.5).float()

        if self.stage == 'stage1':
            # Pos-Pos Pair
            # Choose another positive sample randomly
            idx2 = random.randint(0, len(self.pos_images) - 1)
            # Ensure different image? Ideally yes, but same is okay-ish for alignment if augs different.
            # Let's try to get different one if possible
            if len(self.pos_images) > 1:
                while idx2 == idx:
                    idx2 = random.randint(0, len(self.pos_images) - 1)
            
            item2 = self.pos_images[idx2]
            img2 = self._load_image(item2['image'])
            mask2 = self._load_mask(item2['mask'])
            
            img2_tensor = self.transform(img2)
            mask2_tensor = self.mask_transform(mask2)
            mask2_tensor = (mask2_tensor > 0.5).float()
            
            return {
                'image1': img1_tensor,
                'mask1': mask1_tensor,
                'image2': img2_tensor,
                'mask2': mask2_tensor,
                'type': 'stage1'
            }

        elif self.stage == 'stage2':
            # Pos-Neg Pair
            # Choose a negative sample randomly
            if len(self.neg_images) > 0:
                neg_path = random.choice(self.neg_images)
                img_neg = self._load_image(neg_path)
            else:
                # Fallback if no negatives?? Treating another pos as neg is bad.
                # Assuming negatives exist.
                 raise ValueError("No negative images available for Stage 2.")
            
            img_neg_tensor = self.transform(img_neg)
            
            return {
                'image_pos': img1_tensor,
                'mask_pos': mask1_tensor,
                'image_neg': img_neg_tensor,
                'type': 'stage2'
            }
        
        else:
             raise ValueError(f"Unknown stage: {self.stage}")


class PairedBioMedDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, mask_dir=None, transform=None):
        """
        Paired Dataset for Stage 3 (Pos -> Neg Bridge) with Mask Support.
        Args:
            pos_dir (str): Directory containing positive (tumor) images.
            neg_dir (str): Directory containing negative (generated healthy) images.
            mask_dir (str, optional): Directory containing tumor masks.
            transform (callable, optional): Transform to be applied.
        """
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.image_pairs = []
        
        # Scan neg_dir (Generated images) to find available pairs
        print(f"Scanning paired dataset...")
        print(f" Positive (Source): {pos_dir}")
        print(f" Negative (Target): {neg_dir}")
        if mask_dir:
            print(f" Masks: {mask_dir}")
        
        if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
            raise ValueError(f"One of the directories does not exist: {pos_dir}, {neg_dir}")
            
        valid_exts = ('.png', '.jpg', '.jpeg')
        neg_files = sorted([f for f in os.listdir(neg_dir) if f.lower().endswith(valid_exts)])
        
        for filename in neg_files:
            pos_path = os.path.join(pos_dir, filename)
            neg_path = os.path.join(neg_dir, filename)
            
            # Simple check
            if not os.path.exists(pos_path):
                 continue

            item = {'pos': pos_path, 'neg': neg_path, 'mask': None}
            
            # Find Mask if directory provided
            if self.mask_dir:
                # Try exact match first
                mask_path = os.path.join(self.mask_dir, filename)
                if not os.path.exists(mask_path):
                    # Try _mask suffix
                    name, ext = os.path.splitext(filename)
                    mask_path = os.path.join(self.mask_dir, f"{name}_mask{ext}")
                
                if os.path.exists(mask_path):
                    item['mask'] = mask_path
                else:
                    # If mask is missing, skip or warn? 
                    # For now, we will create a dummy mask or skip. 
                    # User said "masks 데이터가 다 있으니까", so let's warn if missing but keep item
                    # (will handle None in getitem)
                    pass

            self.image_pairs.append(item)
                
        print(f"Found {len(self.image_pairs)} paired images.")
        
        # Image Transform
        if self.transform is None:
             self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            
        # Mask Transform (Resize + ToTensor, No Norm)
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        img_pos = Image.open(pair['pos']).convert('RGB')
        img_neg = Image.open(pair['neg']).convert('RGB')
        
        img_pos = self.transform(img_pos)
        img_neg = self.transform(img_neg)
        
        mask_tensor = None
        if pair['mask'] and os.path.exists(pair['mask']):
            mask_img = Image.open(pair['mask']).convert('L')
            mask_tensor = self.mask_transform(mask_img)
            # Threshold to 0/1
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Fallback: All ones (= Tumor everywhere? No, usually Mask=1 means Inpaint Area)
            # If no mask found, maybe treat as full image inpainting?
            # Or empty mask?
            # Let's assume Mask=1 is "Tumor/Region to Change".
            # If we don't have mask, maybe we shouldn't change anything? Or change everything?
            # Safe default: Full image mask (change everything)
            mask_tensor = torch.ones((1, 224, 224))
            
        return {
            'image_pos': img_pos,
            'image_neg': img_neg,
            'mask': mask_tensor, # [1, H, W]
            'type': 'paired'
        }
