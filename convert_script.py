import sys
import torch
import argparse
import os
import shutil

# pt파일을 zeroshot_script가 돌릴 수 있는 hf_model 폴더(bin 파일 포함)로 변환함
# Add paths for local modules
sys.path.append('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model')
sys.path.append('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src')

import convert
from convert import convert_state_dict
from open_clip import create_model_from_pretrained

def find_best_epoch(log_path):
    import re
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}")
        return None
    
    best_epoch = None
    min_loss = float('inf')
    
    # Looking for lines like: Train Epoch: 6 [ 96/114 (100%)] ... Dpo_loss: 0.54657 (0.56558)
    # capturing epoch and the average loss (value in parentheses)
    epoch_pattern = re.compile(r"Train Epoch:\s+(\d+).*?\(100%\)\].*?Loss:\s+[0-9.]+\s+\(([0-9.]+)\)", re.IGNORECASE)
    
    with open(log_path, 'r') as f:
        for line in f:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1)) + 1
                loss = float(match.group(2))
                if loss < min_loss:
                    min_loss = loss
                    best_epoch = epoch
                    
    return best_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Version name (e.g., biomedclip_dpo_udiat_v13)')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number to convert (ignored if --auto-best is used)')
    parser.add_argument('--auto-best', action='store_true', help='Automatically find the epoch with lowest loss from logs')
    parser.add_argument('--save-dir', type=str, default='hf_model', help='Subdirectory name to save the converted model (e.g., hf_model, best_params)')
    args = parser.parse_args()

    version_log_dir = f'/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/{args.name}'
    
    if args.auto_best:
        log_file = f'/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/train_{args.name}.log'
        best_epoch = find_best_epoch(log_file)
        if best_epoch:
            print(f'Found best epoch: {best_epoch}')
            args.epoch = best_epoch
        else:
            print(f'Could not find best epoch from log. Using provided epoch: {args.epoch}')

    print(f'Starting conversion for {args.name} (epoch {args.epoch}) -> {args.save_dir}...')
    
    openclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    checkpoint_path = os.path.join(version_log_dir, f'checkpoints/epoch_{args.epoch}.pt')
    
    print(f'Loading checkpoint from {checkpoint_path}...')
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    for key in list(checkpoint.keys()):
        if key.startswith('model.'):
            checkpoint[key.replace('model.', '')] = checkpoint.pop(key)
        elif key.startswith('module.'):
            checkpoint[key.replace('module.', '')] = checkpoint.pop(key)

    openclip_model.load_state_dict(checkpoint)
    hf_state_dict = convert_state_dict(openclip_model.state_dict())

    output_dir = os.path.join(version_log_dir, args.save_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Copy necessary HF config files from original BiomedCLIP
    print('Copying config files...')
    shutil.copy('saliency_maps/model/config.json', f'{output_dir}/config.json')
    shutil.copy('saliency_maps/model/configuration_biomed_clip.py', f'{output_dir}/configuration_biomed_clip.py')
    shutil.copy('saliency_maps/model/modeling_biomed_clip.py', f'{output_dir}/modeling_biomed_clip.py')
    # Use shutil.copy only if file exists to avoid 'cannot stat' errors
    proc_file = 'saliency_maps/model/processing_biomed_clip.py'
    if os.path.exists(proc_file):
        shutil.copy(proc_file, f'{output_dir}/processing_biomed_clip.py')

    print(f'Saving pytorch_model.bin to {output_dir}...')
    torch.save(hf_state_dict, f'{output_dir}/pytorch_model.bin')
    print('Success!')

if __name__ == '__main__':
    main()
