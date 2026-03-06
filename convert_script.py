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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Version name (e.g., biomedclip_dpo_udiat_v13)')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number to convert')
    args = parser.parse_args()

    print(f'Starting conversion for {args.name} (epoch {args.epoch})...')
    
    openclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    checkpoint_path = f'/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/{args.name}/checkpoints/epoch_{args.epoch}.pt'
    
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

    output_dir = f'/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/{args.name}/hf_model'
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

    print('Saving pytorch_model.bin...')
    torch.save(hf_state_dict, f'{output_dir}/pytorch_model.bin')
    print('Success!')

if __name__ == '__main__':
    main()
