import sys
import torch
import glob

sys.path.append('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model')
sys.path.append('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src')

print('Loading convert module...')
import convert
from convert import convert_state_dict

from open_clip import create_model_from_pretrained
import json

openclip_config = json.load(open('saliency_maps/model/config.json'))
openclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

print('Loading checkpoint...')
checkpoint = torch.load('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v11/checkpoints/epoch_3.pt', map_location='cpu', weights_only=False)

if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

for key in list(checkpoint.keys()):
    if key.startswith('model.'):
        checkpoint[key.replace('model.', '')] = checkpoint.pop(key)
    elif key.startswith('module.'):
        checkpoint[key.replace('module.', '')] = checkpoint.pop(key)

openclip_model.load_state_dict(checkpoint)
hf_state_dict = convert_state_dict(openclip_model.state_dict())

import os
import shutil

output_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v11/hf_model'
os.makedirs(output_dir, exist_ok=True)

# Copy necessary HF config files from original BiomedCLIP
os.system(f'cp saliency_maps/model/config.json {output_dir}/')
os.system(f'cp saliency_maps/model/configuration_biomed_clip.py {output_dir}/')
os.system(f'cp saliency_maps/model/modeling_biomed_clip.py {output_dir}/')
os.system(f'cp saliency_maps/model/processing_biomed_clip.py {output_dir}/')

print('Saving pytorch_model.bin...')
torch.save(hf_state_dict, f'{output_dir}/pytorch_model.bin')
print('Success!')
