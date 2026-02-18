
import torch
import numpy as np

model_path_1 = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model.bin"
model_path_2 = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model_ori.bin"

print(f"Loading model 1: {model_path_1}")
state_dict_1 = torch.load(model_path_1, map_location="cpu")

print(f"Loading model 2: {model_path_2}")
state_dict_2 = torch.load(model_path_2, map_location="cpu")

keys1 = set(state_dict_1.keys())
keys2 = set(state_dict_2.keys())

if keys1 != keys2:
    print("Warning: Model keys are different!")
    diff_keys_1 = keys1 - keys2
    diff_keys_2 = keys2 - keys1
    print(f"Keys in model 1 only: {len(diff_keys_1)}")
    print(f"Keys in model 2 only: {len(diff_keys_2)}")
else:
    print("Model keys match exactly.")

total_diff = 0.0
total_params = 0
changed_layers = 0

print("\n--- Layer-wise Comparison (Top 10 largest differences) ---")

diffs = []

for key in keys1.intersection(keys2):
    t1 = state_dict_1[key].float()
    t2 = state_dict_2[key].float()
    
    if t1.shape != t2.shape:
        print(f"Shape mismatch for {key}: {t1.shape} vs {t2.shape}")
        continue
        
    diff = torch.abs(t1 - t2).mean().item()
    if diff > 0:
        total_diff += torch.abs(t1 - t2).sum().item()
        changed_layers += 1
        diffs.append((key, diff))
    
    total_params += t1.numel()

diffs.sort(key=lambda x: x[1], reverse=True)

for key, diff in diffs[:10]:
    print(f"{key}: Mean Abs Diff = {diff:.6f}")

print("\n--- Summary ---")
print(f"Total parameters: {total_params}")
print(f"Changed layers: {changed_layers} / {len(keys1)}")
if changed_layers == 0:
    print("Models are identical.")
else:
    print("Models are different.")
