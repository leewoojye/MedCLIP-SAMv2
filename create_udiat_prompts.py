import os
import json

base_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
image_dir = os.path.join(base_dir, "UDIAT/test_images")
output_json = os.path.join(base_dir, "saliency_maps/text_prompts/UDIAT_testing.json")
generic_prompt = "A medical breast mammogram revealing an area of concern suggestive of a breast tumor."

if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} not found.")
    exit(1)

files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
prompt_dict = {f: generic_prompt for f in files}

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, 'w') as f:
    json.dump(prompt_dict, f, indent=2)

print(f"Created {output_json} with {len(prompt_dict)} entries.")
