# BUS-CoT SD-IPC image-to-prompt negative-sample trial

`generate_sd_ipc_trial.py` is the active runner. It follows arXiv:2305.12716:
the normal reference is passed through OpenAI CLIP ViT-L/14, then mapped to
Stable Diffusion's native CLIP text-hidden space with
`pinv(W_text) @ W_visual`. The Stable Diffusion inpainting model architecture
is unchanged.

`generate_trial.py` and its `outputs_attempt*` directories are earlier
diagnostic attempts that directly injected BioMedCLIP vision states. They are
not SD-IPC outputs and must not be used as generated negative samples.

The local BUS-CoT export has no lesion masks. The trial therefore uses three
visually inspected images with explicit, manually drawn ellipse masks:
`000015@0.png`, `000019@0.png`, and `000020@0.png`. The masks are experimental
approximations, not BUS-CoT ground truth; they are written with a five-pixel
dilation into the output folder.

`run_trial.sh` uses `bioclip2/.venv` only as its unchanged base runtime and
loads additive Diffusers dependencies from `pydeps/`. All model caches are in
this folder’s `.hf_cache/`.

Outputs contain generated images, input/mask/generated three-panel comparisons,
the exact masks, and `trial_manifest.json`.
