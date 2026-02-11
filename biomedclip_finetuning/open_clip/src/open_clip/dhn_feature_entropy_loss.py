import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegativeFeatureEntropyLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation with Feature Map Entropy Regularization
    Combines DHN-NCE with Feature Map Entropy Minimization to encourage sparsity in spatial features.
    """
    def __init__(self, temperature=1.0, beta1=1.0, beta2=1.0, alpha=0.0, batch_size=1, entropy_weight=0.0):
        super(HardNegativeFeatureEntropyLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.entropy_weight = entropy_weight
        self.batch_size = batch_size

    def forward(self, image_features, text_features, logit_scale=None, logit_bias=None, feature_maps=None, output_dict=False):
        # image_features: [batch_size, dim] - Pooled/CLS features for contrastive loss
        # text_features: [batch_size, dim]
        # feature_maps: [batch_size, tokens, dim] - Spatial features from the last layer (before pooling)
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # --- Base DHN-NCE Logic (Same as before) ---
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        batch_size = logits_per_image.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(image_features.device)
        neg_mask = ~mask
        N = batch_size - 1
        
        # Image-to-Text Loss
        p_img = torch.exp(logits_per_image * mask).diag()
        sim_neg_img = logits_per_image
        sim_neg_img_exp = torch.exp(sim_neg_img) * neg_mask
        
        sim_neg_img_exp_beta = torch.exp(self.beta1 * sim_neg_img) * neg_mask
        Z_beta_img = sim_neg_img_exp_beta.sum(dim=1, keepdim=True)
        weights_img = (N / (Z_beta_img + 1e-9)) * sim_neg_img_exp_beta
        neg_sum_img = (weights_img * sim_neg_img_exp).sum(dim=1)
        
        loss_i = -torch.log(p_img / (p_img * self.alpha + neg_sum_img + 1e-9)).mean()
        
        # Text-to-Image Loss
        p_text = torch.exp(logits_per_text * mask).diag()
        sim_neg_text_exp = torch.exp(logits_per_text) * neg_mask
        
        sim_neg_text_exp_beta = torch.exp(self.beta2 * logits_per_text) * neg_mask
        Z_beta_text = sim_neg_text_exp_beta.sum(dim=1, keepdim=True)
        weights_text = (N / (Z_beta_text + 1e-9)) * sim_neg_text_exp_beta
        neg_sum_text = (weights_text * sim_neg_text_exp).sum(dim=1)
        
        loss_t = -torch.log(p_text / (p_text * self.alpha + neg_sum_text + 1e-9)).mean()
        
        base_loss = (loss_i + loss_t) / 2
        
        # --- Feature Map Entropy Regularization ---
        entropy_loss = torch.tensor(0.0, device=image_features.device)
        
        if feature_maps is None:
             # DEBUG: Warn if feature_maps is not provided when weight > 0
             if self.entropy_weight > 0:
                  print("DEBUG LOSS: feature_maps is NONE! check model output.")
        
        if self.entropy_weight > 0 and feature_maps is not None:
             # feature_maps: [Batch, Tokens, Dim]
             if feature_maps.dim() != 3:
                  print(f"DEBUG LOSS: feature_maps dim is {feature_maps.dim()}, expected 3. Shape: {feature_maps.shape}")
             
             # We want to calculate entropy of the 'activation magnitude' distribution across spatial tokens.
             # This encourages the model to activate strongly on few tokens (salient regions) and weakly on background.
             
             # 1. Calculate activation magnitude per token
             # x: [B, L, D] -> norm: [B, L]
             # We exclude index 0 if it is CLS token (usually it is). 
             # Assumption: output_tokens=True in model configuration usually returns [CLS, Patch1, Patch2, ...]
             
             # Determine if we need to skip CLS token. 
             # If L = H*W + 1, then index 0 is CLS.
             # Safe bet: use all non-CLS tokens.
             
             if feature_maps.dim() == 3:
                 # Check if CLS token exists (heuristic: usually L is odd like 197 or 257)
                 # Or just assume index 0 is CLS as per standard ViT.
                 patch_features = feature_maps[:, 1:, :] # [B, L-1, D]
                 
                 # L2 norm along dimension D to get "activation strength" of each patch
                 # activation: [B, L-1]
                 activation = torch.norm(patch_features, p=2, dim=-1)
                 
                 # Normalize to probability distribution (spatial softmax)
                 # p(x) = activation / sum(activation)
                 eps = 1e-8
                 p_spatial = activation / (activation.sum(dim=-1, keepdim=True) + eps)
                 
                 # Entropy H(p) = - sum p * log(p)
                 # We want to MINIMIZE this entropy (make distribution peaky/sharp)
                 entropy = -torch.sum(p_spatial * torch.log(p_spatial + eps), dim=-1).mean()
                 
                 entropy_loss = self.entropy_weight * entropy
                 
        total_loss = base_loss + entropy_loss
        
        if output_dict:
             return {"loss": total_loss, "dhn_loss": base_loss, "feature_entropy_loss": entropy_loss}
        
        return total_loss
