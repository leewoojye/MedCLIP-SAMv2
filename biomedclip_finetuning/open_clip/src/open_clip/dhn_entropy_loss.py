import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegativeEntropyLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation with Entropy Regularization
    Combines DHN-NCE (Decoupled Hard Negative NCE) with Attention Entropy Minimization.
    """
    def __init__(self, temperature=1.0, beta1=1.0, beta2=1.0, alpha=0.0, batch_size=1, entropy_weight=0.0):
        super(HardNegativeEntropyLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.entropy_weight = entropy_weight
        self.batch_size = batch_size

    def forward(self, image_features, text_features, logit_scale=None, logit_bias=None, attn_maps=None, output_dict=False):
        # image_features: [batch_size, dim]
        # text_features: [batch_size, dim]

        
        # Normalize features if not already normalized (CLIP typically does, but safety check or re-norm)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute logits. 
        # Note: Standard CLIP uses logit_scale to scale cosine similarities.
        # DHN-NCE paper uses 1/temperature scaling similar to InfoNCE.
        # The existing hnl.py implementation uses just self.temperature division on the product.
        # If logit_scale is provided (from CLIP model), we should probably use it OR stick to fixed temperature logic of HNL.
        # HNL implementation in hnl.py: logits = matmul(...) / self.temperature
        
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        batch_size = logits_per_image.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(image_features.device)
        
        # --- DHN-NCE Logic ---
        
        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image * mask) # exp(sim/t) for positives, 1 for off-diagonal (because mask=0 there and exp(0)=1, BUT wait...)
        # Actually standard implementation:
        # pos should extract only diagonals.
        # logits_per_image * mask zeros out off-diagonals. exp(0) is 1. We want exp(sim/t).
        # But we only care about diagonal values for the positive term numerator.
        # Let's use diagonal extraction for clarity.
        
        # Original code:
        # pos = torch.exp(logits_per_image*mask) 
        # neg_mask = ~mask
        
        # If I look at the original hnl.py:
        # pos = torch.exp(logits_per_image*mask)
        # This results in: Diagonals are exp(sim/t). Off-diagonals are exp(0) = 1.
        # This seems wrong if used directly in sum, but HNL only uses `pos` for the positive term in the loss.
        # loss = -log(pos / (pos*alpha + neg_img))
        # Here `pos` is a matrix?
        # If pos is a matrix, then pos*alpha is a matrix.
        # neg_img is a vector (summed over negatives).
        # This implies line-wise operations.
        # Let's check hnl.py again.
        
        # hnl.py:
        # norm_term_img = torch.sum(torch.exp(logits_per_image*neg_mask),dim=-1)
        # reweight_img = N * (torch.exp(self.beta1*logits_per_image*neg_mask))/norm_term_img
        # neg_img = reweight_img * torch.exp(logits_per_image*neg_mask) 
        # This neg_img is a Matrix (N x N).
        # Actually neg_img calculation:
        # reweight_img is (N x N).
        # torch.exp(...) is (N x N).
        # So neg_img is (N x N).
        # Then loss = -log(pos / ...)
        # pos is (N x N).
        # But we only want the diagonal 'pos' to be the numerator.
        # The original code `pos = torch.exp(logits_per_image*mask)` leaves 1s on off-diagonals.
        # However, `neg_img` has 0s on diagonals (because of neg_mask).
        # So for diagonal element (i, i): 
        # pos[i,i] = exp(sim_ii/t)
        # neg_img[i,i] = 0
        # Denominator: pos[i,i]*alpha + 0.
        # Wait, neg_img is NOT summed yet in original code? 
        # "neg_img = reweight_img * torch.exp(...)"
        # Then loss takes `neg_img`.
        # Usually contrastive loss denominator is sum over negatives.
        # The equation in HNL paper involves sum over negatives.
        # In hnl.py line 47:
        # loss = -torch.log(pos / (pos*self.alpha + neg_img))
        # This looks like it operates element-wise on the NxN matrix?
        # If so, it computes loss for ALL pairs, including negatives?
        # Standard InfoNCE minimizes -log( exp(pos) / sum(exp(pos) + exp(neg)) ).
        # DHN-NCE modifies the negative sampling distribution.
        
        # Let's trust the logic from hnl.py but implement cleaner if possible or just copy it.
        # I will strictly follow hnl.py logic to ensure consistency with what user expects "DHN-NCE" to be in this codebase.
        
        mask = torch.eye(batch_size, dtype=torch.bool).to(image_features.device)
        neg_mask = ~mask
        
        # Recalculate everything to be safe
        pos = torch.exp(logits_per_image * mask) # Diagonals: exp(sim), Off: 1
        # We only really care about the diagonal terms for the final loss mean, 
        # OR the original code sums up everything?
        # "return loss.mean()" implies averaging over N*N elements?
        # If so, it minimizes re-weighted similarity for diagonal and ... something for off-diagonal?
        # Wait, if pos has 1s on off-diagonals.
        # off-diagonal term: -log( 1 / (beta + neg_img_ij) ) ? 
        # This seems suspicious for off-diagonals.
        # Usually we only take the diagonal of the loss matrix (loss for true pairs).
        
        # Let's fix the `pos` to only contain relevant values or mask the loss later.
        # But given I must provide a file that works like hnl.py, I should probably copy it and add entropy.
        
        # Re-implementation of hnl.py logic:
        # Note: In hnl.py, `pos` has 1s on off-diagonals.
        # `neg_img` has values on off-diagonals.
        # If we take `loss.mean()`, we are averaging inclusive of off-diagonals.
        # If `pos` is 1 on off-diagonal, and `neg_img` is large, loss is small?
        # Actually usually we only care about positive pairs (diagonals).
        # Let's assume we should mask the final loss to only consider diagonals? 
        # OR HNL implies all pairs contribute?
        # "The standard InfoNCE loss ... is an average over the batch."
        # Usually batch size N implies N positive pairs.
        # I will modify it to properly select diagonals for the main contrastive part if strictly needed,
        # but to be safe and compatible with their "hnl.py", I will stick to their implementation style but add entropy.
        
        # Wait, looking at hnl.py again:
        # norm_term_img = torch.sum(..., dim=-1)   <-- This is a vector (N,) (or N x 1 if keepdim)
        # reweight_img = N * (...) / norm_term_img <-- Broadcasting vector?
        # If norm_term_img is (N), division works row-wise.
        # neg_img = reweight_img * ...
        
        # The main issue is `pos` having 1s.
        # If I calculate loss matrix `L`, and take `L.mean()`, I'm optimizing off-diagonals to have specific values?
        # Usually `loss = diag(L).mean()`.
        # I will presume `loss.mean()` in hnl.py might be a bug or intentional all-pairs formulation.
        # I will preserve it but add entropy.

        # --- Base DHN-NCE (copied logic) ---
        N = batch_size - 1 # This N seems to be (Batch - 1)
        
        # Image-to-Text
        norm_term_img = torch.sum(torch.exp(logits_per_image * neg_mask), dim=-1, keepdim=True)
        reweight_img = N * (torch.exp(self.beta1 * logits_per_image * neg_mask)) / (norm_term_img + 1e-8)
        neg_img = reweight_img * torch.exp(logits_per_image * neg_mask)
        
        # Text-to-Image
        norm_term_text = torch.sum(torch.exp(logits_per_text * neg_mask), dim=-1, keepdim=True)
        reweight_text = N * (torch.exp(self.beta2 * logits_per_text * neg_mask)) / (norm_term_text + 1e-8)
        neg_text = reweight_text * torch.exp(logits_per_text * neg_mask)

        # In hnl.py: neg_img is used in denominator. 
        # But neg_img is a MATRIX (N x N) because reweight_img is (N x N).
        # And it is NOT summed?
        # "loss = -torch.log(pos / (pos*self.alpha + neg_img))"
        # If neg_img is a matrix, then the denominator is a matrix.
        # The log is element-wise.
        # If this is the case, `loss` is a matrix (N x N).
        # `loss.mean()` averages N^2 values.
        # This implies we are bringing `pos` (diagonal exp) close to `pos*alpha + neg_img` (reweighted neg)?
        # For diagonal: pos is exp(sim), neg_img is 0. -> log(1/alpha). Constant?
        # IF alpha is 0 (default). log(pos / 0) -> Inf?
        # Wait. In hnl.py:
        # neg_img has 0s on diagonal (due to neg_mask).
        # If alpha=0, diagonal becomes -log(pos/0) -> Inf.
        # This suggests hnl.py might be flawed or I am misinterpreting `neg_img`.
        
        # Unless... `neg_img` in the loss formula is supposed to be the SUM of negatives?
        # In InfoNCE, denominator is pos + sum(negatives).
        # If hnl.py line 45 `neg_img = ...` computes reweighted terms.
        # Line 48 use `neg_img` directly.
        # If it is not summed, it is element-wise.
        
        # Let's look at hnl.py line 42:
        # norm_term_img = torch.sum(...)
        # That is the sum of negatives.
        
        # Maybe `neg_img` variable in line 48 IS meant to be the sum?
        # But line 45: `neg_img = reweight_img * ...`  <- Element-wise multiplication of matrices.
        # So `neg_img` is a matrix.
        
        # If the user provided `hnl.py`, maybe it's correct in their context or I should just use it.
        # BUT if alpha=0 and neg_img diagonal is 0, it crashes.
        # Let's check if they provided a working hnl.py.
        # Read file `loss/hnl.py` lines 40-54 again.
        
        # I suspect `neg_img` in line 48 should be `torch.sum(neg_img, dim=-1)`.
        # However, I should probably stick to what I see but safeguard the diagonal if alpha=0.
        # OR, maybe I should use `sum` implicitly?
        
        # To be safe, I will implement what makes sense mathematically for DHN-NCE AND respects the variable names.
        # DHN-NCE: -log ( exp(pos) / (exp(pos) + sum( reweighted_neg )) )
        
        p = torch.exp(logits_per_image * mask).diag() # (B,)
        
        # Reweighting negatives
        # exp(beta * sim) / sum(exp(beta * sim)) * N
        # This is the importance sampling weight.
        
        sim_neg = logits_per_image # (B, B)
        # mask diagonals
        sim_neg_exp = torch.exp(sim_neg) * neg_mask
        
        # q_distribution (beta)
        sim_neg_exp_beta = torch.exp(self.beta1 * sim_neg) * neg_mask
        Z_beta = sim_neg_exp_beta.sum(dim=1, keepdim=True)
        weights = (N / (Z_beta + 1e-9)) * sim_neg_exp_beta
        
        # Weighted sum of negatives
        neg_sum_img = (weights * sim_neg_exp).sum(dim=1)
        
        # Loss i
        # -log ( p / (p + neg_sum_img) ) if alpha=1 (standard) or alpha=0 (decoupled?)
        # hnl.py says "Setting alpha to 0, the loss is equivalent to the decoupled HN-NCE loss"
        # Decoupled usually means removing positive from denominator?
        # DCL: -log( p / sum(neg) ) = -log(p) + log(sum(neg))
        # HNL code: log( p / (p*alpha + neg) )
        # If alpha=0: log( p / neg ) = log(p) - log(neg)
        
        loss_i = -torch.log(p / (p * self.alpha + neg_sum_img + 1e-9)).mean()
        
        # Text side
        p_text = torch.exp(logits_per_text * mask).diag()
        sim_neg_text_exp = torch.exp(logits_per_text) * neg_mask
        sim_neg_text_exp_beta = torch.exp(self.beta2 * logits_per_text) * neg_mask
        Z_beta_text = sim_neg_text_exp_beta.sum(dim=1, keepdim=True)
        weights_text = (N / (Z_beta_text + 1e-9)) * sim_neg_text_exp_beta
        neg_sum_text = (weights_text * sim_neg_text_exp).sum(dim=1)
        
        loss_t = -torch.log(p_text / (p_text * self.alpha + neg_sum_text + 1e-9)).mean()
        
        base_loss = (loss_i + loss_t) / 2
        
        # --- Entropy Regularization ---
        entropy_loss = torch.tensor(0.0, device=image_features.device)
        
        if self.entropy_weight > 0 and attn_maps is not None:
             # attn_maps: Expected [Batch, Heads, Tokens, Tokens] or [Batch, Tokens, Tokens]
             # We want entropy of the cls token attention to patch tokens.
             
             # If using OpenCLIP Transformer, `attn_probs` (from my edit) is likely [N*Heads, L, L] or [N, Heads, L, L] depending on implementation
             # transformer.py uses nn.MultiheadAttention.
             # In `forward`: `attn = attn.view(N, self.num_heads, L, L)`
             # My edit captures `self.attn_probs`.
             # So it is [N, Heads, L, L].
             
             # CLS token is index 0.
             # We want attention FROM CLS TO others. -> attn[:, :, 0, 1:]
             
             # attn_maps might be a list (from multiple layers) or single tensor. 
             # Assuming we passed the last layer's map.
             
             if isinstance(attn_maps, list):
                 attn = attn_maps[-1]
             else:
                 attn = attn_maps
             
             # Check dims
             if attn.dim() == 4: # [B, H, L, L]
                 # Averaging heads
                 attn_cls = attn[:, :, 0, 1:].mean(dim=1) # [B, L-1]
             elif attn.dim() == 3: # [B, L, L] (merged heads?)
                 attn_cls = attn[:, 0, 1:]
             else:
                 attn_cls = None
                 
             if attn_cls is not None:
                 eps = 1e-8
                 p_attn = attn_cls + eps
                 # Normalize again just in case (softmax sums to 1 but averaging might drift/precision)
                 p_attn = p_attn / p_attn.sum(dim=-1, keepdim=True)
                 
                 entropy = -torch.sum(p_attn * torch.log(p_attn), dim=-1).mean()
                 entropy_loss = self.entropy_weight * entropy
                 
        total_loss = base_loss + entropy_loss
        
        if output_dict:
             return {"loss": total_loss, "dhn_loss": base_loss, "entropy_loss": entropy_loss}
        
        return total_loss
