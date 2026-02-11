
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornDistance(nn.Module):
    """
    Adapted from http://arxiv.org/abs/1306.0895
    and https://github.com/dfdazac/wassdistance
    """
    def __init__(self, eps=0.05, max_iter=20, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x, y: [batch_size, n_points, dim]
        # The Sinkhorn algorithm takes as input two distributions (x and y)
        # We assume uniform weights for the points if not specified

        batch_size, n_points, dim = x.shape
        device = x.device

        # Cost matrix: 1 - Cosine Similarity
        # Normalize x and y first
        x_norm = F.normalize(x, dim=2)
        y_norm = F.normalize(y, dim=2)
        
        # C = 1 - x_norm @ y_norm^T
        # Shape: [batch_size, n_points, n_points]
        C = 1 - torch.matmul(x_norm, y_norm.transpose(1, 2))

        # Regularized transport
        # K = exp(-C/eps)
        K = torch.exp(-C / self.eps)

        # Sinkhorn iterations
        # u = ones, v = ones
        u = torch.ones(batch_size, n_points, device=device) / n_points
        v = torch.ones(batch_size, n_points, device=device) / n_points
        
        # Target marginals (uniform)
        r = torch.ones(batch_size, n_points, device=device) / n_points
        c = torch.ones(batch_size, n_points, device=device) / n_points

        for _ in range(self.max_iter):
            # u = r / (K @ v)
            # v = c / (K^T @ u)
            
            # K @ v: [batch, n, n] @ [batch, n, 1] -> [batch, n, 1]
            # Squeeze v for matmul
            
            # Simplified iteration for batch processing
            # u = r / (K * v.unsqueeze(1)).sum(2)
            # v = c / (K.transpose(1, 2) * u.unsqueeze(1)).sum(2)
            
            # Correct logic:
            # v (n_points) -> K (n,n) @ v (n,1) -> (n,1)
            K_v = torch.matmul(K, v.unsqueeze(2)).squeeze(2) 
            u = r / (K_v + 1e-9)
            
            K_t_u = torch.matmul(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
            v = c / (K_t_u + 1e-9)

        # Transport plan P = u * K * v
        # P = u.unsqueeze(2) * K * v.unsqueeze(1)
        P = torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))
        
        # Sinkhorn Distance = sum(P * C)
        distance = torch.sum(P * C, dim=(1, 2))

        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance


class EgoBridgeLoss(nn.Module):
    def __init__(self, sinkhorn_eps=0.05, contrastive_lambda=1.0):
        super().__init__()
        self.sinkhorn = SinkhornDistance(eps=sinkhorn_eps, max_iter=20, reduction='mean')
        self.contrastive_lambda = contrastive_lambda
        self.bce = nn.BCELoss()

    def _extract_masked_features(self, features, mask):
        """
        Extract features corresponding to the mask (Foreground).
        features: [batch, patches, dim] (e.g., 50x197x768)
        mask: [batch, 1, 224, 224] -> need to downsample to patch grid
        """
        # Downsample mask to patch grid
        # ViT patch size 16 -> 224/16 = 14x14 = 196 patches + 1 CLS = 197
        batch_size, seq_len, dim = features.shape
        
        # Assuming CLS token is at index 0
        patch_features = features[:, 1:, :] # [batch, 196, dim]
        
        # Resize mask to 14x14
        # mask is [batch, 1, H, W]
        mask_down = F.interpolate(mask, size=(14, 14), mode='nearest') # [batch, 1, 14, 14]
        mask_flat = mask_down.flatten(2).transpose(1, 2) # [batch, 196, 1]
        
        # Weighted average or top-k? 
        # For Sinkhorn, we need sets of points.
        # Problem: variable number of masked patches per image.
        # Sinkhorn implementation usually expects fixed size sets.
        # Approach: Select Top-K salient patches or use Weighted Sinkhorn.
        # Or simpler: Masked Average Pooling for a single vector?
        # NO, EgoBridge paper implies structural alignment, so preserving patches is better.
        # BUT standard Sinkhorn requires N points.
        
        # Simplification for implementation:
        # 1. Masked Average Pooling -> Single Vector
        # 2. Sinkhorn on reduced set?
        
        # If we use strict Sinkhorn on 196 patches, we can weigh them by the mask?
        # My Sinkhorn implementation assumes uniform weights.
        # Let's use Masked Features directly.
        
        # Actually, let's keep it simple for now:
        # Masked Average Feature (Foreground Representation)
        # Background Average Feature (Background Representation)
        
        # wait, standard Sinkhorn is for Sets.
        # If I do average pooling, I just have 1 point. Sinkhorn is overkill (defaults to cosine distance).
        
        # Let's try to filter patches.
        # Use simple masking: F_masked = F * mask
        pass

    def forward_stage1(self, feat1, mask1, feat2, mask2):
        """
        Stage 1: Pos-Pos Alignment
        feat1, feat2: [batch, 197, dim] (Sequence output from ViT)
        mask1, mask2: [batch, 1, 224, 224]
        """
        
        # 1. Prepare Features (remove CLS, shape 196)
        f1_patches = feat1[:, 1:, :] # [B, 196, D]
        f2_patches = feat2[:, 1:, :] # [B, 196, D]
        
        # 2. Downsample masks
        m1_down = F.interpolate(mask1, size=(14, 14), mode='nearest').flatten(2).transpose(1, 2) # [B, 196, 1]
        m2_down = F.interpolate(mask2, size=(14, 14), mode='nearest').flatten(2).transpose(1, 2) # [B, 196, 1]
        
        # 3. Sinkhorn Loss (Foreground Alignment)
        # Filter patches?
        # Since Sinkhorn implementation takes fixed N, let's just weigh the features by mask
        # or treat the whole 196 grid as the set, but "masked out" features should be ignored?
        # A trick: Force masked-out features to be very far or zero?
        # Actually, if we just multiply features by mask, the background becomes zero vectors.
        # Sinkhorn will try to match zeros to zeros (background to background) which is fine!
        # So we align the WHOLE image structure, but background is blacked out.
        
        f1_masked = f1_patches * m1_down
        f2_masked = f2_patches * m2_down
        
        sinkhorn_loss = self.sinkhorn(f1_masked, f2_masked)
        
        # 4. Contrastive Loss (Foreground vs Background)
        # Separate FG and BG
        # FG = Masked Mean
        # BG = Inverse Masked Mean
        
        # Avoid division by zero
        m1_sum = m1_down.sum(1) + 1e-6
        m1_inv_sum = (1-m1_down).sum(1) + 1e-6
        
        fg1 = (f1_patches * m1_down).sum(1) / m1_sum
        bg1 = (f1_patches * (1-m1_down)).sum(1) / m1_inv_sum
        
        # We want FG and BG to be dissimilar -> 1 - CosineSim(FG, BG) should be LARGE.
        # wait, Cosine Dist = 1 - Sim. We want to maximize Distance.
        # So Minimize Similarity.
        # Loss = Sim(FG, BG)^2 or just Sim(FG, BG)?
        # Cosine Similarity is [-1, 1]. We want -1 (opposite) or 0 (orthogonal).
        # Let's minimize Sim^2 or abs(Sim).
        
        sim_fg_bg = F.cosine_similarity(fg1, bg1, dim=1)
        contrastive_loss = (sim_fg_bg).mean() # If we want to minimize similarity. 
        # But if we minimize raw similarity [-1, 1], it tries to make them -1.
        # Orthogonal (0) is usually better for disentanglement? Or just "different"?
        # Let's just minimize similarity.
        
        total_loss = sinkhorn_loss + self.contrastive_lambda * contrastive_loss
        return total_loss

    def forward_stage2(self, pos_feat, pos_mask, neg_feat):
        """
        Stage 2: Pos-Neg Contrast
        pos_feat: [B, 197, D]
        neg_feat: [B, 197, D]
        """
        # CLS tokens
        pos_cls = pos_feat[:, 0, :]
        neg_cls = neg_feat[:, 0, :]
        
        # Or patch-based?
        # EgoBridge Stage 2: Contrast Positive (Tumor) vs Negative (Normal).
        # Treat Negative sample as "Background".
        
        # Let's use pooled features (CLS) for simplicity in Stage 2,
        # or masked average for Positive.
        
        # Get FG of Positive
        # Downsample mask
        m_pos_down = F.interpolate(pos_mask, size=(14, 14), mode='nearest').flatten(2).transpose(1, 2)
        pos_patches = pos_feat[:, 1:, :] 
        pos_fg = (pos_patches * m_pos_down).sum(1) / (m_pos_down.sum(1) + 1e-6)
        
        # Get Negative feat (Average of all patches or CLS)
        neg_rep = neg_feat[:, 0, :] 
        
        # Contrastive Loss: Maximize Distance (Minimize Similarity)
        # We want pos_fg and neg_rep to be different.
        
        sim = F.cosine_similarity(pos_fg, neg_rep, dim=1)
        loss = sim.mean() # Minimize similarity
        
        return loss
