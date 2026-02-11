import torch
import torch.nn as nn
from torch.nn import functional as F
import math

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class HardNegativeLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation proposed in https://arxiv.org/abs/2301.02280
    beta1: hardness parameter for image features
    beta2: hardness parameter for text features
    alpha: the weighting function of the positive sample loss
    Setting alpha to 0, the loss is equivalent to the decoupled HN-NCE loss (DHN-NCE)
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, temperature=1.0,beta1=1.0, beta2 = 1.0, alpha=0.0, batch_size=1):
        super(HardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.batch_size = batch_size

    def forward(self, image_features, text_features, logit_scale=None, logit_bias=None, feature_maps=None, output_dict=False):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity between image and text features
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        mask = torch.eye(logits_per_image.size(0), dtype=torch.bool)
        mask = mask.to(image_features.device)

        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image*mask)

        # Negative pairs: off-diagonal elements
        N = self.batch_size - 1

        neg_mask = ~mask

        # Calculate reweighting factors
        norm_term_img = torch.sum(torch.exp(logits_per_image*neg_mask),dim=-1)
        reweight_img = N * (torch.exp(self.beta1*logits_per_image*neg_mask))/norm_term_img
        norm_term_text = torch.sum(torch.exp(logits_per_text*neg_mask),dim=-1)
        reweight_text = N * (torch.exp(self.beta2*logits_per_text*neg_mask))/norm_term_text

        neg_img = reweight_img * torch.exp(logits_per_image*neg_mask)
        neg_text = reweight_text * torch.exp(logits_per_text*neg_mask)

        # Calculate loss
        loss = -torch.log(pos / (pos*self.alpha + neg_img)) -torch.log(pos / (pos*self.alpha + neg_text))

        return {"contrastive_loss": loss.mean()} if output_dict else loss.mean()


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P` points,
    outputs an approximation of the OT cost with regularization parameter :math:`\epsilon`
    niter: max. number of Sinkhorn iterations
    """
    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        
        batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).to(x.device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).to(x.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        
        # Sinkhorn iterations
        thresh = 1e-1 # stopping criterion

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        # "$M_{ij} = (-C_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i - y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C


class EgoBridgeLoss(nn.Module):
    def __init__(self, mode='stage1', sinkhorn_eps=0.05, sinkhorn_max_iter=50, contrastive_lambda=1.0):
        super().__init__()
        self.mode = mode
        self.sinkhorn = SinkhornDistance(eps=sinkhorn_eps, max_iter=sinkhorn_max_iter, reduction='mean')
        self.contrastive_lambda = contrastive_lambda
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, img1_features, img2_features, mask1, mask2, output_dict=False):
        # img_features expected shape: [Batch, Grid_Size, Feature_Dim] (Flattened feature map)
        # mask expected shape: [Batch, 1, H, W] or [Batch, Grid_Size] if already flattened and resized
        
        # We need to resize masks to match the Grid_Size of features if they are not already
        B, N_tokens, D = img1_features.shape 
        
        # Check if features include CLS token (usually N_tokens is 197 for 14x14 patches + 1 CLS)
        # We assume square grid. If N_tokens = 197, grid_size = 14. 
        # If N_tokens = 196, grid_size = 14.
        # Simple heuristic: is sqrt(N_tokens) integer? 
        # 197 is not square. 196 is 14^2. 
        # If not square, assume first token is CLS and remove it for spatial masking.
        
        grid_size = int(math.sqrt(N_tokens))
        if grid_size * grid_size != N_tokens:
            # Try removing CLS
            if int(math.sqrt(N_tokens - 1)) ** 2 == N_tokens - 1:
                img1_features = img1_features[:, 1:, :]
                img2_features = img2_features[:, 1:, :]
                N_tokens -= 1
                grid_size = int(math.sqrt(N_tokens))
        
        if mask1.dim() == 4: # [B, 1, H, W]
             mask1 = F.interpolate(mask1, size=(grid_size, grid_size), mode='nearest')
             mask1 = mask1.flatten(2).transpose(1, 2).squeeze(-1) # [B, Grid_Size]
        if mask2 is not None and mask2.dim() == 4:
             mask2 = F.interpolate(mask2, size=(grid_size, grid_size), mode='nearest')
             mask2 = mask2.flatten(2).transpose(1, 2).squeeze(-1) # [B, Grid_Size]

        loss = 0.0
        
        if self.mode == 'stage1':
            # Normalize features for stability (Cosine Similarity-based distance)
            img1_features = F.normalize(img1_features, p=2, dim=-1)
            img2_features = F.normalize(img2_features, p=2, dim=-1)

            # 1. Sinkhorn Loss (Positive-Positive Tumor Alignment)
            # Filter features by mask > 0.5 (foreground)
            # Since Sinkhorn expects fixed size, we might need to pad or sample? 
            # Or we can just zero out background features?
            # Better approach for OT with variable support: 
            # Weighted OT? Or just select top-k tokens?
            # Standard Sinkhorn requires fixed N points.
            # Strategy: Mask the Cost Matrix? 
            # Simpler Strategy: Hard Thresholding + Sampling fixed number of points OR 
            # Zeroing out background features effectively makes them "far" if we handle it right, 
            # but simplest is to perform OT on the entire grid but weight the mass (mu, nu) by the mask.
            
            # Weighted Sinkhorn: mu = mask1 / sum(mask1), nu = mask2 / sum(mask2)
            # This aligns the *probability distributions* defined by the masks.
            
            # Flatten masks to [B, N] and normalize to sum to 1
            mu = mask1 / (mask1.sum(dim=1, keepdim=True) + 1e-6)
            nu = mask2 / (mask2.sum(dim=1, keepdim=True) + 1e-6)
            
            # We need to modify Sinkhorn to accept custom mu and nu
            # For now, let's just pass the features weighted by mask? 
            # No, OT is about moving mass.
            # Let's use the provided Sinkhorn but we need to implement the "weighted" version logic 
            # inside specific logic or modify the class. 
            # Actually, standard Sinkhorn implementation above assumes uniform distribution.
            # Let's stick to the plan: "Align distributions".
            # If we simply multiply features by mask, the background becomes zero vector.
            # Zero vectors will align with zero vectors.
            
            # Let's refine: We want to match the *tumor* features.
            # We can mask the features: f_masked = f * mask.
            # And then run Sinkhorn on the whole grid. 
            # Backgrounds (zeros) will map to Backgrounds (zeros) with cost 0 (good).
            # Tumors will map to Tumors.
            # This works if spatial layout is similar, but we want to be invariant to location.
            # OT cost matrix is pairwise distance. Distance between (Tumor at pos 1) and (Tumor at pos 2).
            # If we include background (zeros), the OT might just map everything lazily.
            
            # Better: use the mask to valid tokens only? Batching is hard due to variable length.
            # Compromise: Use the features masked: f1_m = f1 * mask1, f2_m = f2 * mask2.
            # The Sinkhorn will compute distance between all tokens.
            # Cost between bg (zero) and bg (zero) is 0.
            # Cost between tumor and tumor is |f1-f2|^2.
            # Cost between bg and tumor is |0-f|^2 = |f|^2 (large).
            # So OT will naturally prefer mapping bg->bg and tumor->tumor.
            # And within tumor->tumor, it will try to minimize distance (align features).
            # So simply passing masked features to Sinkhorn is a valid approximation!
            
            f1_masked = img1_features * mask1.unsqueeze(-1)
            f2_masked = img2_features * mask2.unsqueeze(-1)
            
            sinkhorn_loss, _, _ = self.sinkhorn(f1_masked, f2_masked)
            
            # 2. Background Contrastive Loss
            # Goal: Push tumor features away from background features within the same image?
            # Plan said: "Background Contrastive Loss: Masked vs Unmasked within same image"
            # bg_mask = 1 - mask
            # avg_tumor = (f * mask).sum(1) / mask.sum(1)
            # avg_bg = (f * bg_mask).sum(1) / bg_mask.sum(1)
            # maximize distance or minimize cosine similarity between avg_tumor and avg_bg
            
            bg_mask1 = 1.0 - mask1
            avg_tumor1 = (img1_features * mask1.unsqueeze(-1)).sum(1) / (mask1.sum(1, keepdim=True) + 1e-6)
            avg_bg1 = (img1_features * bg_mask1.unsqueeze(-1)).sum(1) / (bg_mask1.sum(1, keepdim=True) + 1e-6)
            
            # We want cosine dist to be 0 (sim to be 1)? NO, we want them separated.
            # Minimize Cosine Similarity -> 0? Or -1? 
            # Usually we want them orthogonal (sim=0) or opposite (sim=-1).
            # Let's say minimize similarity. 
            # Loss = CosineSimilarity(tumor, bg)^2 (pushes towards 0)
            # OR Loss = max(0, CosineSim - margin)
            
            sim1 = self.cosine_sim(avg_tumor1, avg_bg1)
            contrastive_loss = (sim1 ** 2).mean() # Push towards orthogonality
            
            # Repeat for img2
            # bg_mask2 = 1.0 - mask2
            # avg_tumor2 = (img2_features * mask2.unsqueeze(-1)).sum(1) / (mask2.sum(1, keepdim=True) + 1e-6)
            # avg_bg2 = (img2_features * bg_mask2.unsqueeze(-1)).sum(1) / (bg_mask2.sum(1, keepdim=True) + 1e-6)
            # sim2 = self.cosine_sim(avg_tumor2, avg_bg2)
            # contrastive_loss += (sim2 ** 2).mean()
            # contrastive_loss /= 2.0
            
            loss = sinkhorn_loss + self.contrastive_lambda * contrastive_loss
            
            return {"loss": loss, "sinkhorn_loss": sinkhorn_loss, "bg_contrastive_loss": contrastive_loss} if output_dict else loss

        elif self.mode == 'stage2':
            # Positive (Tumor) vs Negative (Normal)
            # img1 = Positive, mask1 = Tumor Mask
            # img2 = Negative (Normal), mask2 = None (Normal image is all "background" relative to tumor?)
            # Goal: Tumor features should be far from Normal features.
            
            avg_tumor = (img1_features * mask1.unsqueeze(-1)).sum(1) / (mask1.sum(1, keepdim=True) + 1e-6)
            avg_normal = img2_features.mean(1) # Average of all tokens in normal image
            
            # Minimize Similarity? OR Maximize Distance?
            # Contrastive: Sim(pos, neg) should be low.
            sim = self.cosine_sim(avg_tumor, avg_normal)
            loss = (sim ** 2).mean() # Push towards orthogonality
            
            return {"loss": loss, "stage2_contrastive_loss": loss} if output_dict else loss
            
        else:
            return torch.tensor(0.0, device=img1_features.device, requires_grad=True)
