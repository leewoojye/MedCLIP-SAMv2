import torch
import torch.nn as nn

from .loss import HardNegativeLoss


class HardNegativeLossWithEntropy(nn.Module):
    """Decoupled hard negative loss with optional spatial entropy regularization."""

    def __init__(
        self,
        temperature: float = 1.0,
        beta1: float = 1.0,
        beta2: float = 1.0,
        alpha: float = 0.0,
        batch_size: int = 1,
        entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_loss = HardNegativeLoss(
            temperature=temperature,
            beta1=beta1,
            beta2=beta2,
            alpha=alpha,
            batch_size=batch_size,
        )
        self.entropy_weight = entropy_weight

    def forward(
        self,
        image_features,
        text_features,
        logit_scale=None,
        logit_bias=None,
        feature_maps=None,
        output_dict: bool = False,
    ):
        outputs = self.base_loss(
            image_features,
            text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            feature_maps=feature_maps,
            output_dict=True,
        )
        base_loss = outputs["contrastive_loss"]

        entropy_loss = base_loss.new_zeros(())
        if self.entropy_weight > 0:
            if feature_maps is None:
                entropy_loss = base_loss.new_zeros(())
            elif feature_maps.dim() == 3:
                patch_features = feature_maps[:, 1:, :]
                activation = torch.norm(patch_features, p=2, dim=-1)
                eps = 1e-8
                probs = activation / (activation.sum(dim=-1, keepdim=True) + eps)
                entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()
                entropy_loss = self.entropy_weight * entropy

        total_loss = base_loss + entropy_loss

        if output_dict:
            outputs["feature_entropy_loss"] = entropy_loss
            outputs["loss"] = total_loss
            return outputs

        return total_loss
