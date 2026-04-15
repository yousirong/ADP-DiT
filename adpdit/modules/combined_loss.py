"""
Combined Loss for ADPDiT: MSE (image reconstruction) + CrossEntropy (classification)
with uncertainty-based automatic weighting.

Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics" (Kendall et al., CVPR 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss for joint training of image generation and classification
    using uncertainty-based weighting.

    Loss = (1/(2*σ₁²)) * MSE + (1/σ₂²) * CE + log(σ₁) + log(σ₂)

    where σ₁, σ₂ are learnable uncertainty parameters (homoscedastic uncertainty).

    - Low uncertainty (σ ↓): Task is easy/confident → Higher weight
    - High uncertainty (σ ↑): Task is hard/noisy → Lower weight

    This allows the model to automatically balance the two tasks based on
    their relative difficulties.

    Args:
        num_classes: Number of classification classes (default: 3 for CN, MCI, AD)
        label_smoothing: Label smoothing factor for CE loss (default: 0.1)
        init_log_var_mse: Initial log variance for MSE task (default: 0.0)
        init_log_var_ce: Initial log variance for CE task (default: 0.0)
        min_ce_loss: Minimum threshold for CE loss to prevent numerical instability (default: 1e-4)
        sigma_min: Minimum allowed sigma value (default: 0.1)
        sigma_max: Maximum allowed sigma value (default: 10.0)
    """
    def __init__(self, num_classes=3, label_smoothing=0.1,
                 init_log_var_mse=0.0, init_log_var_ce=0.0,
                 min_ce_loss=1e-4, sigma_min=0.1, sigma_max=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.min_ce_loss = min_ce_loss
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Learnable uncertainty parameters
        # We learn log(σ²) = s for numerical stability
        # This prevents division by zero and uses exp() which is always positive
        self.log_var_mse = nn.Parameter(torch.tensor(init_log_var_mse, dtype=torch.float32))
        self.log_var_ce = nn.Parameter(torch.tensor(init_log_var_ce, dtype=torch.float32))

        # Classification loss (CrossEntropy with label smoothing)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smoothing)

    def get_sigma_mse(self):
        """Get current sigma (uncertainty) for MSE task with clipping."""
        sigma = torch.exp(0.5 * self.log_var_mse)
        return torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)

    def get_sigma_ce(self):
        """Get current sigma (uncertainty) for CE task with clipping."""
        sigma = torch.exp(0.5 * self.log_var_ce)
        return torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)

    def get_weight_mse(self):
        """Get current weight for MSE task (1 / (2 * σ₁²))."""
        return 1.0 / (2.0 * torch.exp(self.log_var_mse))

    def get_weight_ce(self):
        """Get current weight for CE task (1 / σ₂²)."""
        return 1.0 / torch.exp(self.log_var_ce)

    def forward(self, pred_noise, target_noise, cls_logits, cls_labels):
        """
        Compute combined loss.

        Args:
            pred_noise: Predicted noise/image (B, C, H, W)
            target_noise: Target noise/image (B, C, H, W)
            cls_logits: Classification logits (B, num_classes)
            cls_labels: Classification labels (B,) with values in [0, num_classes-1]
                       0: Cognitive Normal (CN)
                       1: Mild Cognitive Impairment (MCI)
                       2: Alzheimer's Disease (AD)

        Returns:
            Dictionary containing:
                - 'loss': Total combined loss with uncertainty weighting
                - 'mse_loss': MSE loss component (raw value)
                - 'ce_loss': CrossEntropy loss component (raw value)
                - 'mse_weight': Current weight for MSE (1 / (2*σ₁²))
                - 'ce_weight': Current weight for CE (1 / σ₂²)
                - 'mse_sigma': Current uncertainty (σ₁) for MSE
                - 'ce_sigma': Current uncertainty (σ₂) for CE
                - 'regularization': Log-variance regularization term
        """
        # MSE loss for image reconstruction
        mse_loss = F.mse_loss(pred_noise, target_noise, reduction='mean')

        # CrossEntropy loss for classification with minimum threshold
        ce_loss_raw = self.ce_loss(cls_logits, cls_labels)
        ce_loss = torch.clamp(ce_loss_raw, min=self.min_ce_loss)  # Prevent CE from going to 0

        # Uncertainty-based weighting following Kendall et al. 2018
        # Total Loss = (1/(2*σ₁²)) * L_MSE + (1/σ₂²) * L_CE + log(σ₁) + log(σ₂)

        # Get clipped sigmas
        sigma_mse = self.get_sigma_mse()
        sigma_ce = self.get_sigma_ce()

        # Compute precision (inverse variance) from clipped sigmas
        precision_mse = 1.0 / (sigma_mse ** 2)
        precision_ce = 1.0 / (sigma_ce ** 2)

        weighted_mse = 0.5 * precision_mse * mse_loss  # (1/(2*σ₁²)) * L_MSE
        weighted_ce = precision_ce * ce_loss           # (1/σ₂²) * L_CE

        # Regularization terms (prevent σ from going to infinity)
        # Use clipped sigma values for regularization
        reg_mse = torch.log(sigma_mse)
        reg_ce = torch.log(sigma_ce)
        regularization = reg_mse + reg_ce

        # Total loss
        total_loss = weighted_mse + weighted_ce + regularization

        # Get current weights for logging
        weight_mse = self.get_weight_mse()
        weight_ce = self.get_weight_ce()

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'ce_loss': ce_loss,
            'mse_weight': weight_mse,
            'ce_weight': weight_ce,
            'mse_sigma': sigma_mse,
            'ce_sigma': sigma_ce,
            'mse_alpha': weight_mse / (weight_mse + weight_ce),  # Normalized weight for compatibility
            'ce_alpha': weight_ce / (weight_mse + weight_ce),    # Normalized weight for compatibility
            'regularization': regularization
        }

    def get_log_var_mse_value(self):
        """Get current log variance for MSE task."""
        return self.log_var_mse.item()

    def get_log_var_ce_value(self):
        """Get current log variance for CE task."""
        return self.log_var_ce.item()

    def get_mse_weight_value(self):
        """Get current MSE weight as Python float."""
        return self.get_weight_mse().item()

    def get_ce_weight_value(self):
        """Get current CE weight as Python float."""
        return self.get_weight_ce().item()

    def get_mse_sigma_value(self):
        """Get current MSE uncertainty (sigma) as Python float."""
        return self.get_sigma_mse().item()

    def get_ce_sigma_value(self):
        """Get current CE uncertainty (sigma) as Python float."""
        return self.get_sigma_ce().item()
