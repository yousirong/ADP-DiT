"""
Training monitor utilities for detecting anomalies and preventing training failures.
"""
import torch
import numpy as np
from collections import deque


class LossMonitor:
    """Monitor training losses and detect anomalies."""

    def __init__(self, window_size=50, explosion_threshold=10.0, warmup_steps=0):
        """
        Args:
            window_size: Number of recent losses to track
            explosion_threshold: Multiplier for detecting loss explosion
            warmup_steps: Number of warmup steps to skip explosion detection
        """
        self.window_size = window_size
        self.explosion_threshold = explosion_threshold
        self.warmup_steps = warmup_steps

        self.total_losses = deque(maxlen=window_size)
        self.mse_losses = deque(maxlen=window_size)
        self.ce_losses = deque(maxlen=window_size)
        self.alphas = deque(maxlen=window_size)

        # Track total number of updates
        self.step_count = 0

    def update(self, total_loss, mse_loss, ce_loss, alpha):
        """Update with new loss values."""
        self.total_losses.append(total_loss)
        self.mse_losses.append(mse_loss)
        self.ce_losses.append(ce_loss)
        self.alphas.append(alpha)
        self.step_count += 1

    def check_nan_inf(self, total_loss, mse_loss, ce_loss, alpha):
        """Check for NaN or Inf values."""
        values = [total_loss, mse_loss, ce_loss, alpha]
        names = ['total_loss', 'mse_loss', 'ce_loss', 'alpha']

        for val, name in zip(values, names):
            if torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                return True, f"{name} is NaN or Inf: {val}"

        return False, None

    def check_explosion(self):
        """Check if loss has exploded compared to recent average."""
        # Skip explosion check during warmup period
        if self.step_count <= self.warmup_steps:
            return False, None

        if len(self.total_losses) < 10:
            return False, None

        recent_avg = np.mean(list(self.total_losses)[:-1])
        current = self.total_losses[-1]

        if current > recent_avg * self.explosion_threshold:
            return True, f"Loss explosion detected: {current:.4f} vs avg {recent_avg:.4f} (step {self.step_count})"

        return False, None

    def check_alpha_extreme(self):
        """Check if alpha has reached extreme values.

        Note: Alpha of 0.95 is the default (95% MSE, 5% CE) to prevent classifier overfitting.
        Only warn if alpha goes beyond reasonable bounds (< 0.02 or > 0.98).
        """
        if len(self.alphas) < 5:
            return False, None

        current_alpha = self.alphas[-1]

        # Updated thresholds: 0.02-0.98 (was 0.05-0.95)
        # 0.95 is our intentional default value
        if current_alpha < 0.02 or current_alpha > 0.98:
            return True, f"Alpha at extreme value: {current_alpha:.4f}"

        return False, None

    def check_all(self, total_loss, mse_loss, ce_loss, alpha):
        """
        Perform all checks and return status.

        Returns:
            (is_critical, is_warning, message)
        """
        # Check NaN/Inf (critical)
        has_nan, msg = self.check_nan_inf(total_loss, mse_loss, ce_loss, alpha)
        if has_nan:
            return True, False, f"CRITICAL: {msg}"

        # Update history
        self.update(total_loss, mse_loss, ce_loss, alpha)

        # Check explosion (critical)
        has_explosion, msg = self.check_explosion()
        if has_explosion:
            return True, False, f"CRITICAL: {msg}"

        # Check alpha extreme (warning)
        has_extreme_alpha, msg = self.check_alpha_extreme()
        if has_extreme_alpha:
            return False, True, f"WARNING: {msg}"

        return False, False, "OK"

    def get_statistics(self):
        """Get current statistics."""
        if len(self.total_losses) == 0:
            return None

        return {
            'total_loss': {
                'current': self.total_losses[-1],
                'mean': np.mean(self.total_losses),
                'std': np.std(self.total_losses),
                'min': np.min(self.total_losses),
                'max': np.max(self.total_losses),
            },
            'mse_loss': {
                'current': self.mse_losses[-1],
                'mean': np.mean(self.mse_losses),
            },
            'ce_loss': {
                'current': self.ce_losses[-1],
                'mean': np.mean(self.ce_losses),
            },
            'alpha': {
                'current': self.alphas[-1],
                'mean': np.mean(self.alphas),
                'std': np.std(self.alphas),
            }
        }


def format_statistics(stats):
    """Format statistics for logging."""
    if stats is None:
        return "No statistics available"

    lines = []
    lines.append("Training Statistics (recent window):")
    lines.append(f"  Total Loss: {stats['total_loss']['current']:.4f} "
                 f"(avg: {stats['total_loss']['mean']:.4f} ± {stats['total_loss']['std']:.4f})")
    lines.append(f"  MSE Loss:   {stats['mse_loss']['current']:.4f} "
                 f"(avg: {stats['mse_loss']['mean']:.4f})")
    lines.append(f"  CE Loss:    {stats['ce_loss']['current']:.4f} "
                 f"(avg: {stats['ce_loss']['mean']:.4f})")
    lines.append(f"  Alpha:      {stats['alpha']['current']:.4f} "
                 f"(avg: {stats['alpha']['mean']:.4f} ± {stats['alpha']['std']:.4f})")

    return "\n".join(lines)
