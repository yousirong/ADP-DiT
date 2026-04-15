"""
Image Quality Metrics for ADP-DiT-G Evaluation

Implements PSNR, MSE, SSIM, and FID metrics for evaluating
generated medical images against ground truth.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models


def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]
        img2: Second image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]
        data_range: The data range of the input image (default: 255 for uint8)

    Returns:
        PSNR value in dB. Higher is better.
        Typical values: 20-40 dB (higher means more similar)
    """
    # Ensure images are the same shape
    assert img1.shape == img2.shape, f"Image shapes must match: {img1.shape} vs {img2.shape}"

    # Convert to float if needed
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Auto-detect data range if images are in [0, 1]
    detected_range = data_range
    if img1.max() <= 1.0 and img2.max() <= 1.0:
        detected_range = 1.0

    # Compute PSNR using scikit-image
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=detected_range)

    return float(psnr_value)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.

    Args:
        img1: First image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]
        img2: Second image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]

    Returns:
        MSE value normalized to [0, 1] range. Lower is better.
        For [0, 255] images: returns MSE / 255^2 = MSE / 65025
        For [0, 1] images: returns MSE directly
        (0 = identical images, 1 = completely different)
    """
    # Ensure images are the same shape
    assert img1.shape == img2.shape, f"Image shapes must match: {img1.shape} vs {img2.shape}"

    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Auto-normalize to [0, 1] if images are in [0, 255]
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0

    # Compute MSE (now always in [0, 1] range)
    mse_value = np.mean((img1 - img2) ** 2)

    return float(mse_value)


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                 data_range: float = 255.0,
                 multichannel: bool = None) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]
        img2: Second image as numpy array (H, W, C) or (H, W), values in [0, 255] or [0, 1]
        data_range: The data range of the input image (default: 255)
        multichannel: Whether to treat image as multichannel (auto-detect if None)

    Returns:
        SSIM value in [0, 1]. Higher is better (1 = identical).
        Typical values: 0.7-0.95 for good quality
    """
    # Ensure images are the same shape
    assert img1.shape == img2.shape, f"Image shapes must match: {img1.shape} vs {img2.shape}"

    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Auto-detect data range if images are in [0, 1]
    detected_range = data_range
    if img1.max() <= 1.0 and img2.max() <= 1.0:
        detected_range = 1.0

    # Auto-detect multichannel
    if multichannel is None:
        multichannel = (len(img1.shape) == 3 and img1.shape[2] > 1)

    # Compute SSIM
    if multichannel:
        # For RGB images, use channel_axis parameter
        ssim_value = structural_similarity(
            img1, img2,
            data_range=detected_range,
            channel_axis=2  # Channels are in the last dimension
        )
    else:
        # For grayscale images
        ssim_value = structural_similarity(
            img1, img2,
            data_range=detected_range
        )

    return float(ssim_value)


def compute_fid(real_images: Union[List[np.ndarray], List[Image.Image]],
                fake_images: Union[List[np.ndarray], List[Image.Image]],
                device: str = 'cuda',
                batch_size: int = 50,
                use_radimagenet: bool = True,
                radimagenet_ckpt_path: str = None) -> float:
    """
    Compute Frechet Inception Distance (FID) between real and generated images.

    FID measures the distance between feature distributions of real and generated images.
    Lower FID indicates better quality and diversity.

    Args:
        real_images: List of real/ground truth images
        fake_images: List of generated/fake images
        device: Device to run computation on ('cuda' or 'cpu')
        batch_size: Batch size for processing images
        use_radimagenet: If True, use RadImageNet backbone; if False, use standard InceptionV3
        radimagenet_ckpt_path: Path to RadImageNet checkpoint file (optional, will use local file if available)

    Returns:
        FID score. Lower is better.
        Typical values:
        - < 10: Excellent
        - 10-20: Good
        - 20-50: Acceptable
        - > 50: Poor

    Note:
        Requires at least 50-100 images per set for reliable statistics.
        RadImageNet-based FID is recommended for medical images as it's trained on medical images.
    """
    import os

    # Convert PIL Images to numpy if needed
    def to_numpy(img):
        if isinstance(img, Image.Image):
            return np.array(img)
        return img

    real_images = [to_numpy(img) for img in real_images]
    fake_images = [to_numpy(img) for img in fake_images]

    # Ensure we have enough images
    if len(real_images) < 10 or len(fake_images) < 10:
        raise ValueError(
            f"FID requires at least 10 images per set. "
            f"Got {len(real_images)} real and {len(fake_images)} fake images."
        )

    # Convert images to torch tensors and normalize to [0, 1]
    def preprocess_images(images):
        """Convert numpy arrays to torch tensors"""
        tensors = []
        for img in images:
            # Ensure shape is (H, W, C)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)  # Grayscale to RGB
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0

            # Convert to torch tensor (C, H, W)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensors.append(tensor)

        return torch.stack(tensors)

    real_tensors = preprocess_images(real_images).to(device)
    fake_tensors = preprocess_images(fake_images).to(device)

    if use_radimagenet:
        # Use RadImageNet backbone for medical images
        print("Using RadImageNet backbone for FID computation (medical imaging optimized)")
        model = None

        # Try to load from local checkpoint file first
        if radimagenet_ckpt_path is None:
            # Check default locations
            default_paths = [
                './InceptionV3.pt',
                'InceptionV3.pt'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    radimagenet_ckpt_path = path
                    break

        if radimagenet_ckpt_path and os.path.exists(radimagenet_ckpt_path):
            try:
                print(f"Loading RadImageNet checkpoint from: {radimagenet_ckpt_path}")
                # Load RadImageNet model from checkpoint
                import timm
                model = timm.create_model('radimagenet_resnet50', pretrained=False, features_only=True)

                # Load checkpoint
                checkpoint = torch.load(radimagenet_ckpt_path, map_location='cpu')
                model.load_state_dict(checkpoint)

                model = model.to(device)
                model.eval()
                print("✓ RadImageNet checkpoint loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load RadImageNet checkpoint ({e})")
                model = None

        # If checkpoint loading failed, try to load from timm
        if model is None:
            try:
                import timm
                print("Downloading RadImageNet model from timm...")
                # Load RadImageNet-pretrained ResNet50
                model = timm.create_model('radimagenet_resnet50', pretrained=True, features_only=True)
                model = model.to(device)
                model.eval()
            except ImportError:
                print("Warning: timm not available, falling back to InceptionV3")
                use_radimagenet = False
            except Exception as e:
                print(f"Warning: Could not load RadImageNet model ({e}), falling back to InceptionV3")
                use_radimagenet = False

    if not use_radimagenet:
        # Fall back to standard InceptionV3
        print("Using InceptionV3 backbone for FID computation (standard)")
        try:
            from pytorch_fid.inception import InceptionV3
        except ImportError:
            raise ImportError(
                "pytorch-fid is required for FID computation. "
                "Install with: pip install pytorch-fid"
            )

        # Initialize Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()

    # Extract features
    def get_activations(images, model, batch_size=50, use_radimagenet=True):
        """Extract feature representations from images"""
        model.eval()
        activations = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                if use_radimagenet:
                    # ResNet50 expects 224x224 input
                    batch_resized = torch.nn.functional.interpolate(
                        batch, size=(224, 224), mode='bilinear', align_corners=False
                    )
                    # RadImageNet uses standard ImageNet normalization
                    # Apply ImageNet normalization for better results
                    from torchvision import transforms
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    batch_resized = normalize(batch_resized)

                    # Get features from last layer before classification
                    features = model(batch_resized)
                    # features is a list with one element (features_only=True)
                    feat = features[-1]  # Last layer features

                    # Global average pooling
                    if feat.dim() == 4:  # (B, C, H, W)
                        feat = torch.nn.functional.adaptive_avg_pool2d(feat, output_size=(1, 1))
                        feat = feat.squeeze(-1).squeeze(-1)  # (B, C)

                    activations.append(feat.cpu().numpy())
                else:
                    # InceptionV3 expects 299x299 input
                    batch_resized = torch.nn.functional.interpolate(
                        batch, size=(299, 299), mode='bilinear', align_corners=False
                    )
                    pred = model(batch_resized)[0]

                    # If model output is not scalar, apply global spatial average pooling
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

                    activations.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())

        return np.concatenate(activations, axis=0)

    real_activations = get_activations(real_tensors, model, batch_size, use_radimagenet)
    fake_activations = get_activations(fake_tensors, model, batch_size, use_radimagenet)

    # Compute statistics
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)

    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    # Compute FID score
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return float(fid)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two Gaussian distributions.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
        eps: Small value for numerical stability

    Returns:
        Frechet distance
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Means must have the same shape"
    assert sigma1.shape == sigma2.shape, "Covariances must have the same shape"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array"""
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    # Clip to valid range
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


# =========================================================
# RadImageNet InceptionV3 Feature Extraction & Metrics
# =========================================================


class RadImageNetInceptionV3Backbone(nn.Module):
    """
    RadImageNet InceptionV3 Backbone for feature extraction.

    Loads InceptionV3.pt checkpoint and extracts 2048-dim features
    from the second-to-last layer.
    """
    def __init__(self):
        super().__init__()
        try:
            base = models.inception_v3(weights=None, aux_logits=False, transform_input=False)
        except TypeError:
            base = models.inception_v3(pretrained=False, aux_logits=False, transform_input=False)

        encoder_layers = list(base.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])  # drop final fc layer

    def forward(self, x):
        """Extract 2048-dim features from input"""
        y = self.backbone(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = torch.flatten(y, 1)  # (N, 2048)
        return y


class RadImageNetFeatureExtractor:
    """
    Feature extractor using RadImageNet-pretrained InceptionV3.

    Uses local checkpoint: ./InceptionV3.pt
    """
    def __init__(self, ckpt_path: str = None, device: str = "cuda"):
        """
        Initialize feature extractor.

        Args:
            ckpt_path: Path to InceptionV3.pt checkpoint. If None, uses default path.
            device: Device to run on ('cuda' or 'cpu')
        """
        import os

        self.device = device

        # Use default path if not provided
        if ckpt_path is None:
            default_paths = [
                './InceptionV3.pt',
                'InceptionV3.pt'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    ckpt_path = path
                    print(f"[RadImageNetFeatureExtractor] Found checkpoint at: {ckpt_path}")
                    break

        if ckpt_path is None:
            raise FileNotFoundError(
                "Could not find InceptionV3.pt. "
                "Please provide explicit path or ensure file exists at: "
                "./InceptionV3.pt"
            )

        # Build model and load checkpoint
        self.model = RadImageNetInceptionV3Backbone().to(device).eval()

        print(f"[RadImageNetFeatureExtractor] Loading checkpoint from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")

        # Handle various checkpoint formats
        if isinstance(sd, dict) and ("state_dict" in sd):
            sd = sd["state_dict"]
        if isinstance(sd, dict) and ("model_state_dict" in sd):
            sd = sd["model_state_dict"]

        # Remove 'module.' prefix if present
        if isinstance(sd, dict):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[RadImageNetFeatureExtractor] Loaded checkpoint - "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

        # Define preprocessing pipeline
        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_one(self, pil_im: Image.Image) -> np.ndarray:
        """
        Extract 2048-dim feature vector from a single PIL image.

        Args:
            pil_im: PIL Image

        Returns:
            Feature vector as numpy array (2048,)
        """
        if pil_im.mode != "RGB":
            pil_im = pil_im.convert("RGB")
        x = self.tf(pil_im).unsqueeze(0).to(self.device)  # (1,3,299,299)
        feat = self.model(x)  # (1,2048)
        return feat.detach().cpu().numpy().reshape(-1)    # (2048,)

    @torch.no_grad()
    def extract_batch(self, pil_ims: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple PIL images.

        Args:
            pil_ims: List of PIL Images
            batch_size: Batch size for processing

        Returns:
            Feature matrix as numpy array (N, 2048)
        """
        all_feats = []
        for i in range(0, len(pil_ims), batch_size):
            batch_ims = pil_ims[i:i + batch_size]
            tensors = []

            for pil_im in batch_ims:
                if pil_im.mode != "RGB":
                    pil_im = pil_im.convert("RGB")
                tensors.append(self.tf(pil_im))

            x = torch.stack(tensors).to(self.device)  # (B, 3, 299, 299)
            feats = self.model(x).detach().cpu().numpy()  # (B, 2048)
            all_feats.append(feats)

        return np.concatenate(all_feats, axis=0)


def compute_pair_metrics(pred_pil: Image.Image, gt_pil: Image.Image) -> dict:
    """
    Compute image quality metrics (SSIM, PSNR, MAE, RMSE, MSE) for a prediction-ground truth pair.

    Args:
        pred_pil: Predicted image as PIL Image
        gt_pil: Ground truth image as PIL Image

    Returns:
        Dict with keys: 'ssim', 'psnr', 'mae', 'rmse', 'mse'
    """
    gt = np.array(gt_pil.convert("L")).astype(np.float32) / 255.0
    pr = np.array(pred_pil.convert("L")).astype(np.float32) / 255.0

    if gt.shape != pr.shape:
        pr = np.array(
            Image.fromarray((pr * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    win_size = 7 if min(gt.shape) >= 7 else 3
    ssim = structural_similarity(gt, pr, data_range=1.0, win_size=win_size)
    psnr = peak_signal_noise_ratio(gt, pr, data_range=1.0)

    mae = float(np.mean(np.abs(gt - pr)))
    mse = float(np.mean((gt - pr) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        'ssim': float(ssim),
        'psnr': float(psnr),
        'mae': mae,
        'rmse': rmse,
        'mse': mse
    }


def pfid_single(pred_pil: Image.Image, gt_pil: Image.Image,
                extractor: RadImageNetFeatureExtractor) -> float:
    """
    Compute Per-image FID (P-FID) using RadImageNet features.

    P-FID is the normalized L2 distance between feature vectors in RadImageNet space (0~100).

    Args:
        pred_pil: Predicted image as PIL Image
        gt_pil: Ground truth image as PIL Image
        extractor: RadImageNetFeatureExtractor instance

    Returns:
        P-FID score (0~100), lower is better
    """
    f_gt = extractor.extract_one(gt_pil)
    f_pr = extractor.extract_one(pred_pil)
    l2 = np.linalg.norm(f_gt - f_pr)
    maxd = np.sqrt(f_gt.shape[0])  # sqrt(2048)
    return float(min(l2 / maxd, 1.0) * 100.0)
