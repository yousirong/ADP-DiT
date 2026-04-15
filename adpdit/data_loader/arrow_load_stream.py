import pickle
import random
from pathlib import Path
import ast
import numpy as np
import re
import json
import time
from functools import partial
from PIL import Image, ImageFilter, ImageEnhance
import cv2

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from scipy import ndimage
import torchvision.transforms.functional as TF_func

from IndexKits.index_kits import ArrowIndexV2, MultiResolutionBucketIndexV2, MultiIndexV2


class GaussianNoise(object):
    """Apply Gaussian Noise to a tensor image.
    Assumes input is a torch.Tensor with shape (C, H, W) and dtype float.
    """
    def __init__(self, mean=0.0, std=0.03, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(img.size()) * self.std + self.mean
            img = img + noise
        return img


class ElasticDeformation(object):
    """Apply elastic deformation to simulate anatomical variations in MRI scans.
    """
    def __init__(self, alpha=1.0, sigma=0.1, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if isinstance(img, torch.Tensor):
                # Convert to numpy for processing
                was_tensor = True
                img_np = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.numpy()
            else:
                was_tensor = False
                img_np = np.array(img)

            # Handle RGB images
            if len(img_np.shape) == 3:
                h, w, c = img_np.shape
                # Create deformation field based on spatial dimensions only
                dx = ndimage.gaussian_filter((np.random.random((h, w)) * 2 - 1), self.sigma) * self.alpha
                dy = ndimage.gaussian_filter((np.random.random((h, w)) * 2 - 1), self.sigma) * self.alpha

                x, y = np.meshgrid(np.arange(w), np.arange(h))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

                # Apply same deformation to each channel
                deformed = np.zeros_like(img_np)
                for channel in range(c):
                    deformed[:, :, channel] = ndimage.map_coordinates(
                        img_np[:, :, channel], indices, order=1
                    ).reshape((h, w))
            else:
                # Handle grayscale images
                h, w = img_np.shape
                dx = ndimage.gaussian_filter((np.random.random((h, w)) * 2 - 1), self.sigma) * self.alpha
                dy = ndimage.gaussian_filter((np.random.random((h, w)) * 2 - 1), self.sigma) * self.alpha

                x, y = np.meshgrid(np.arange(w), np.arange(h))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

                deformed = ndimage.map_coordinates(img_np, indices, order=1).reshape((h, w))

            if was_tensor:
                if img.dim() == 3:
                    deformed = torch.from_numpy(deformed).permute(2, 0, 1)
                else:
                    deformed = torch.from_numpy(deformed)
                return deformed.float()
            else:
                # Handle both grayscale and RGB cases
                if len(deformed.shape) == 2:
                    return Image.fromarray((deformed * 255).astype(np.uint8), mode='L')
                else:
                    return Image.fromarray((deformed * 255).astype(np.uint8), mode='RGB')
        return img


class RandomGammaCorrection(object):
    """Apply random gamma correction to simulate different MRI scanner characteristics.
    """
    def __init__(self, gamma_range=(0.7, 1.4), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            gamma = random.uniform(*self.gamma_range)
            if isinstance(img, torch.Tensor):
                # Clamp to [0, 1] range before applying gamma
                img_clamped = torch.clamp(img, 0, 1)
                return torch.pow(img_clamped, gamma)
            else:
                # For PIL Image
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_gamma = np.power(img_array, gamma)
                img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
                # Preserve original image mode
                mode = 'RGB' if len(img_gamma.shape) == 3 else 'L'
                return Image.fromarray(img_gamma, mode=mode)
        return img


class RandomBiasField(object):
    """Apply random bias field to simulate MRI intensity inhomogeneity.
    """
    def __init__(self, strength=0.3, p=0.25):
        self.strength = strength
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    c, h, w = img.shape
                    img_work = img.clone()
                else:
                    h, w = img.shape
                    img_work = img.clone()
            else:
                w, h = img.size
                img_array = np.array(img)
                if len(img_array.shape) == 3:  # RGB
                    img_work = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                else:  # Grayscale
                    img_work = torch.from_numpy(img_array).float() / 255.0

            # Create smooth bias field
            x = torch.linspace(-1, 1, w)
            y = torch.linspace(-1, 1, h)
            X, Y = torch.meshgrid(x, y, indexing='xy')

            # Random polynomial coefficients
            a = random.uniform(-self.strength, self.strength)
            b = random.uniform(-self.strength, self.strength)
            c = random.uniform(-self.strength, self.strength)

            bias_field = 1 + a * X + b * Y + c * X * Y
            bias_field = torch.clamp(bias_field, 0.5, 2.0)

            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    # Apply bias field to each channel
                    bias_field = bias_field.unsqueeze(0).expand(img_work.shape[0], -1, -1)
                img_biased = img_work * bias_field
                return torch.clamp(img_biased, 0, 1)
            else:
                if len(np.array(img).shape) == 3:  # RGB
                    bias_field_expanded = bias_field.unsqueeze(0).expand(3, -1, -1)
                    img_biased = img_work * bias_field_expanded
                    img_biased = torch.clamp(img_biased, 0, 1)
                    img_np = (img_biased.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    return Image.fromarray(img_np, mode='RGB')
                else:  # Grayscale
                    img_biased = img_work * bias_field
                    img_biased = torch.clamp(img_biased, 0, 1)
                    img_np = (img_biased.numpy() * 255).astype(np.uint8)
                    return Image.fromarray(img_np, mode='L')
        return img


class RandomDropout(object):
    """Apply random dropout to simulate scan artifacts or missing data.
    """
    def __init__(self, dropout_prob=0.05, block_size=8, p=0.2):
        self.dropout_prob = dropout_prob
        self.block_size = block_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if isinstance(img, torch.Tensor):
                img_work = img.clone()
                if img.dim() == 3:
                    _, h, w = img.shape
                else:
                    h, w = img.shape

                # Random block dropout
                num_blocks_h = h // self.block_size
                num_blocks_w = w // self.block_size

                for i in range(num_blocks_h):
                    for j in range(num_blocks_w):
                        if random.random() < self.dropout_prob:
                            start_h = i * self.block_size
                            end_h = min((i + 1) * self.block_size, h)
                            start_w = j * self.block_size
                            end_w = min((j + 1) * self.block_size, w)

                            if img.dim() == 3:
                                img_work[:, start_h:end_h, start_w:end_w] = 0
                            else:
                                img_work[start_h:end_h, start_w:end_w] = 0

                return img_work
            else:
                img_array = np.array(img)
                h, w = img_array.shape

                num_blocks_h = h // self.block_size
                num_blocks_w = w // self.block_size

                for i in range(num_blocks_h):
                    for j in range(num_blocks_w):
                        if random.random() < self.dropout_prob:
                            start_h = i * self.block_size
                            end_h = min((i + 1) * self.block_size, h)
                            start_w = j * self.block_size
                            end_w = min((j + 1) * self.block_size, w)

                            img_array[start_h:end_h, start_w:end_w] = 0

                return Image.fromarray(img_array, mode='L')
        return img


class RandomSharpening(object):
    """Apply random sharpening/smoothing for different scanner characteristics.
    """
    def __init__(self, factor_range=(0.5, 2.0), p=0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            factor = random.uniform(*self.factor_range)
            if isinstance(img, torch.Tensor):
                # Convert to PIL for processing, then back to tensor
                if img.dim() == 3:
                    img_pil = TF.to_pil_image(img)
                else:
                    img_np = (img.numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='L')

                enhancer = ImageEnhance.Sharpness(img_pil)
                img_enhanced = enhancer.enhance(factor)

                if img.dim() == 3:
                    return TF.to_tensor(img_enhanced)
                else:
                    return torch.from_numpy(np.array(img_enhanced)).float() / 255.0
            else:
                enhancer = ImageEnhance.Sharpness(img)
                return enhancer.enhance(factor)
        return img


class RandomTranslation(object):
    """Apply small random translations to simulate patient movement.
    """
    def __init__(self, max_translation=0.1, p=0.3):
        self.max_translation = max_translation
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    _, h, w = img.shape
                else:
                    h, w = img.shape

                max_dx = int(self.max_translation * w)
                max_dy = int(self.max_translation * h)

                dx = random.randint(-max_dx, max_dx)
                dy = random.randint(-max_dy, max_dy)

                # Use affine transformation
                angle = 0
                scale = 1.0
                shear = 0

                img_transformed = TF.affine(img, angle=angle, translate=[dx, dy],
                                          scale=scale, shear=shear,
                                          interpolation=TF.InterpolationMode.BILINEAR,
                                          fill=0)
                return img_transformed
            else:
                w, h = img.size
                max_dx = int(self.max_translation * w)
                max_dy = int(self.max_translation * h)

                dx = random.randint(-max_dx, max_dx)
                dy = random.randint(-max_dy, max_dy)

                return img.transform(img.size, Image.Transform.AFFINE,
                                   (1, 0, dx, 0, 1, dy), fillcolor=0)
        return img


def _make_prefix_logger(log_fn):
    """
    Returns a wrapper function that prepends the module name as a prefix.
    This function is picklable.
    """
    def _logger(msg):
        log_fn(f"    {Path(__file__).stem} | " + msg)
    return _logger


class TextImageArrowStream(Dataset):
    def __init__(self,
                 args,
                 resolution=512,
                 random_flip=None,
                 enable_CN=False,
                 log_fn=print,
                 index_file=None,
                 multireso=False,
                 batch_size=-1,
                 world_size=1,
                 random_shrink_size_cond=False,
                 merge_src_cond=False,
                 uncond_p=0.0,
                 text_ctx_len=77,
                 tokenizer=None,
                 uncond_p_t5=0.0,
                 text_ctx_len_t5=256,
                 tokenizer_t5=None,
                 # MRI-specific parameters
                 enable_medical_augment=True,
                 ):
        self.args = args
        self.resolution = resolution
        self.log_fn = _make_prefix_logger(log_fn)
        self.random_flip = random_flip
        self.enable_CN = enable_CN
        self.index_file = index_file
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.index_manager = self.load_index()
        self.enable_medical_augment = enable_medical_augment

        # clip params
        self.uncond_p = uncond_p
        self.text_ctx_len = text_ctx_len
        self.tokenizer = tokenizer

        # t5 params
        self.uncond_p_t5 = uncond_p_t5
        self.text_ctx_len_t5 = text_ctx_len_t5
        self.tokenizer_t5 = tokenizer_t5

        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond

        # Build augmentation pipeline for MRI data
        base_transforms = []

        # Medical image specific augmentations (applied before tensor conversion)
        if self.enable_medical_augment:
            # Elastic deformation for anatomical variation simulation
            base_transforms.append(ElasticDeformation(alpha=1.0, sigma=0.1, p=0.3))

            # Random translation for patient movement simulation
            base_transforms.append(RandomTranslation(max_translation=0.05, p=0.3))

        # Geometric transformations
        base_transforms.append(T.RandomApply([T.RandomRotation(degrees=10)], p=0.3))  # Increased angle for MRI

        if not self.multireso:
            base_transforms.append(T.RandomResizedCrop(size=(self.resolution, self.resolution),
                                                     scale=(0.85, 1.0)))  # Slightly more aggressive crop
        else:
            base_transforms.append(T.Resize((self.resolution, self.resolution)))

        # Horizontal flip is useful for brain MRI (symmetric)
        flip_p = 0.3 if self.random_flip else 0.0  # Increased probability for MRI
        base_transforms.append(T.RandomHorizontalFlip(p=flip_p))

        # Pre-tensor medical augmentations
        if self.enable_medical_augment:
            base_transforms.append(RandomSharpening(factor_range=(0.7, 1.5), p=0.3))

        # Convert to tensor
        base_transforms.append(T.ToTensor())

        # Post-tensor medical augmentations
        if self.enable_medical_augment:
            # Bias field simulation (common MRI artifact)
            base_transforms.append(RandomBiasField(strength=0.2, p=0.25))

            # Gamma correction for different scanner characteristics
            base_transforms.append(RandomGammaCorrection(gamma_range=(0.8, 1.3), p=0.3))

            # Dropout for scan artifacts
            base_transforms.append(RandomDropout(dropout_prob=0.03, block_size=6, p=0.2))

        # Standard augmentations (adjusted for medical images)
        base_transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.25))

        # Contrast adjustment (important for MRI)
        base_transforms.append(T.ColorJitter(contrast=(0.7, 1.4)))  # More aggressive contrast for MRI

        # Reduced noise for medical precision
        base_transforms.append(GaussianNoise(mean=0.0, std=0.02, p=0.25))

        # Normalize
        base_transforms.append(T.Normalize([0.5], [0.5]))

        self.image_transforms = T.Compose(base_transforms)

        if self.merge_src_cond:
            self.log_fn("Enable merging src condition: (oriW, oriH) --> ((WH)**0.5, (WH)**0.5)")

        self.log_fn("Enable image_meta_size condition (original_size, target_size, crop_coords)")
        if self.enable_medical_augment:
            self.log_fn("Medical image augmentations enabled: ElasticDeformation, BiasField, GammaCorrection, Dropout, etc.")
        self.log_fn(f"Image_transforms: {self.image_transforms}")

    def load_index(self):
        multireso = self.multireso
        index_file = self.index_file
        batch_size = self.batch_size
        world_size = self.world_size

        if multireso:
            if isinstance(index_file, (list, tuple)):
                if len(index_file) > 1:
                    raise ValueError(f"When enabling multireso, index_file should be a single file, but got {index_file}")
                index_file = index_file[0]
            index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)
            self.log_fn(f"Using MultiResolutionBucketIndexV2: {len(index_manager):,}")
        else:
            if isinstance(index_file, str):
                index_file = [index_file]
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(index_file[0])
                self.log_fn(f"Using ArrowIndexV2: {len(index_manager):,}")
            else:
                index_manager = MultiIndexV2(index_file)
                self.log_fn(f"Using MultiIndexV2: {len(index_manager):,}")

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="input_image"):
        """
        Get raw image from arrow file.
        For text-guided img2img:
            - image_key="input_image": source/condition image
            - image_key="edited_image": target/ground truth image
        """
        try:
            ret = self.index_manager.get_image(index, column=image_key)
            # Convert to grayscale first (for MRI data)
            if ret.mode != 'L':
                ret = ret.convert('L')
            # Convert grayscale to RGB for VAE compatibility (3 channels)
            ret = ret.convert('RGB')
        except Exception as e:
            self.log_fn(f'get_raw_image | Error: {e}, image_key: {image_key}')
            # Create RGB image for VAE compatibility
            ret = Image.new("RGB", (256, 256), (128, 128, 128))  # Gray RGB background
        return ret

    @staticmethod
    def random_crop_image(image, origin_size, target_size):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])
        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        if new_width > target_size[0]:
            x_start = random.randint(0, new_width - target_size[0])
            y_start = 0
        else:
            x_start = 0
            y_start = random.randint(0, new_height - target_size[1])
        image_crop = image.crop((x_start, y_start, x_start + target_size[0], y_start + target_size[1]))
        crops_coords_top_left = (x_start, y_start)
        return image_crop, crops_coords_top_left

    def get_style(self, index):
        "Here we use a default learned embedder layer for future extension."
        style = 0
        return style

    def get_image_with_hwxy(self, index, image_key="input_image"):
        image = self.get_raw_image(index, image_key=image_key)
        origin_size = image.size

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            image, crops_coords_top_left = self.index_manager.resize_and_crop(
                image, target_size, resample=Image.LANCZOS, crop_type='random')
            image_tensor = self.image_transforms(image)
        else:
            target_size = (self.resolution, self.resolution)
            crops_coords_top_left = (0, 0)
            image_tensor = self.image_transforms(image)

        if self.random_shrink_size_cond:
            origin_size = (1024 if origin_size[0] < 1024 else origin_size[0],
                           1024 if origin_size[1] < 1024 else origin_size[1])
        if self.merge_src_cond:
            val = (origin_size[0] * origin_size[1]) ** 0.5
            origin_size = (val, val)

        image_meta_size = tuple(origin_size) + tuple(target_size) + tuple(crops_coords_top_left)
        kwargs = {
            'image_meta_size': image_meta_size,
        }

        style = self.get_style(index)
        kwargs['style'] = style

        return image_tensor, kwargs

    def get_images_for_img2img(self, index):
        """
        Get both input and edited images for text-guided image-to-image training.
        Returns input_image (condition), edited_image (target), and metadata.
        """
        # Get input image (condition)
        input_image = self.get_raw_image(index, image_key="input_image")
        # Get edited image (target/ground truth)
        edited_image = self.get_raw_image(index, image_key="edited_image")

        origin_size = input_image.size

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            input_image, crops_coords_top_left = self.index_manager.resize_and_crop(
                input_image, target_size, resample=Image.LANCZOS, crop_type='random')
            edited_image, _ = self.index_manager.resize_and_crop(
                edited_image, target_size, resample=Image.LANCZOS, crop_type='random')
            input_tensor = self.image_transforms(input_image)
            edited_tensor = self.image_transforms(edited_image)
        else:
            target_size = (self.resolution, self.resolution)
            crops_coords_top_left = (0, 0)
            input_tensor = self.image_transforms(input_image)
            edited_tensor = self.image_transforms(edited_image)

        if self.random_shrink_size_cond:
            origin_size = (1024 if origin_size[0] < 1024 else origin_size[0],
                           1024 if origin_size[1] < 1024 else origin_size[1])
        if self.merge_src_cond:
            val = (origin_size[0] * origin_size[1]) ** 0.5
            origin_size = (val, val)

        image_meta_size = tuple(origin_size) + tuple(target_size) + tuple(crops_coords_top_left)
        kwargs = {
            'image_meta_size': image_meta_size,
        }

        style = self.get_style(index)
        kwargs['style'] = style

        return input_tensor, edited_tensor, kwargs

    def get_text_info_with_encoder(self, description):
        pad_num = 0
        text_inputs = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.text_ctx_len,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        attention_mask = text_inputs.attention_mask[0].bool()
        if pad_num > 0:
            attention_mask[1:pad_num + 1] = False
        return description, text_input_ids, attention_mask

    def fill_t5_token_mask(self, fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat((fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1)
        return fill_tensor

    def get_text_info_with_encoder_t5(self, description_t5):
        text_tokens_and_mask = self.tokenizer_t5(
            description_t5,
            max_length=self.text_ctx_len_t5,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_input_ids_t5 = self.fill_t5_token_mask(text_tokens_and_mask["input_ids"], fill_number=1, setting_length=self.text_ctx_len_t5).long()
        attention_mask_t5 = self.fill_t5_token_mask(text_tokens_and_mask["attention_mask"], fill_number=0, setting_length=self.text_ctx_len_t5).bool()
        return description_t5, text_input_ids_t5, attention_mask_t5

    def get_original_text(self, ind):
        """Get edit_prompt for text-guided image-to-image."""
        text = self.index_manager.get_attribute(ind, 'edit_prompt')
        text = str(text).strip()
        return text

    def get_text(self, ind):
        text = self.get_original_text(ind)
        if text == '':
            text = 'Medical brain MRI scan'  # More appropriate for MRI dataset
        return text

    def __getitem__(self, ind):
        # Get text (edit_prompt)
        if random.random() < self.uncond_p:
            description = ""
        else:
            description = self.get_text(ind)

        # Get text for t5
        if random.random() < self.uncond_p_t5:
            description_t5 = ""
        else:
            description_t5 = self.get_text(ind)

        # Get both input and edited images for text-guided img2img
        input_image_tensor, edited_image_tensor, kwargs = self.get_images_for_img2img(ind)

        # Use encoder to embed tokens online
        text, text_embedding, text_embedding_mask = self.get_text_info_with_encoder(description)
        text_t5, text_embedding_t5, text_embedding_mask_t5 = self.get_text_info_with_encoder_t5(description_t5)

        # Add raw text to kwargs for auxiliary encoder parsing
        # The auxiliary encoder will parse this in the training script
        kwargs['raw_text'] = description if description else description_t5

        # Extract classification label from prompt
        # Label mapping: Cognitive Normal -> 0, Mild Cognitive Impairment -> 1, Alzheimer Disease -> 2
        cls_label = self._extract_cls_label(kwargs['raw_text'])
        kwargs['cls_label'] = cls_label

        return (
            input_image_tensor,          # Condition/source image
            edited_image_tensor,         # Target/ground truth image
            text_embedding.clone().detach(),
            text_embedding_mask.clone().detach(),
            text_embedding_t5.clone().detach(),
            text_embedding_mask_t5.clone().detach(),
            {k: torch.tensor(np.array(v)).clone().detach() if k not in ['raw_text', 'cls_label'] else v
             for k, v in kwargs.items()},
        )

    def _extract_cls_label(self, text):
        """
        Extract classification label from prompt text.

        Returns:
            int: 0 for Cognitive Normal, 1 for Mild Cognitive Impairment, 2 for Alzheimer Disease
        """
        text_lower = text.lower()

        # Check for Alzheimer's Disease (most specific first)
        if 'alzheimer' in text_lower or 'ad' in text_lower:
            return 2
        # Check for Mild Cognitive Impairment
        elif 'mild cognitive impairment' in text_lower or 'mci' in text_lower:
            return 1
        # Check for Cognitive Normal
        elif 'cognitive normal' in text_lower or 'cn' in text_lower or 'normal' in text_lower:
            return 0
        else:
            # Default to CN if no match (conservative default)
            return 0

    def __len__(self):
        return len(self.index_manager)

    # ==============================================
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove log_fn when pickling
        state.pop('log_fn', None)
        return state

    def __setstate__(self, state):
        # Called after unpickling
        self.__dict__.update(state)
        # In child processes, replace log_fn with plain print
        self.log_fn = print