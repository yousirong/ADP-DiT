from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from peft.utils import (
    ModulesToSaveWrapper,
    _get_submodules,
)
from timm.models.vision_transformer import Mlp
from torch.utils import checkpoint
from tqdm import tqdm
from transformers.integrations import PeftAdapterMixin

from .attn_layers import Attention, FlashCrossMHAModified, FlashSelfMHAModified, CrossAttention
from .embedders import TimestepEmbedder, PatchEmbed, timestep_embedding
# ,TimestepEmbedderWithMetadata,MetadataEmbedder
from .norm_layers import RMSNorm
from .auxiliary_encoder import AuxiliaryMetadataEncoder
from .poolers import AttentionPool


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class FP32_SiLU(nn.SiLU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)


class ADPDiTBlock(nn.Module):
    """
    A ADPDiT block with `add` conditioning.
    """
    def __init__(self,
                 hidden_size,
                 c_emb_size,
                 num_heads,
                 mlp_ratio=4.0,
                 text_states_dim=1280,
                 use_flash_attn=False,
                 qk_norm=False,
                 norm_type="layer",
                 skip=False,
                 ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        if use_flash_attn:
            self.attn1 = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        else:
            self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        if use_flash_attn:
            self.attn2 = FlashCrossMHAModified(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True,
                                               qk_norm=qk_norm)
        else:
            self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True,
                                        qk_norm=qk_norm)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            # Retain original layers for backward compatibility with existing checkpoints
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)

            # Additional layers to support input image conditioning
            self.skip_norm_with_input = norm_layer(3 * hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear_with_input = nn.Linear(3 * hidden_size, hidden_size)
        else:
            self.skip_norm = None
            self.skip_linear = None
            self.skip_norm_with_input = None
            self.skip_linear_with_input = None

        self.gradient_checkpointing = False

    def _forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None and skip is not None:
            # Check if skip contains concatenated input image (dimension indicates this)
            skip_dim = skip.shape[-1]
            cat = torch.cat([x, skip], dim=-1)

            if skip_dim == 2 * x.shape[-1] and self.skip_linear_with_input is not None:
                # skip contains input image patches - use new layers
                cat = self.skip_norm_with_input(cat)
                x = self.skip_linear_with_input(cat)
            else:
                # normal skip without input image - use original layers
                cat = self.skip_norm(cat)
                x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        attn_inputs = (
            self.norm1(x) + shift_msa, freq_cis_img,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (
            self.norm3(x), text_states, freq_cis_img
        )
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x

    def forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward, x, c, text_states, freq_cis_img, skip, use_reentrant=False)
        return self._forward(x, c, text_states, freq_cis_img, skip)


class FinalLayer(nn.Module):
    """
    The final layer of ADPDiT.
    """
    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels, num_classes=3, cls_dropout=0.1):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

        # Classification head: adaln -> linear projection -> MLP -> logits
        # We use the same c embedding that goes through adaLN_modulation
        # Added dropout to prevent overfitting
        self.num_classes = num_classes
        self.cls_projection = nn.Linear(c_emb_size, final_hidden_size, bias=True)
        self.cls_mlp = nn.Sequential(
            FP32_SiLU(),
            nn.Linear(final_hidden_size, final_hidden_size // 2, bias=True),
            nn.Dropout(p=cls_dropout),  # Dropout after first linear layer
            FP32_SiLU(),
            nn.Linear(final_hidden_size // 2, num_classes, bias=True)
        )

    def forward(self, x, c, return_cls_logits=False):
        # Image generation branch
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_modulated = modulate(self.norm_final(x), shift, scale)
        x_out = self.linear(x_modulated)

        # Classification branch (using the same c embedding)
        if return_cls_logits:
            # Project c to hidden_size, then pass through MLP
            cls_hidden = self.cls_projection(c)  # (B, hidden_size)
            cls_logits = self.cls_mlp(cls_hidden)  # (B, num_classes)
            return x_out, cls_logits

        return x_out


class ADPDiT(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    ADPDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """
    @register_to_config
    def __init__(self,
                 args: Any,
                 input_size: tuple = (32, 32),
                 patch_size: int = 2,
                 in_channels: int = 4,
                 hidden_size: int = 1152,
                 depth: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 log_fn: callable = print,
    ):
        super().__init__()
        self.args = args
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = args.learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if args.learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        self.norm = args.norm

        use_flash_attn = args.infer_mode == 'fa' or args.use_flash_attn
        if use_flash_attn:
            log_fn(f"    Enable Flash Attention.")
        qk_norm = args.qk_norm  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias=True),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=torch.float32))

        # Attention pooling
        pooler_out_dim = 1024
        # Uncomment below when using CLIP model; comment out when using BERT model
        # self.text_states_dim = 768  # Update to match input tensor if fixed
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=pooler_out_dim)

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        if args.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256

        if args.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = nn.Embedding(1, hidden_size)
            self.extra_in_dim += hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_embedder = nn.Sequential(
            nn.Linear(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # Input image embedder (separate from noise embedder)
        self.input_img_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)

        # Image features projection (reserved for future use)
        # This creates a projection layer that matches hidden_size to text_states_dim
        self.image_to_cross_attn_proj = nn.Linear(hidden_size, self.text_states_dim)

        # Auxiliary metadata encoder
        self.use_auxiliary_encoder = getattr(args, 'use_auxiliary_encoder', False)
        if self.use_auxiliary_encoder:
            self.auxiliary_encoder = AuxiliaryMetadataEncoder(
                clinical_hidden_size=256,
                cognitive_hidden_size=512,
                output_size=hidden_size,  # Match model hidden size
                num_cognitive_scores=13
            )
            log_fn(f"    Auxiliary metadata encoder enabled")
            # Update extra_in_dim to include auxiliary embeddings
            self.extra_in_dim += hidden_size

        # ADP Blocks
        self.blocks = nn.ModuleList([
            ADPDiTBlock(hidden_size=hidden_size,
                            c_emb_size=hidden_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            text_states_dim=self.text_states_dim,
                            use_flash_attn=use_flash_attn,
                            qk_norm=qk_norm,
                            norm_type=self.norm,
                            skip=layer > depth // 2,
                            )
            for layer in range(depth)
        ])

        # Classification dropout rate (default 0.3 to prevent overfitting)
        cls_dropout = getattr(args, 'cls_dropout', 0.3)
        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels,
                                       num_classes=3, cls_dropout=cls_dropout)
        self.unpatchify_channels = self.out_channels

        # Uncertainty parameters for multi-task learning (Kendall et al. 2018)
        # These will be learned during training to automatically balance MSE and CE losses
        # Initialize to 0.0 (σ = 1.0) for balanced starting point
        self.log_var_mse = nn.Parameter(torch.tensor(0.0))
        self.log_var_ce = nn.Parameter(torch.tensor(0.0))

        self.initialize_weights()

    def check_condition_validation(self, image_meta_size, style):
        if self.args.size_cond is None and image_meta_size is not None:
            raise ValueError(f"When `size_cond` is None, `image_meta_size` should be None, but got "
                             f"{type(image_meta_size)}. ")
        if self.args.size_cond is not None and image_meta_size is None:
            raise ValueError(f"When `size_cond` is not None, `image_meta_size` should not be None. ")
        if not self.args.use_style_cond and style is not None:
            raise ValueError(f"When `use_style_cond` is False, `style` should be None, but got {type(style)}. ")
        if self.args.use_style_cond and style is None:
            raise ValueError(f"When `use_style_cond` is True, `style` should be not None.")

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = False

    def forward(self,
                x,
                t,
                encoder_hidden_states=None,
                text_embedding_mask=None,
                encoder_hidden_states_t5=None,
                text_embedding_mask_t5=None,
                image_meta_size=None,
                style=None,
                cos_cis_img=None,
                sin_cis_img=None,
                return_dict=True,
                controls=None,
                input_image=None,
                auxiliary_data=None,
                ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            Noisy image tensor (B, D, H, W)
        t: torch.Tensor
            Timestep (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        input_image: torch.Tensor, optional
            Clean input image tensor to be concatenated with skip connections (B, D, H, W)
        auxiliary_data: dict, optional
            Dictionary containing:
                - 'clinical_data': dict of clinical metadata tensors
                - 'cognitive_scores': (B, num_scores) tensor of cognitive test scores
        """
        text_states = encoder_hidden_states  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.bool()  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()  # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat(
            [text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1
        )  # 2,205，1024
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask
        text_states = torch.where(
            clip_t5_mask.unsqueeze(2),
            text_states,
            self.text_embedding_padding.to(text_states),
        )
        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)  # Noisy image patches

        # ========================= Input image embedding =========================
        input_img_patches = None
        if input_image is not None:
            # Match input_image dtype to the model's dtype
            if input_image.dtype != x.dtype:
                input_image = input_image.to(x.dtype)
            input_img_patches = self.input_img_embedder(input_image)  # Clean input image patches

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        if self.args.size_cond == None:
            image_meta_size = None
        self.check_condition_validation(image_meta_size, style)
        # Build image meta size tokens if applicable
        if image_meta_size is not None:
            image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)   # [B * 6, 256]
            if self.args.use_fp16:
                image_meta_size = image_meta_size.half()
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        # Build style tokens
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Build auxiliary metadata embedding
        if self.use_auxiliary_encoder and auxiliary_data is not None:
            clinical_data = auxiliary_data['clinical_data']
            cognitive_scores = auxiliary_data['cognitive_scores']

            # Move to correct device and dtype
            clinical_data = {k: v.to(x.device) for k, v in clinical_data.items()}
            cognitive_scores = cognitive_scores.to(x.device)

            # Get auxiliary embedding
            auxiliary_emb = self.auxiliary_encoder(clinical_data, cognitive_scores)  # [B, hidden_size]
            extra_vec = torch.cat([extra_vec, auxiliary_emb], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through ADPDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                if controls is not None:
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()

                # Concatenate input image patches if available (for skip connection)
                if input_img_patches is not None:
                    # CFG doubles the batch size, so we need to expand input_img_patches to match
                    batch_ratio = skip.shape[0] // input_img_patches.shape[0]
                    if batch_ratio > 1:
                        # Repeat input_img_patches for CFG
                        input_img_patches_expanded = input_img_patches.repeat(batch_ratio, 1, 1)
                    else:
                        input_img_patches_expanded = input_img_patches
                    skip = torch.cat([skip, input_img_patches_expanded], dim=-1)

                x = block(x, c, text_states, freqs_cis_img, skip)   # (N, L, D)
            else:
                x = block(x, c, text_states, freqs_cis_img)         # (N, L, D)

            if layer < (self.depth // 2 - 1):
                skips.append(x)
        if controls is not None and len(controls) != 0:
            raise ValueError("The number of controls is not equal to the number of skip connections.")

        # ========================= Final layer =========================
        # Get classification logits if requested
        if return_dict:
            x, cls_logits = self.final_layer(x, c, return_cls_logits=True)  # (N, L, patch_size ** 2 * out_channels), (N, num_classes)
            x = self.unpatchify(x, th, tw)  # (N, out_channels, H, W)
            return {'x': x, 'cls_logits': cls_logits}
        else:
            x = self.final_layer(x, c, return_cls_logits=False)  # (N, L, patch_size ** 2 * out_channels)
            x = self.unpatchify(x, th, tw)  # (N, out_channels, H, W)
            return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize input image embedder:
        w_input = self.input_img_embedder.proj.weight.data
        nn.init.xavier_uniform_(w_input.view([w_input.shape[0], -1]))
        nn.init.constant_(self.input_img_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.extra_embedder[0].weight, std=0.02)
        nn.init.normal_(self.extra_embedder[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ADPDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.default_modulation[-1].weight, 0)
            nn.init.constant_(block.default_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize classification head:
        nn.init.normal_(self.final_layer.cls_projection.weight, std=0.02)
        if self.final_layer.cls_projection.bias is not None:
            nn.init.constant_(self.final_layer.cls_projection.bias, 0)
        for layer in self.final_layer.cls_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize log variances to 0.0 (σ = 1.0 for balanced starting point)
        nn.init.constant_(self.log_var_mse, 0.0)
        nn.init.constant_(self.log_var_ce, 0.0)

    def get_log_var_mse(self):
        """Get current log variance for MSE task"""
        return self.log_var_mse

    def get_log_var_ce(self):
        """Get current log variance for CE task"""
        return self.log_var_ce

    def get_sigma_mse(self):
        """Get current uncertainty (σ) for MSE task"""
        return torch.exp(0.5 * self.log_var_mse)

    def get_sigma_ce(self):
        """Get current uncertainty (σ) for CE task"""
        return torch.exp(0.5 * self.log_var_ce)

    def get_weight_mse(self):
        """Get current weight for MSE task (1 / (2*σ₁²))"""
        return 1.0 / (2.0 * torch.exp(self.log_var_mse))

    def get_weight_ce(self):
        """Get current weight for CE task (1 / σ₂²)"""
        return 1.0 / torch.exp(self.log_var_ce)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.get_base_layer()
        elif hasattr(child, "quant_linear_module"):
            # TODO maybe not necessary to have special treatment?
            child = child.quant_linear_module

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ranknum" in name:
                module.to(child.weight.device)

    def merge_and_unload(self,
                         merge=True,
                        progressbar: bool = False,
                        safe_merge: bool = False,
                        adapter_names = None,):
        if merge:
            if getattr(self, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge layers when the model is gptq quantized")

        def merge_recursively(module):
            # helper function to recursively merge the base_layer of the target
            path = []
            layer = module
            while hasattr(layer, "base_layer"):
                path.append(layer)
                layer = layer.base_layer
            for layer_before, layer_after in zip(path[:-1], path[1:]):
                layer_after.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                layer_before.base_layer = layer_after.base_layer
            module.merge(safe_merge=safe_merge, adapter_names=adapter_names)

        key_list = [key for key, _ in self.named_modules()]
        desc = "Unloading " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    merge_recursively(target)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)



#################################################################################
#                            ADPDiT Configs                                     #
#################################################################################

ADP_DIT_CONFIG = {
    'DiT-g/2': {'depth': 40, 'hidden_size': 1408, 'patch_size': 2, 'num_heads': 16, 'mlp_ratio': 4.3637},
}


def DiT_g_2(args, **kwargs):
    return ADPDiT(args, depth=40, hidden_size=1408, patch_size=2, num_heads=16, mlp_ratio=4.3637, **kwargs)

ADP_DIT_MODELS = {
    'DiT-g/2':  DiT_g_2,
}
