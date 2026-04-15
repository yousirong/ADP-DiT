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
from transformers import BertForMaskedLM, CLIPModel  # BERT and CLIP
from transformers.integrations import PeftAdapterMixin

from .attn_layers import Attention, FlashCrossMHAModified, FlashSelfMHAModified, CrossAttention
from .embedders import TimestepEmbedder, PatchEmbed, timestep_embedding
from .norm_layers import RMSNorm
from .poolers import AttentionPool


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class FP32_SiLU(nn.SiLU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)


class ADPDiTBlock(nn.Module):
    def __init__(self, hidden_size, c_emb_size, num_heads, mlp_ratio=4.0, text_states_dim=1024, use_flash_attn=False):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        # Flash Attention or Standard Attention
        if use_flash_attn:
            self.attn1 = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True)
            self.attn2 = FlashCrossMHAModified(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True)
        else:
            self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
            self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True)

        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=FP32_SiLU)
        self.norm1 = FP32_Layernorm(hidden_size)
        self.norm2 = FP32_Layernorm(hidden_size)
        self.norm3 = FP32_Layernorm(hidden_size)
        self.default_modulation = nn.Sequential(FP32_SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True))

    def forward(self, x, c=None, text_states=None):
        shift_msa = self.default_modulation(c).unsqueeze(1)
        x = x + self.attn1(self.norm1(x) + shift_msa)
        x = x + self.attn2(self.norm3(x), text_states)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(FP32_SiLU(), nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ADPDiT(ModelMixin, ConfigMixin, PeftAdapterMixin):
    @register_to_config
    def __init__(self, args, input_size=(32, 32), patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16):
        super().__init__()
        self.use_bert = args.use_bert  # Choose BERT or CLIP
        if self.use_bert:
            self.text_states_dim = 768  # BERT hidden size
            print(f"Loading pretrained BERT model...")
            self.text_encoder = BertForMaskedLM.from_pretrained('bert-base-uncased')
        else:
            self.text_states_dim = 1024  # CLIP hidden size
            print(f"Loading pretrained CLIP model...")
            self.text_encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

        self.blocks = nn.ModuleList([
            ADPDiTBlock(hidden_size, hidden_size, num_heads, text_states_dim=self.text_states_dim, use_flash_attn=args.use_flash_attn)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, in_channels)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

    def forward(self, x, t, encoder_hidden_states=None):
        if self.use_bert:
            encoder_outputs = self.text_encoder.bert(
                input_ids=encoder_hidden_states['input_ids'],
                attention_mask=encoder_hidden_states['attention_mask']
            )
            text_states = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        else:
            text_states = self.text_encoder.text_model(
                input_ids=encoder_hidden_states['input_ids'],
                attention_mask=encoder_hidden_states['attention_mask']
            ).last_hidden_state

        t = self.t_embedder(t)
        x = self.x_embedder(x)
        for block in self.blocks:
            x = block(x, c=t, text_states=text_states)
        x = self.final_layer(x, t)
        return x


ADP_DIT_CONFIG = {
    'DiT-g/2': {'depth': 40, 'hidden_size': 1408, 'patch_size': 2, 'num_heads': 16, 'mlp_ratio': 4.3637},
}


def DiT_g_2(args, **kwargs):
    return ADPDiT(args, depth=40, hidden_size=1408, patch_size=2, num_heads=16, mlp_ratio=4.3637, **kwargs)


ADP_DIT_MODELS = {
    'DiT-g/2':  DiT_g_2,
}
