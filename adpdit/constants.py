import torch

# =======================================================
NOISE_SCHEDULES = {
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
}

PREDICT_TYPE = {
    "epsilon",
    "sample",
    "v_prediction",
}

# =======================================================
NEGATIVE_PROMPT = 'significant changes, large anatomical shift, brain shrinkage, distorted anatomy, unrelated artifacts'

# =======================================================
TRT_MAX_BATCH_SIZE = 1
TRT_MAX_WIDTH = 1280
TRT_MAX_HEIGHT = 1280

# =======================================================
# Constants about models
# =======================================================
VAE_EMA_PATH = "ckpts/t2i/sdxl-vae-fp16-fix"
TOKENIZER = "ckpts/t2i/tokenizer"
TEXT_ENCODER = 'ckpts/t2i/clip_text_encoder'

T5_ENCODER = {
    'T5': 'ckpts/t2i/T5',
    'attention_mask': True,
    'layer_index': -1,
    'attention_pool': True,
    'torch_dtype': torch.float16,
    'learnable_replace': True
}

# =======================================================
# Scheduler configuration
# =======================================================

SAMPLER_FACTORY = {
    'ddpm': {
        'scheduler': 'DDPMScheduler',
        'name': 'DDPM',
        'kwargs': {
            'num_train_timesteps': 1000,
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
        }
    },
    'ddim': {
        'scheduler': 'DDIMScheduler',
        'name': 'DDIM',
        'kwargs': {
            'num_train_timesteps': 1000,
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
        }
    },
    'dpmms': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPMMS',
        'kwargs': {
            'num_train_timesteps': 1000,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
            'solver_order': 2,
            'algorithm_type': 'dpmsolver++',
        }
    },
    'dpmpp_2m_sde_exp': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPM++ 2M SDE Exp',
        'kwargs': {
            'num_train_timesteps': 1000,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
        },
        'config': {
            'algorithm_type': 'dpmsolver++',
            'solver_order': 2,
            'use_karras_sigmas': False,
        }
    },
    'dpmpp_3m_sde_karras': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPM++ 3M SDE Karras',
        'kwargs': {
            'num_train_timesteps': 1000,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
        },
        'config': {
            'algorithm_type': 'dpmsolver++',
            'solver_order': 3,
            'use_karras_sigmas': True,
        }
    },
    'dpmpp_2m_karras': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPM++ 2M Karras',
        'kwargs': {
            'num_train_timesteps': 1000,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.018,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
        },
        'config': {
            'algorithm_type': 'dpmsolver++',
            'solver_order': 2,
            'use_karras_sigmas': True,
        }
    }
}
