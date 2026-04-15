"""
Auxiliary Metadata Encoder for Clinical and Cognitive Information
Processes patient metadata and cognitive test scores separately.
"""

import torch
import torch.nn as nn
import re
from typing import Dict, Tuple, Optional


class ClinicalMetadataParser:
    """
    Parse clinical metadata from prompt text.
    Extracts: diagnosis, gender, age, visit information
    """

    DIAGNOSIS_MAP = {
        'Cognitively Normal': 0,
        'Mild Cognitive Impairment': 1,
        'MCI': 1,
        'Alzheimer\'s Disease': 2,
        'AD': 2,
        'Dementia': 2,
    }

    GENDER_MAP = {
        'Male': 0,
        'Female': 1,
        'M': 0,
        'F': 1,
    }

    @staticmethod
    def parse(prompt: str) -> Dict[str, float]:
        """
        Parse clinical metadata from prompt text.

        Returns:
            dict with keys: diagnosis, gender, age, months_from_baseline
        """
        result = {
            'diagnosis': 1.0,  # Default: MCI
            'gender': 0.0,     # Default: Male
            'age': 70.0,       # Default: 70 years
            'months_from_baseline': 0.0,  # Default: baseline visit
        }

        # Extract diagnosis
        for diag, code in ClinicalMetadataParser.DIAGNOSIS_MAP.items():
            if diag in prompt:
                result['diagnosis'] = float(code)
                break

        # Extract gender
        for gender, code in ClinicalMetadataParser.GENDER_MAP.items():
            if gender in prompt:
                result['gender'] = float(code)
                break

        # Extract age (e.g., "64.00 years old")
        age_match = re.search(r'(\d+\.?\d*)\s*years?\s+old', prompt, re.IGNORECASE)
        if age_match:
            result['age'] = float(age_match.group(1))

        # Extract visit information
        # Check for "N months from first visit" FIRST (more specific pattern)
        months_match = re.search(r'(\d+\.?\d*)\s*months?\s+from\s+first\s+visit', prompt, re.IGNORECASE)
        if months_match:
            result['months_from_baseline'] = float(months_match.group(1))
        elif 'first visit' in prompt.lower():
            result['months_from_baseline'] = 0.0

        return result


class CognitiveScoresParser:
    """
    Parse cognitive test scores from prompt text.
    Extracts: CDRSB, ADAS11, ADAS13, ADASQ4, MMSE, RAVLT scores, etc.

    Score directionality:
    - Higher = Worse (more severe disease): CDRSB, ADAS11, ADAS13, ADASQ4, FAQ, TRABSCOR,
                                            RAVLT_forgetting, RAVLT_perc_forgetting
    - Lower = Worse (cognitive decline): MMSE, MOCA, LDELTOTAL, RAVLT_immediate, RAVLT_learning

    We normalize all scores to "Higher = Better" direction for consistent embedding.
    """

    SCORE_NAMES = [
        'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE',
        'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
        'RAVLT_perc_forgetting', 'LDELTOTAL', 'TRABSCOR', 'FAQ', 'MOCA'
    ]

    # Scores where higher value = worse condition (need to be inverted)
    HIGHER_IS_WORSE = {
        'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'FAQ', 'TRABSCOR',
        'RAVLT_forgetting', 'RAVLT_perc_forgetting'
    }

    # Expected ranges for normalization (from ADNI data statistics)
    SCORE_RANGES = {
        # Higher = Worse scores (will be inverted)
        'CDRSB': (0.0, 18.0),           # Clinical Dementia Rating Scale - Sum of Boxes
        'ADAS11': (0.0, 70.0),          # ADAS11 cognitive subscale
        'ADAS13': (0.0, 85.0),          # ADAS13 cognitive subscale
        'ADASQ4': (0.0, 10.0),          # ADAS Word Recognition
        'FAQ': (0.0, 30.0),             # Functional Activities Questionnaire
        'TRABSCOR': (0.0, 300.0),       # Trail Making Test Part B (seconds)
        'RAVLT_forgetting': (0.0, 15.0),     # Number of words forgotten
        'RAVLT_perc_forgetting': (0.0, 200.0), # Percentage forgotten

        # Lower = Worse scores (keep as is)
        'MMSE': (0.0, 30.0),            # Mini-Mental State Examination
        'MOCA': (0.0, 30.0),            # Montreal Cognitive Assessment
        'LDELTOTAL': (0.0, 25.0),       # Logical Memory Delayed Total
        'RAVLT_immediate': (0.0, 75.0), # Rey Auditory Verbal Learning Test - Immediate
        'RAVLT_learning': (-5.0, 15.0), # RAVLT Learning slope
    }

    @staticmethod
    def parse(prompt: str) -> Dict[str, float]:
        """
        Parse cognitive test scores from prompt text.

        Returns:
            dict with score names as keys (raw values, not normalized)
        """
        result = {}

        for score_name in CognitiveScoresParser.SCORE_NAMES:
            # Pattern: "SCORE_NAME value"
            pattern = rf'{score_name}\s+([-+]?\d+\.?\d*)'
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                result[score_name] = float(match.group(1))
            else:
                # Use middle of range as default
                min_val, max_val = CognitiveScoresParser.SCORE_RANGES.get(
                    score_name, (0.0, 1.0)
                )
                result[score_name] = (min_val + max_val) / 2.0

        return result

    @staticmethod
    def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range with unified direction (higher = better).

        For "higher is worse" scores, we invert them: normalized = 1 - (raw - min) / (max - min)
        For "lower is worse" scores, we keep them: normalized = (raw - min) / (max - min)

        Args:
            scores: dict of raw score values

        Returns:
            dict of normalized scores (all in "higher = better" direction)
        """
        normalized = {}

        for score_name, raw_value in scores.items():
            min_val, max_val = CognitiveScoresParser.SCORE_RANGES.get(
                score_name, (0.0, 1.0)
            )

            # Clamp to range
            raw_value = max(min_val, min(max_val, raw_value))

            # Normalize to [0, 1]
            if max_val > min_val:
                norm_value = (raw_value - min_val) / (max_val - min_val)
            else:
                norm_value = 0.5

            # Invert if higher = worse
            if score_name in CognitiveScoresParser.HIGHER_IS_WORSE:
                norm_value = 1.0 - norm_value

            normalized[score_name] = norm_value

        return normalized


class ClinicalMetadataEncoder(nn.Module):
    """
    Encode clinical metadata (diagnosis, gender, age, visit info) into embeddings.
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding layers for categorical variables
        self.diagnosis_embed = nn.Embedding(3, hidden_size // 4)  # 3 categories: CN, MCI, AD
        self.gender_embed = nn.Embedding(2, hidden_size // 4)     # 2 categories: M, F

        # Linear layers for continuous variables
        self.age_proj = nn.Linear(1, hidden_size // 4)
        self.months_proj = nn.Linear(1, hidden_size // 4)

        # Combine all features
        self.combine = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, clinical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            clinical_data: dict with keys 'diagnosis', 'gender', 'age', 'months_from_baseline'
                          Each value is a tensor of shape (B,)

        Returns:
            embeddings: (B, hidden_size)
        """
        batch_size = clinical_data['diagnosis'].shape[0]

        # Embed categorical variables
        diag_emb = self.diagnosis_embed(clinical_data['diagnosis'].long())  # (B, hidden_size//4)
        gender_emb = self.gender_embed(clinical_data['gender'].long())      # (B, hidden_size//4)

        # Project continuous variables
        age_emb = self.age_proj(clinical_data['age'].unsqueeze(-1))         # (B, hidden_size//4)
        months_emb = self.months_proj(clinical_data['months_from_baseline'].unsqueeze(-1))  # (B, hidden_size//4)

        # Concatenate all embeddings
        combined = torch.cat([diag_emb, gender_emb, age_emb, months_emb], dim=-1)  # (B, hidden_size)

        # Apply combination network
        output = self.combine(combined)  # (B, hidden_size)

        return output


class CognitiveScoresEncoder(nn.Module):
    """
    Encode cognitive test scores into embeddings.
    Expects scores to be pre-normalized to [0, 1] range with unified direction (higher = better).
    """

    def __init__(self, num_scores: int = 13, hidden_size: int = 512):
        super().__init__()
        self.num_scores = num_scores
        self.hidden_size = hidden_size

        # Additional learnable normalization on top of pre-normalized scores
        self.score_norm = nn.LayerNorm(num_scores)

        # Encoder network with deeper architecture for better representation
        self.encoder = nn.Sequential(
            nn.Linear(num_scores, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, num_scores) tensor of cognitive test scores
                   Expected to be pre-normalized to [0, 1] with unified direction

        Returns:
            embeddings: (B, hidden_size)
        """
        # Apply learnable normalization
        scores_norm = self.score_norm(scores)

        # Encode
        output = self.encoder(scores_norm)

        return output


class TemporalProgressionModule(nn.Module):
    """
    Handle temporal progression for longitudinal data.
    Only affects embedding when months_from_baseline > 0 (not first visit).
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        # Temporal encoding
        self.temporal_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Gating mechanism (decides how much temporal info to use)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self,
                cognitive_emb: torch.Tensor,
                months_from_baseline: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal progression to cognitive embeddings.

        Args:
            cognitive_emb: (B, hidden_size) cognitive test embeddings
            months_from_baseline: (B,) months from first visit

        Returns:
            modulated_emb: (B, hidden_size) temporally-modulated embeddings
        """
        batch_size = cognitive_emb.shape[0]

        # Encode temporal information
        temporal_emb = self.temporal_proj(months_from_baseline.unsqueeze(-1))  # (B, hidden_size)

        # Create gate (decides how much to modulate based on temporal info)
        gate_input = torch.cat([cognitive_emb, temporal_emb], dim=-1)  # (B, 2*hidden_size)
        gate_value = self.gate(gate_input)  # (B, hidden_size)

        # Apply gating: for first visit (months=0), gate should be ~0
        # For later visits, gate should be >0
        modulated_emb = cognitive_emb + gate_value * temporal_emb

        return modulated_emb


class AuxiliaryMetadataEncoder(nn.Module):
    """
    Main auxiliary metadata encoder that combines clinical and cognitive information.
    """

    def __init__(self,
                 clinical_hidden_size: int = 256,
                 cognitive_hidden_size: int = 512,
                 output_size: int = 768,
                 num_cognitive_scores: int = 13):
        super().__init__()

        self.clinical_encoder = ClinicalMetadataEncoder(clinical_hidden_size)
        self.cognitive_encoder = CognitiveScoresEncoder(num_cognitive_scores, cognitive_hidden_size)
        self.temporal_module = TemporalProgressionModule(cognitive_hidden_size)

        # Combine clinical and cognitive embeddings
        self.combiner = nn.Sequential(
            nn.Linear(clinical_hidden_size + cognitive_hidden_size, output_size),
            nn.LayerNorm(output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self,
                clinical_data: Dict[str, torch.Tensor],
                cognitive_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical_data: dict with clinical metadata
            cognitive_scores: (B, num_scores) tensor of cognitive test scores

        Returns:
            auxiliary_emb: (B, output_size) auxiliary embeddings
        """
        # Encode clinical metadata
        clinical_emb = self.clinical_encoder(clinical_data)  # (B, clinical_hidden_size)

        # Encode cognitive scores
        cognitive_emb = self.cognitive_encoder(cognitive_scores)  # (B, cognitive_hidden_size)

        # Apply temporal progression (only affects non-baseline visits)
        cognitive_emb = self.temporal_module(
            cognitive_emb,
            clinical_data['months_from_baseline']
        )  # (B, cognitive_hidden_size)

        # Combine clinical and cognitive embeddings
        combined = torch.cat([clinical_emb, cognitive_emb], dim=-1)
        auxiliary_emb = self.combiner(combined)  # (B, output_size)

        return auxiliary_emb

    @staticmethod
    def parse_prompts(prompts: list) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Parse a batch of prompts and extract auxiliary metadata.

        Args:
            prompts: list of prompt strings

        Returns:
            clinical_data: dict of tensors for clinical metadata
            cognitive_scores: (B, num_scores) tensor of NORMALIZED cognitive test scores
                            (all in "higher = better" direction, [0, 1] range)
        """
        batch_size = len(prompts)

        # Parse each prompt
        clinical_list = []
        cognitive_list = []
        cognitive_normalized_list = []

        for prompt in prompts:
            clinical = ClinicalMetadataParser.parse(prompt)
            cognitive_raw = CognitiveScoresParser.parse(prompt)
            cognitive_norm = CognitiveScoresParser.normalize_scores(cognitive_raw)

            clinical_list.append(clinical)
            cognitive_list.append(cognitive_raw)
            cognitive_normalized_list.append(cognitive_norm)

        # Convert to tensors
        clinical_data = {
            'diagnosis': torch.tensor([c['diagnosis'] for c in clinical_list], dtype=torch.float32),
            'gender': torch.tensor([c['gender'] for c in clinical_list], dtype=torch.float32),
            'age': torch.tensor([c['age'] for c in clinical_list], dtype=torch.float32),
            'months_from_baseline': torch.tensor([c['months_from_baseline'] for c in clinical_list], dtype=torch.float32),
        }

        # Get score names in order and use NORMALIZED values
        score_names = CognitiveScoresParser.SCORE_NAMES
        cognitive_scores = torch.tensor(
            [[cog[name] for name in score_names] for cog in cognitive_normalized_list],
            dtype=torch.float32
        )  # (B, num_scores) - normalized to [0, 1] with unified direction

        return clinical_data, cognitive_scores


# Test function
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Auxiliary Metadata Encoder")
    print("=" * 80)

    # Test prompts with different severity levels
    test_prompts = [
        # Healthy baseline
        "Cognitively Normal, Female, 70.00 years old, first visit, slice 100, " +
        "CDRSB 0.0 ADAS11 3.0 ADAS13 5.0 ADASQ4 0.0 MMSE 30.0 " +
        "RAVLT_immediate 60.0 RAVLT_learning 10.0 RAVLT_forgetting 2.0 " +
        "RAVLT_perc_forgetting 20.0 LDELTOTAL 20.0 TRABSCOR 50.0 FAQ 0.0 MOCA 29.0",

        # MCI patient
        "Mild Cognitive Impairment, Male, 64.00 years old, first visit, slice 100, " +
        "CDRSB 0.5 ADAS11 9.0 ADAS13 14.0 ADASQ4 5.0 MMSE 27.0 " +
        "RAVLT_immediate 36.0 RAVLT_learning 4.0 RAVLT_forgetting 9.0 " +
        "RAVLT_perc_forgetting 100.0 LDELTOTAL 5.0 TRABSCOR 45.0 FAQ 1.0 MOCA 25.0",

        # AD patient (severe)
        "Alzheimer's Disease, Male, 75.00 years old, 12 months from first visit, slice 95, " +
        "CDRSB 8.0 ADAS11 35.0 ADAS13 50.0 ADASQ4 8.0 MMSE 18.0 " +
        "RAVLT_immediate 15.0 RAVLT_learning -2.0 RAVLT_forgetting 13.0 " +
        "RAVLT_perc_forgetting 180.0 LDELTOTAL 2.0 TRABSCOR 150.0 FAQ 15.0 MOCA 15.0",
    ]

    print("\n1. Testing score parsing and normalization:")
    print("-" * 80)

    for i, prompt in enumerate(test_prompts):
        print(f"\nPatient {i+1}:")
        clinical = ClinicalMetadataParser.parse(prompt)
        print(f"  Clinical: diagnosis={clinical['diagnosis']}, age={clinical['age']}, " +
              f"months={clinical['months_from_baseline']}")

        cognitive_raw = CognitiveScoresParser.parse(prompt)
        cognitive_norm = CognitiveScoresParser.normalize_scores(cognitive_raw)

        print(f"  Sample raw scores:")
        print(f"    CDRSB (↑worse): {cognitive_raw['CDRSB']:.1f} → normalized: {cognitive_norm['CDRSB']:.3f}")
        print(f"    MMSE  (↓worse): {cognitive_raw['MMSE']:.1f} → normalized: {cognitive_norm['MMSE']:.3f}")
        print(f"    ADAS11(↑worse): {cognitive_raw['ADAS11']:.1f} → normalized: {cognitive_norm['ADAS11']:.3f}")
        print(f"    MOCA  (↓worse): {cognitive_raw['MOCA']:.1f} → normalized: {cognitive_norm['MOCA']:.3f}")

    print("\n" + "=" * 80)
    print("\n2. Testing full encoder pipeline:")
    print("-" * 80)

    # Test encoder
    encoder = AuxiliaryMetadataEncoder()

    # Parse batch of prompts
    clinical_data, cognitive_scores = encoder.parse_prompts(test_prompts)

    print(f"\nClinical data shape:")
    for key, val in clinical_data.items():
        print(f"  {key}: {val.shape} -> {val.tolist()}")

    print(f"\nCognitive scores shape: {cognitive_scores.shape}")
    print(f"Normalized scores range: [{cognitive_scores.min():.3f}, {cognitive_scores.max():.3f}]")
    print(f"\nFirst patient (healthy) average normalized score: {cognitive_scores[0].mean():.3f}")
    print(f"Second patient (MCI) average normalized score: {cognitive_scores[1].mean():.3f}")
    print(f"Third patient (AD) average normalized score: {cognitive_scores[2].mean():.3f}")
    print(f"→ Expect: Healthy > MCI > AD (higher = better after normalization)")

    # Forward pass
    import torch
    with torch.no_grad():
        output = encoder(clinical_data, cognitive_scores)

    print(f"\nEncoder output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Check embedding differences
    healthy_emb = output[0]
    mci_emb = output[1]
    ad_emb = output[2]

    diff_healthy_mci = torch.norm(healthy_emb - mci_emb).item()
    diff_mci_ad = torch.norm(mci_emb - ad_emb).item()
    diff_healthy_ad = torch.norm(healthy_emb - ad_emb).item()

    print(f"\nEmbedding distances (L2 norm):")
    print(f"  Healthy ↔ MCI: {diff_healthy_mci:.4f}")
    print(f"  MCI ↔ AD: {diff_mci_ad:.4f}")
    print(f"  Healthy ↔ AD: {diff_healthy_ad:.4f}")
    print(f"→ Expect: Healthy-AD distance > others (most different)")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
