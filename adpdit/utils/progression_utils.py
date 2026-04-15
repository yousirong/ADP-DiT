"""
Utility functions for disease progression inference.

Handles patient grouping and progression type classification based on prompts.
"""
import re
from typing import List, Tuple, Dict


def extract_condition_from_prompt(prompt: str) -> str:
    """
    Extract disease condition from prompt text.

    Args:
        prompt: Text prompt containing disease condition

    Returns:
        Disease condition: "Cognitive Normal", "Mild Cognitive Impairment", or "Alzheimer Disease"
    """
    prompt_lower = prompt.lower()

    # Check in order of specificity (most specific first)
    if "alzheimer disease" in prompt_lower or "alzheimer's disease" in prompt_lower:
        return "Alzheimer Disease"
    elif "mild cognitive impairment" in prompt_lower:
        return "Mild Cognitive Impairment"
    elif "cognitive normal" in prompt_lower:
        return "Cognitive Normal"
    else:
        # Default to CN if no match
        return "Cognitive Normal"


def extract_cls_label_from_prompt(prompt: str) -> int:
    """
    Extract classification label from prompt.

    Args:
        prompt: Text prompt containing disease condition

    Returns:
        Classification label: 0 (CN), 1 (MCI), 2 (AD)
    """
    condition = extract_condition_from_prompt(prompt)

    if condition == "Cognitive Normal":
        return 0
    elif condition == "Mild Cognitive Impairment":
        return 1
    elif condition == "Alzheimer Disease":
        return 2
    else:
        return 0  # Default to CN


def is_first_visit(prompt: str) -> bool:
    """
    Check if prompt indicates first visit (NOT a follow-up visit).

    Args:
        prompt: Text prompt

    Returns:
        True if first visit (without months indicator), False otherwise
    """
    prompt_lower = prompt.lower()
    # Must contain "first visit" but NOT "months from first visit"
    if "first visit" in prompt_lower:
        # Check if it's a follow-up (contains "months from first visit" or "month from first visit")
        if "months from first visit" in prompt_lower or "month from first visit" in prompt_lower:
            return False
        # Otherwise, it's a true first visit
        return True
    return False


def group_by_patient(data_rows: List[Tuple]) -> List[List[Tuple]]:
    """
    Group data rows by patient (consecutive rows until next 'first visit').

    Args:
        data_rows: List of (input_image, edited_image, edit_prompt) tuples

    Returns:
        List of patient groups, where each group is a list of visits for that patient
    """
    patients = []
    current_patient = []

    for row in data_rows:
        input_img, edited_img, prompt = row

        if is_first_visit(prompt):
            # Start of new patient
            if current_patient:
                patients.append(current_patient)
            current_patient = [row]
        else:
            # Continuation of current patient
            current_patient.append(row)

    # Don't forget the last patient
    if current_patient:
        patients.append(current_patient)

    return patients


def classify_progression_type(input_prompt: str, target_prompt: str) -> str:
    """
    Classify the type of progression based on input and target conditions.

    Args:
        input_prompt: Prompt for input image
        target_prompt: Prompt for target/edited image

    Returns:
        Progression type: "CNtoCN", "CNtoMCI", "MCItoMCI", "MCItoAD", "ADtoAD"
    """
    input_condition = extract_condition_from_prompt(input_prompt)
    target_condition = extract_condition_from_prompt(target_prompt)

    # Map to short forms
    condition_map = {
        "Cognitive Normal": "CN",
        "Mild Cognitive Impairment": "MCI",
        "Alzheimer Disease": "AD"
    }

    input_short = condition_map[input_condition]
    target_short = condition_map[target_condition]

    return f"{input_short}to{target_short}"


def get_conversion_folder(input_prompt: str, target_prompt: str) -> str:
    """
    Get the output folder name based on conversion type.

    Args:
        input_prompt: Prompt for input image
        target_prompt: Prompt for target/edited image

    Returns:
        Folder name: "CNtoCN", "CNtoMCI", "MCItoMCI", "MCItoAD", or "ADtoAD"
    """
    return classify_progression_type(input_prompt, target_prompt)


def is_valid_progression(input_prompt: str, target_prompt: str) -> bool:
    """
    Check if the progression is valid (allowed combination).

    Allowed progressions:
    - CN -> CN (stable)
    - CN -> MCI (progression)
    - MCI -> MCI (stable)
    - MCI -> AD (progression)
    - AD -> AD (stable)

    Args:
        input_prompt: Prompt for input image
        target_prompt: Prompt for target/edited image

    Returns:
        True if valid progression, False otherwise
    """
    input_condition = extract_condition_from_prompt(input_prompt)
    target_condition = extract_condition_from_prompt(target_prompt)

    allowed_pairs = {
        ("Cognitive Normal", "Cognitive Normal"),
        ("Cognitive Normal", "Mild Cognitive Impairment"),
        ("Mild Cognitive Impairment", "Mild Cognitive Impairment"),
        ("Mild Cognitive Impairment", "Alzheimer Disease"),
        ("Alzheimer Disease", "Alzheimer Disease")
    }

    return (input_condition, target_condition) in allowed_pairs


def create_progression_prompt(target_prompt: str, input_condition: str, target_condition: str) -> str:
    """
    Create progression-aware prompt for better guidance.

    Args:
        target_prompt: Original target prompt
        input_condition: Input disease condition
        target_condition: Target disease condition

    Returns:
        Enhanced prompt with progression context
    """
    # Remove slice information
    prompt_cleaned = re.sub(r',?\s*slice[:\s_-]*\d+', '', target_prompt, flags=re.IGNORECASE)
    prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned).strip()

    # Add progression context if conditions differ
    if input_condition != target_condition:
        progression_prefix = f"progression from {input_condition} to {target_condition}, transitioning to {target_condition}, "

        # Add pathological features for AD
        if target_condition == "Alzheimer Disease":
            pathology = "enlarged ventricles, cortical atrophy, hippocampal atrophy, thinning cortical ribbon, "
            prompt_cleaned = progression_prefix + pathology + prompt_cleaned
        else:
            prompt_cleaned = progression_prefix + prompt_cleaned

    return prompt_cleaned
