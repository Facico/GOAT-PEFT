"""
PEFT and Task Types

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
import enum


class PeftType(str, enum.Enum):
    """
    PEFT Adapter Types
    """
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    GOAT = "GOAT"


class TaskType(str, enum.Enum):
    """
    PEFT Task Type
    """
    CAUSAL_LM = "CAUSAL_LM"
