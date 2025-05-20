"""
Configure and Model Mappings

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""

from .goat import GOATConfig, GOATModel
from .utils.peft_types import PeftType

PEFT_TYPE_TO_CONFIG_MAPPING = {
    PeftType.GOAT: GOATConfig,
}
PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.GOAT: GOATModel,
}
