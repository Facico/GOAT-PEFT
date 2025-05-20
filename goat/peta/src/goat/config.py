from dataclasses import dataclass, field

from peft import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class GOATConfig(LoraConfig):
    num_experts: int = field(default=8, metadata={"help": "The number of experts in MoE."})
    top_k: int = field(default=2, metadata={"help": "The k in top-k gating"})
    init_type: str = field(default="simple")
    init_cof : float = field(default=1.0)

    def __post_init__(self):
        self.peft_type = PeftType.GOAT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
