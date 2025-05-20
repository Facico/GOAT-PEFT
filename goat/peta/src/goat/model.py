from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from torch import nn

from .config import GOATConfig
from .layer import GOATLayer, LinearGOATLayer
from peft import LoraModel

class GOATModel(LoraModel):
    prefix: str = "lora_"
        
    def _create_and_replace(
        self, goat_config: GOATConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:

        kwargs = {
            "lora_rank": goat_config.r,
            "lora_alpha": goat_config.lora_alpha,
            "lora_dropout": goat_config.lora_dropout,
            "init_lora_weights": goat_config.init_lora_weights,
            "num_experts": goat_config.num_experts,
            "top_k": goat_config.top_k,
            "init_type": goat_config.init_type,
            "init_cof": goat_config.init_cof,
        }

        if isinstance(target, GOATLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if min(target.weight.shape[0],target.weight.shape[1]) < kwargs['lora_rank']:
                return target
            new_module = LinearGOATLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module

    def get_aux_loss(self, adapter_name="default") -> torch.Tensor:
        model_loss = torch.tensor(0, dtype=torch.float).to(self.model.device)
        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                layer_loss = module[adapter_name].layer_loss
                model_loss += layer_loss
        return model_loss

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, GOATLayer):
                module.disable_adapters = False if enabled else True

    def set_adapter(self, adapter_name="default"):

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper
        
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]
        
        _adapters_has_been_set = False
        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)