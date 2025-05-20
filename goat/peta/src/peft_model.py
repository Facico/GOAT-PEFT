"""
PEFT Model

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from peft.utils import (
    WEIGHTS_NAME,
    _set_adapter,
    _set_trainable,
    infer_device,
    load_peft_weights,
)
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from .config import PeftConfig
from .mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_MODEL_MAPPING,
)
from .utils.peft_types import TaskType
from .utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict
)


class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Parameter-Efficient Fine-Tuning (PEFT) Model

    :ivar base_model: base transformer model used for PEFT
    :ivar peft_config: configuration of the PEFT model
    :ivar modules_to_save: list of submodule names to save when saving the model
    """
    base_model: [torch.nn.Module]
    peft_config: [PeftConfig]
    modules_to_save: [str]

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        """
        Initialize PeftModel

        :param model: base transformer model used for PEFT
        :param peft_config: configuration of the PEFT model
        :param adapter_name: name of the adapter
        """
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        peft_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
        self.base_model = peft_model(model, {adapter_name: peft_config}, adapter_name)
        self.set_additional_trainable_modules(peft_config, adapter_name)

        if getattr(model, "is_gradient_checkpointing", True):
            _ = self._prepare_model_for_gradient_checkpointing(model)

    def __getattr__(self, name: str):
        """
        Forward missing attributes to the wrapped module
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def peft_config(self) -> Dict[str, PeftConfig]:
        """
        Get the PEFT configuration
        """
        return self.base_model.peft_config

    @peft_config.setter
    def peft_config(self, value: Dict[str, PeftConfig]):
        """
        Set the PEFT configuration
        """
        self.base_model.peft_config = value

    @property
    def active_adapters(self) -> list[str]:
        """
        Active adapters
        """
        try:
            adapters = self.base_model.active_adapters
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    def _get_base_model_class(self):
        """
        Return the base model class
        """
        return self.base_model.model.__class__

    def get_base_model(self) -> torch.nn.Module:
        """
        Return the base model
        """
        return self.base_model.model

    def set_additional_trainable_modules(self, peft_config: PeftConfig, adapter_name: str) -> None:
        """
        Set additional trainable modules
        """
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        """
        Prepares the model for gradient checkpointing if necessary
        """
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """
        Return the number of trainable parameters and the number of all parameters in the model
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable: {trainable_params / all_param:.2%}")

    def save_pretrained(
        self,
        save_directory: str,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Save the adapter model and the adapter configuration files to a directory, so that it can be reloaded

        :param save_directory: a directory where the adapter model and configuration files will be saved
        :param selected_adapters: a list of adapters to be saved (default to all adapters)
        :param save_embedding_layers: if `True`, save the embedding layers in addition to adapter weights;
            if `auto`, checks the common embedding layers in config's `target_modules` when available
        :param is_main_process: whether the process calling this is the main process or not
        :param kwargs: additional keyword arguments
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]

            # Save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # Save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.model.__dict__.get("name_or_path", None)
                )

            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            if is_main_process:
                peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ) -> PeftModel:
        """
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights (Note that the passed `model`
        may be modified inplace.)

        :param model: the transformer model to be adapted
        :param model_id: the name of the PEFT configuration to use
        :param adapter_name: the name of the adapter to be loaded
        :param is_trainable: whether the adapter should be trainable or not
        :param config: the configuration object to use instead of an automatically loaded configuration
        :param kwargs: additional keyword arguments passed along to the specific PEFT configuration class
        :return: the PEFT model
        """
        # Load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        config.inference_mode = not is_trainable
        model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model

    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any):
        """
        Load a trained adapter into the model (The new adapter is not automatically set as the active adapter.
        Use `PeftModel.set_adapter` to set the active adapter.)

        :param model_id: the name of the adapter to be added
        :param adapter_name: the configuration of the adapter to be added
        :param is_trainable: whether the adapter should be trainable or not
        :param kwargs: additional arguments to modify the way the adapter is loaded
        :return:
        """
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # Load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                **hf_hub_download_kwargs,
            )
            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # Load the weights into the model
        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)
        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)

        # Set model in evaluation mode to deactivate dropout modules by default
        if not is_trainable:
            self.eval()

        return load_result

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        """
        Split keyword arguments
        """
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if (
                key in inspect.signature(hf_hub_download).parameters
                or key in _kwargs_not_in_hf_hub_download_signature
            ):
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration (The new adapter is not automatically set as
        the active adapter. Use `PeftModel.set_adapter` to set the active adapter.)

        :param adapter_name: the name of the adapter to be added
        :param peft_config: the configuration of the adapter to be added
        """
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}.")

        try:
            self.peft_config[adapter_name] = peft_config
            self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter (Only one adapter can be active at a time.)

        :param adapter_name: the name of the adapter to be set as active
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the model
        """
        return self.get_base_model()(*args, **kwargs)


class PeftModelForCausalLM(PeftModel):
    """
    PEFT Model for Causal Language Modeling
    """

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        """
        Initialize PeftModelForCausalLM

        :param model: base transformer model
        :param peft_config: PEFT configuration
        :param adapter_name: adapter name
        """
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward function
        """
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        """
        Generate the text
        """
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Prepare inputs for text generation
        """
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs


MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    TaskType.CAUSAL_LM: PeftModelForCausalLM,
}


def get_peft_model(
    model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default",
) -> PeftModel:
    """
    Return a PEFT model object from a pre-trained model and a PEFT config

    :param model: model to be wrapped
    :param peft_config: configuration containing the parameters of the PEFT model
    :param adapter_name: name of the adapter to be injected
    :return:
    """

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        return PeftModel(model, peft_config, adapter_name=adapter_name)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
