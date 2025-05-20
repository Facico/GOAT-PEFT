"""
PEFT Configuration

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from peft.utils import CONFIG_NAME
from transformers.utils import PushToHubMixin

from .utils.peft_types import PeftType, TaskType


@dataclass
class PeftConfigMixin(PushToHubMixin):
    """
    Base Configuration Class for PEFT Models
    """
    peft_type: Optional[PeftType] = field(
        default=None, metadata={"help": "The type of Peft method to use."})
    auto_mapping: Optional[dict] = field(
        default=None, metadata={"help": "An auto mapping dict to help retrieve the base model class if needed."}
    )

    def to_dict(self) -> Dict:
        """
        Return the configuration for the adapter model as a dictionary
        """
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """
        Save the configuration of the adapter model in a directory

        :param save_directory: the directory where the configuration will be saved
        :param kwargs: additional keyword arguments
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # Converting set type to list
        output_dict = asdict(self)
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # Save the configuration
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        """
         Loads the configuration of the adapter model from a directory

        :param pretrained_model_name_or_path: the directory or the Hub repository id where the configuration is saved
        :param subfolder: subfolder for the directory
        :param kwargs: additional keyword arguments passed along to the child class initialization
        :return:
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)
        if "peft_type" in loaded_attributes:
            peft_type = loaded_attributes["peft_type"]
            config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        else:
            config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}
        config = config_cls(**kwargs)
        return config

    @classmethod
    def from_json_file(cls, path_json_file: str):
        """
        Load a configuration file from a JSON file

        :param path_json_file: the path to the JSON file
        :return: a JSON object
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)
        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def _get_peft_type(cls, model_id: str, **hf_hub_download_kwargs):
        subfolder = hf_hub_download_kwargs.get("subfolder", None)
        path = os.path.join(model_id, subfolder) if subfolder is not None else model_id

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    model_id,
                    CONFIG_NAME,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{model_id}'")

        loaded_attributes = cls.from_json_file(config_file)
        return loaded_attributes["peft_type"]


@dataclass
class PeftConfig(PeftConfigMixin):
    """
    Base configuration class to store the configuration of a PEFT model
    """
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."})
    revision: Optional[str] = field(
        default=None, metadata={"help": "The specific model version to use."})
    peft_type: Optional[Union[str, PeftType]] = field(
        default=None, metadata={"help": "The type of PEFT method to use."})
    task_type: Optional[Union[str, TaskType]] = field(
        default=None, metadata={"help": "The type of task to perform."})
    inference_mode: bool = field(
        default=False, metadata={"help": "Whether to use the PEFT model in inference mode."})
