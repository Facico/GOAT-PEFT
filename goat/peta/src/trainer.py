"""
Trainer
"""
import os
from typing import Optional

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME, logger

from .peft_model import PeftModel


class PeftTrainer(Trainer):
    """
    Trainer for the PEFT Model
    """

    def __init__(self, aux_loss_coeff=1e-2, **kwargs):
        """
        Initialize PeftTrainer

        :param aux_loss_coeff: a coefficient for the load balancing loss in Mixture-of-Experts (MoE) models
        :param kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.aux_loss_coeff = aux_loss_coeff
        
        if torch.cuda.device_count() > 1:
            self.loss_fn = nn.DataParallel(self.loss_fn)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model and tokenizer
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model: PeftModel, inputs, return_outputs=False):
        """
        Compute the loss by the trainer
        """
        outputs = model(**inputs)
        if "loss" in outputs:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            if hasattr(model, 'module') and hasattr(model.module, 'get_aux_loss'):
                aux_loss = model.module.get_aux_loss()
                loss += self.aux_loss_coeff * aux_loss
        if hasattr(model, 'get_aux_loss'):
            aux_loss = model.get_aux_loss()
            loss += self.aux_loss_coeff * aux_loss
        return (loss, outputs) if return_outputs else loss
