import os
import argparse
import logging
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    TextGenerationPipeline,
    GenerationConfig,
    TrainerCallback,
)
import torch.distributed as dist
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from torchvision.transforms import RandomResizedCrop, Resize, CenterCrop, Compose, Normalize, ToTensor, Lambda, InterpolationMode
from transformers.image_transforms import convert_to_rgb 
from datasets import load_dataset
import tqdm
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict, load_dataset
import transformers
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import default_data_collator
import importlib
import peft
from peft import LoraConfig, PeftModel
from packaging import version
import peta
from peta.utils import TitledLog
from transformers import AutoImageProcessor,AutoFeatureExtractor,AutoProcessor
import wandb

import math

log = logging.getLogger(__name__)
# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

def _is_peft_model(model):
    classes_to_check = (PeftModel,)
    # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
        from peft import PeftMixedModel

        classes_to_check = (*classes_to_check, PeftMixedModel)
    return isinstance(model, classes_to_check)


import pandas as pd
def save_csv(data, out_path):
    # save excel
    columns = sorted(list(data.keys()))
    df = pd.DataFrame(data, index=[0]).reindex(columns=columns)
    os.makedirs(out_path, exist_ok=True)
    xlsx_path = os.path.join(out_path, 'results.csv')

    if os.path.exists(xlsx_path):
        previous = pd.read_csv(xlsx_path, index_col=0)
        df = pd.concat([previous, df])

    df.to_csv(xlsx_path, index=True)

def split_dataset(dataset, rank, world_size):
    total_size = len(dataset)
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    return subset

def gather_from_all_processes(data):
    gathered_data = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_data, data)
    # Flatten the list of lists
    return [item for sublist in gathered_data for item in sublist.cpu().tolist()]

class CustomCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset, args, **kwargs) -> None:
        super().__init__()
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.local_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if self.world_size > 1:
            self.test_dataset = split_dataset(test_dataset, self.local_rank, self.world_size)
        self.args = args

    def on_epoch_end(self, args, state, control, **kwargs):

        with torch.inference_mode():

            label, predict = [], []
            train_loader = torch.utils.data.DataLoader(
                self.test_dataset,  # Replace with your dataset
                batch_size=32,         # Set batch size
                shuffle=False,         # No need to shuffle during inference
                collate_fn=collate_fn  # Use the same collate function
            )
            for batch in tqdm.tqdm(train_loader):
                inputs = {k: v.to('cuda') for k, v in batch.items() if k != 'labels'}
                outputs = self.trainer.model(**inputs)
                label.append(batch['labels'].to('cuda'))
                predict.append(outputs.logits.argmax(-1))
            label = torch.cat(label, dim=0)
            predict = torch.cat(predict, dim=0)
        if self.world_size > 1:
            predict = gather_from_all_processes(predict)
            label = gather_from_all_processes(label)
        else:
            predict = predict.cpu().tolist()
            label = label.cpu().tolist()
        
        if self.local_rank == 0:
            
            correct_rate = sum(p==q for p,q in zip(predict, label)) / len(predict)
            wandb.log({f'test/{self.args.task}-acc': correct_rate}, step=self.trainer._globalstep_last_logged)
            try:
                ans = {
                    'task': self.args.task,
                    'model': self.args.prj +'/' + str(self.trainer._globalstep_last_logged),
                    'acc':  round(correct_rate.item() * 100, 2),
                }
            except:
                ans = {
                    'task': self.args.task,
                    'model': self.args.prj +'/' + str(self.trainer._globalstep_last_logged),
                    'acc':  round(correct_rate * 100, 2),
                }
            save_csv(ans, self.args.result)


class CustomTrainer(Trainer):
    def __init__(self, *args, cus_args=None, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cus_args = cus_args
        if 'ft' not in cus_args.lora:
            self.scaling_factor = scaling_factor
        self.aux_loss_coeff = self.cus_args.aux_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs['labels']
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'get_aux_loss'):
                aux_loss = unwrapped_model.get_aux_loss()
                loss += self.aux_loss_coeff * aux_loss

        return (loss, outputs) if return_outputs else loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', default="lora-pro", type=str)
    parser.add_argument('--model', default="roberta-large", type=str)
    parser.add_argument('--result', default="", type=str)
    parser.add_argument('--prj', default="lora-pro", type=str)
    parser.add_argument('--task', default="math", type=str)
    parser.add_argument('--output', default="output", type=str)
    parser.add_argument('--experts', default=1, type=int)
    parser.add_argument('--aux_loss_coeff', default=1., type=float)
    parser.add_argument('--shared_experts', default=1, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help="Resume training from the last checkpoint.")              
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Resume training from the last checkpoint.")              
    parser.add_argument('--ep', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--bz', default=1, type=int)
    parser.add_argument('--gacc', default=1, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--sched', default="cosine", type=str)
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--alpha', default=16, type=int)
    parser.add_argument('--git_hash', default='', type=str)
    parser.add_argument('--modules', type=str, default='qkvoudg', help='target modules in lora layers')
    args = parser.parse_args()
    # assert args.lora in ["lora-pro", "rslora-pro"]
    return args      


epochs = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR10": 6,
    "CIFAR100": 6,
    "STL10": 60,
    "Food101": 4,
    "Flowers102": 147,
    "FER2013": 10,
    "PCAM": 1,
    "OxfordIIITPet": 82,
    "RenderedSST2": 39,
    "EMNIST": 2,
    "FashionMNIST": 5,
    "KMNIST": 5,
}

def collate_fn(examples):
    # [b, 3, 224, 224]
    pixel_values = torch.stack([torch.tensor(example[0]) for example in examples])
    assert pixel_values.shape[1:] == (3,224,224)
    # [b, ]
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    
    args = get_arguments()
    
    log.info(f"set seed to {args.seed}")
    transformers.set_seed(args.seed)
    set_seed(args.seed)
    
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if rank == 0:
        wandb.init(
            project=f'{args.model}-LoRA'.replace('/',''),  
            name=args.prj,
            config=args,
        )
    
    lora_r = args.rank
    lora_alpha = args.alpha
    path = args.model
    image_processor = transformers.CLIPImageProcessor.from_pretrained(path)
    text_processor = AutoProcessor.from_pretrained(path)

    train_transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        RandomResizedCrop(
            size=image_processor.size["shortest_edge"],
            scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC,
        ), 
        ToTensor(), 
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    val_transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        Resize(size=image_processor.size["shortest_edge"], interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(size=image_processor.size["shortest_edge"]),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        from peta.tasks.cv import get_dataset
        dataset = get_dataset(
            args.task + 'Val',  # xxVal is the train set !
            train_transform, 
            location='../dataset/~data', 
            batch_size=1
        )
        test_dataset = get_dataset(
            args.task,  # xx is the test set !
            val_transform, 
            location='../dataset/~data', 
            batch_size=1     
        ).test_dataset

    id2label = dataset.classnames
    label2id = {label: i for i, label in enumerate(id2label)}
    dataset = dataset.train_dataset

    # [goat setting: ]The classifier is obtained using prompts such as “a photo of a {class}” and kept frozen during fine-tuning
    model = transformers.CLIPForImageClassification.from_pretrained(
        path, 
        label2id=label2id,
        id2label=id2label, # set label number
        ignore_mismatched_sizes=True,  # Allow resizing the final classification layer
        # attn_implementation="flash_attention_2", 
        # torch_dtype=torch.bfloat16, 
        device_map={"": local_rank}
    )
    model.config.hidden_size = model.config.vision_config.hidden_size
    
    scaling_factor = 2
    if 'ft' != args.lora:
        lora_r = args.rank
        lora_alpha = args.alpha
        target_modules = 'all-linear'

        if 'src' in args.lora:

            import peta.src
            try:
                peft_type, init_type, init_cof = args.lora.split('.')
            except:
                peft_type, init_type = args.lora.split('.')
                init_cof = 1 / args.experts
            
            peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update(
                {"GOAT": peta.src.GOATModel }
            )
            lora_config = peta.src.GOATConfig(
                r=lora_r,
                use_rslora=True if "rs" in args.lora else False,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=target_modules,
                modules_to_save=['classifier'], 
                exclude_modules='classifier.*', # avoid add lora to untrained classifier head 
                # task_type="SEQ_CLS", cannot use peft seq cls task (only support NLP input_ids)
                bias="none",
                num_experts=args.experts,
                top_k=args.k,
                init_type=init_type,
                init_cof=float(init_cof),
            )
        
        if local_rank == 0:
            print('>>>', lora_config)

        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if args.lora not in ["lora", "rslora"]: 
            for name, module in model.named_modules():
                if "lora_" in name:  
                    module.to(torch.float32)
        scaling_factor = (lora_config.lora_alpha / math.sqrt(lora_config.r)) if "rs" in args.lora else (lora_config.lora_alpha / lora_config.r)
    
    if local_rank == 0:
        unique_patterns = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                if '.layer.' in n:
                    names = n.split('.layer.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                elif '.layers.' in n:
                    names = n.split('.layers.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                unique_patterns.add(n)
        print(unique_patterns)

    trainer = CustomTrainer(
        scaling_factor=scaling_factor,
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=image_processor,
        args=TrainingArguments(
            output_dir=args.output, 
            logging_dir="./transformers_logs",
            do_train=True,
            num_train_epochs=args.ep,
            per_device_train_batch_size=args.bz,
            gradient_accumulation_steps=args.gacc,
            optim="adamw_torch",
            logging_steps=1,
            bf16=True,
            learning_rate=args.lr,
            weight_decay=0, # No weight decay
            warmup_ratio=0.03, # warmup step override it 
            warmup_steps=args.warmup,
            lr_scheduler_type=args.sched,
            report_to="wandb" if rank == 0 else None, 
            label_names=["labels"],  
            # ddp_find_unused_parameters=False if 'adamole' not in args.lora else True,
            ddp_find_unused_parameters=True, # clip for image classification donnot use the vision_model.post_layernorm, cause no grad
            evaluation_strategy='no',
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_eval_batch_size=1,
            eval_steps=-1,
            save_strategy="no",
            save_steps=-1,
            save_total_limit=100,
            deepspeed="./config/deepspeed_zero2.json" if world_size > 1 and 'adamole' not in args.lora else None, 
         ),
        # data_collator=default_data_collator,
        data_collator=collate_fn,
        cus_args = args,
    )
    trainer.add_callback(CustomCallback(trainer,test_dataset,args))
    trainer.train(resume_from_checkpoint=args.resume)
    if rank == 0:
        print(f'saved in {args.output}')
        wandb.finish()

if __name__ == "__main__":
    main()
    