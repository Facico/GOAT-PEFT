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
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    TextGenerationPipeline,
    GenerationConfig,
    TrainerCallback,
)

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
import importlib
import peft
from peft import LoraConfig, PeftModel, get_peft_model
from packaging import version
from transformers.training_args import TrainingArguments
from peta.utils import TitledLog
import torch.distributed as dist
import wandb
import tqdm 
import math
import re
import peta
log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

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
    # subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    subset = dataset.select(range(start_index, end_index))
    return subset

def gather_from_all_processes(data):
    gathered_data = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_data, data)
    # Flatten the list of lists
    return [item for sublist in gathered_data for item in sublist]

def _is_peft_model(model):
    classes_to_check = (PeftModel,)
    # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
        from peft import PeftMixedModel
        classes_to_check = (*classes_to_check, PeftMixedModel)
    return isinstance(model, classes_to_check)

class CustomCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset, args, **kwargs):
        super().__init__()
        self.trainer = trainer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            padding_side="left", # important
        )
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.args = args
        self.local_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.test_bz = 8

        if self.args.task == 'metamathqa100k':

            self.test_dataset = peta.tasks.load_gsm8k_eval(self.tokenizer, max_cnt=os.getenv('MAX_TEST_SIZE'))

        elif self.args.task == 'codefeedback100k':

            self.test_dataset = peta.tasks.load_human_eval(self.tokenizer, max_cnt=os.getenv('MAX_TEST_SIZE'))

        elif self.args.task == 'wizardlm52k':
            return
            # self.test_dataset = peta.tasks.load_alpaca_eval()
        elif self.args.task == "commonsense170k":
            return
        elif self.args.task in ["arc_c", "arc_e", "openbookqa", "allenai/winogrande", "boolq", "piqa", "allenai/social_i_qa", "Rowan/hellaswag"]:

            self.test_dataset = test_dataset
        else:
            raise NotImplementedError(f"Unsupported task: {self.args.task}")

        # if self.local_rank == 0:
        #     print('>>>', tokenizer.decode(self.test_dataset[0]['input_ids']))
        #     # CustomDataCollatorWithPadding(self.tokenizer)([self.test_dataset[0],self.test_dataset[1]])

        if self.world_size > 1:
            self.test_dataset = split_dataset(self.test_dataset, self.local_rank, self.world_size)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if os.getenv('DEBUG', '0') != '0':
            return self.on_epoch_end(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        predictions, references = [], []

        if self.args.task == 'wizardlm52k':
            # offline eval
            return

        if self.args.task == 'codefeedback100k':
            predictions = peta.tasks.infer_humaneval(
                self.test_dataset,
                self.test_bz,
                self.tokenizer,
                self.trainer.model,
            )
            all_predictions = gather_from_all_processes(predictions)
            if self.local_rank == 0:
                import human_eval.data
                os.makedirs(self.args.result, exist_ok=True)
                sample_file=f"{self.args.result}/humaneval_samples_{self.args.prj.replace('/', '')}_{self.args.seed}.jsonl"
                human_eval.data.write_jsonl(sample_file, all_predictions)

                from human_eval.evaluation import evaluate_functional_correctness
                # only eval PASS@1
                correct_rate = evaluate_functional_correctness(sample_file, k=[1])
                wandb.log({f'test/{self.args.task}-pass@1': correct_rate['pass@1']}, step=self.trainer._globalstep_last_logged)
                save_csv({
                    'task': self.args.task,
                    'model': f"{self.args.prj}/{self.trainer._globalstep_last_logged}",
                    'pass@1': round(correct_rate['pass@1'] * 100, 2),
                }, self.args.result)

            return 

        if self.args.task == 'metamathqa100k':
            predictions, references = peta.tasks.infer_gsm8k(
                self.test_dataset,
                self.test_bz,
                self.tokenizer,
                self.trainer.model,
            )
        elif self.args.task == "commonsense170k":
            # offline eval
            return
        elif self.args.task in ["arc_c", "arc_e", "openbookqa", "allenai/winogrande", "boolq", "piqa", "allenai/social_i_qa", "Rowan/hellaswag"]:
            predictions, references = peta.tasks.infer_commonsense(
                self.test_dataset,
                self.test_bz,
                self.tokenizer,
                self.trainer.model,
            )

        predictions = gather_from_all_processes(predictions)
        references = gather_from_all_processes(references)

        if self.local_rank == 0:
            correct_rate = sum(p == q for p, q in zip(predictions, references)) / len(predictions)
            wandb.log({f'test/{self.args.task}-acc': correct_rate}, step=self.trainer._globalstep_last_logged)
            save_csv({
                'task': self.args.task,
                'model': f"{self.args.prj}/{self.trainer._globalstep_last_logged}",
                'acc': round(correct_rate * 100, 2),
            }, self.args.result)

class CustomTrainer(Trainer):
    def __init__(self, *args, cus_args=None, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cus_args = cus_args
        if 'ft' not in cus_args.lora:
            self.scaling_factor = scaling_factor
        self.aux_loss_coeff = self.cus_args.aux_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

            if hasattr(unwrapped_model, 'get_aux_loss'):
                aux_loss = unwrapped_model.get_aux_loss()
                loss += self.aux_loss_coeff * aux_loss
            
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            unwrapped_model = self.accelerator.unwrap_model(model)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
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
    
    path =  args.model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(path,) #  padding_side="left"
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash_attn is deterministic under this version by setting env vars; view transformers/modeling_flash_attention_utils.py:L237
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    model = transformers.LlamaForCausalLM.from_pretrained(
        path, max_length=1024, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        device_map={"": local_rank}
    )

    scaling_factor = 2
    if 'ft' != args.lora:
        lora_r = args.rank
        lora_alpha = args.alpha
        lora_module_maps = {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
            "u": "up_proj",
            "d": "down_proj",
            "g": "gate_proj",
        }
        target_modules = [lora_module_maps[char] for char in args.modules]
        
        if 'src' in args.lora:

            import peta.src
            import peft.peft_model
            peft.peft_model.get_peft_model_state_dict = peta.src.utils.save_and_load.get_peft_model_state_dict
            peft.peft_model.set_peft_model_state_dict = peta.src.utils.save_and_load.set_peft_model_state_dict
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
                task_type="CAUSAL_LM",
                bias="none",
                num_experts=args.experts,
                top_k=args.k,
                init_type=init_type,
                init_cof=float(init_cof),
            )
        if local_rank == 0:
            print('>>>', lora_config)
        model = get_peft_model(model, lora_config)
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
    
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        import peta
        # NOTE: you may need to rm -rf ./data_cache
        if args.task == 'metamathqa100k':
            datasets, preprocessor = peta.tasks.load_meta_math(), peta.tasks.MetaMathQA100k_Preprocessor(tokenizer=tokenizer)
            test_dataset = None
        elif args.task == 'codefeedback100k':
            datasets, preprocessor = peta.tasks.load_codefeedback(), peta.tasks.CodeFeedback100k_Preprocessor(tokenizer=tokenizer)
            test_dataset = None
        elif args.task == 'wizardlm52k':
            datasets, preprocessor = peta.tasks.load_wizardlm(), peta.tasks.WizardLM52k_Preprocessor(tokenizer)
            test_dataset = None
        elif args.task in ["commonsense170k", "arc_c", "arc_e", "openbookqa", "allenai/winogrande", "boolq", "piqa", "allenai/social_i_qa", "Rowan/hellaswag"]:
            from peta.tasks.data import get_formatted_datasets
            max_length = 256
            tokenizer = transformers.LlamaTokenizer.from_pretrained(path, padding_side="left", add_eos_token=True)
            datasets = get_formatted_datasets(data_path=args.task, prompt_only=True)
            test_dataset = datasets.pop('validation')
            if local_rank == 0:
                print('>>>', datasets)
            tokenizer.pad_token = tokenizer.eos_token

            def preprocessor(examples):
                tokenized_input = tokenizer(
                    [i.strip() for i in examples["text"]],
                    truncation=True,
                    max_length=max_length,
                )
                len_input = [len(i) - 1 for i in tokenized_input["input_ids"]]
                full_text = [text.strip() + ans.strip() for text, ans in zip(examples["text"], examples["answer"])]
                tokenized_full = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                return {
                    "input_ids": tokenized_full["input_ids"], 
                    # must call label or labels, and you can only have either
                    "labels": [[-100] * len_input[i] + tokenized_full["input_ids"][i][len_input[i]:] for i in range(len(len_input))], 
                    "attention_mask": tokenized_full["attention_mask"]
                }
        else: 
            raise NotImplementedError(f"Task {args.task} not implemented")
            # tokenize_text(datasets['train'][0:1]) for debug

        datasets = datasets.map(
            preprocessor,
            batched=True,
            batch_size=20,
            num_proc=10,
            desc="Running tokenizer on dataset",
            remove_columns=datasets["train"].column_names, # remove multiple label (will error in collator)
        )
        if local_rank == 0:
            print('>>>', tokenizer.decode(datasets['train'][0]['input_ids']))
            print('>>>', tokenizer.decode([l for l in datasets['train'][0]['labels'] if l != -100]))
    
    trainer = CustomTrainer(
        scaling_factor=scaling_factor,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
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
            ddp_find_unused_parameters=False if 'adamole' not in args.lora else True,
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_eval_batch_size=1,
            evaluation_strategy="no",
            eval_steps=-1,
            save_strategy="no" if 'wizardlm52k' != args.task and 'commonsense170k' != args.task and'ft' not in args.lora else 'epoch',
            save_steps=-1,
            save_total_limit=100,
            # deepspeed="./config/deepspeed_zero2.json" if world_size > 1  and 'ft' not in args.lora and 'adamole' not in args.lora else None, 
        ),
        # data_collator=default_data_collator,
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
        cus_args = args,
    )
    if 'wizardlm52k' != args.task and 'commonsense170k' != args.task and 'ft' not in args.lora:
        trainer.add_callback(CustomCallback(trainer, test_dataset , args))
    trainer.train(resume_from_checkpoint=args.resume)
    if rank == 0:
        print(f'saved in {args.output}')
        wandb.finish()
            

if __name__ == "__main__":
    main()









































