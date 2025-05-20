import argparse
import os
import re
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TextGenerationPipeline,
    GenerationConfig,
)
from peta.tasks.data import get_formatted_datasets, extract_cs_answer
from accelerate import Accelerator, DistributedDataParallelKwargs
import accelerate
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import StoppingCriteriaList
import json
import pandas as pd
import math
import logging
from pathlib import Path
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict, load_dataset
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
import importlib
import peft
from peft import LoraConfig, PeftModel
from packaging import version
import peta
from peta.utils import TitledLog
import wandb
import torch.distributed as dist

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

class QA:


    @staticmethod
    def predict_choices(tokenizer, model, examples):
        prompts = examples['text']
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, -1, :]
        choices = [chr(ord('A') + i) for i in range(max(examples['num_choices']))]
        choice_ids = [tokenizer.encode(choice, add_special_tokens=False)[-1] for choice in choices]

        predicted_ids = torch.argmax(logits[:, choice_ids], dim=-1)
        predictions = [choices[predicted_id] for predicted_id in predicted_ids.cpu().numpy()]
        return {
            'prediction': predictions
        }

    @staticmethod
    def compute_accuracy(predictions, references):
        assert len(predictions) > 0
        correct = sum(pred == ref for pred, ref in zip(predictions, references))
        return correct / len(predictions) 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', default="lora-pro", type=str)
    parser.add_argument('--task', default="math", type=str)
    parser.add_argument('--output', default="output", type=str)
    parser.add_argument('--result', default="output", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bz', default=1, type=int)
    args = parser.parse_args()
    # assert args.lora in ["lora-pro", "rslora-pro"]
    return args    


def test():

    args = get_arguments()
    transformers.set_seed(args.seed)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    
    with torch.inference_mode():
        if 'lora' in args.output:
            import peta.src
            import peft.peft_model
            peft.peft_model.get_peft_model_state_dict = peta.src.utils.save_and_load.get_peft_model_state_dict
            peft.peft_model.set_peft_model_state_dict = peta.src.utils.save_and_load.set_peft_model_state_dict
            peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update(
                    {"GOAT": peta.src.GOATConfig }
                )
            peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update(
                    {"GOAT": peta.src.GOATModel }
                )
            model = transformers.LlamaForCausalLM.from_pretrained(
                'meta-llama/Llama-2-7b-hf',
                max_length=1024,
                torch_dtype=torch.bfloat16,
                device_map={"": local_rank}
            )
            lora_config = LoraConfig.from_pretrained(args.output)
            model = PeftModel.from_pretrained(model, args.output, config=lora_config)
            # model = model.merge_and_unload(progressbar=True)

        else:
            model = transformers.LlamaForCausalLM.from_pretrained(
                args.output,
                max_length=1024,
                torch_dtype=torch.bfloat16,
                device_map={"": local_rank}
            )
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.task == 'metamathqa100k':

        dataset = peta.tasks.load_gsm8k_eval(tokenizer, max_cnt=os.getenv('MAX_TEST_SIZE'))

    elif args.task == 'codefeedback100k':

        dataset = peta.tasks.load_human_eval(tokenizer, max_cnt=os.getenv('MAX_TEST_SIZE'))

    elif args.task == 'wizardlm52k':

        dataset = peta.tasks.load_alpaca_eval()
    elif args.task.startswith("commonsense170k-"):
        dataset =  get_formatted_datasets(args.task, prompt_only=True)['validation']
    elif args.task in ["arc_c", "arc_e", "openbookqa", "allenai/winogrande", "boolq", "piqa", "allenai/social_i_qa", "Rowan/hellaswag"]:

        dataset =  get_formatted_datasets(args.task, prompt_only=True)['validation']
    
    else:

        raise NotImplementedError(f"Unsupported task: {args.task}")

    if world_size > 1:
        dataset = split_dataset(dataset, local_rank, world_size)

    if args.task == 'codefeedback100k':

        predictions = peta.tasks.infer_humaneval(
            dataset,
            args.bz,
            tokenizer,
            model,
        )
        all_predictions = gather_from_all_processes(predictions)
        if local_rank == 0:
            import human_eval
            os.makedirs(args.result, exist_ok=True)
            sample_file=f"{args.result}/humaneval_samples_{args.prj.replace('/', '')}.jsonl"
            human_eval.data.write_jsonl(sample_file, all_predictions)

            from human_eval.evaluation import evaluate_functional_correctness
            # only eval PASS@1
            correct_rate = evaluate_functional_correctness(sample_file, k=[1])
            ans = {
                'task': args.task,
                'model': args.output,
                'acc':  round(correct_rate['pass@1'] * 100, 2),
            }
            save_csv(ans, args.result)

        return 

    if args.task == 'wizardlm52k':

        predictions = peta.tasks.infer_alpacaeval(
            dataset,
            args.bz,
            tokenizer,
            model,
        )
        all_predictions = gather_from_all_processes(predictions)
        # ...
        return

    if args.task == 'metamathqa100k':

        predictions, references = peta.tasks.infer_gsm8k(
            dataset,
            args.bz,
            tokenizer,
            model,
        )
    elif args.task.startswith("commonsense170k-"):
        predictions, references = peta.tasks.infer_commonsense(
            dataset,
            args.bz,
            tokenizer,
            model,
            em=True,
        )
    elif args.task in ["arc_c", "arc_e", "openbookqa", "allenai/winogrande", "boolq", "piqa", "allenai/social_i_qa", "Rowan/hellaswag"]:

        predictions, references = peta.tasks.infer_commonsense(
            dataset,
            args.bz,
            tokenizer,
            model,
            em=False,
        )
    
    else:

        raise NotImplementedError(f"Unsupported task: {args.task}")

    predictions = gather_from_all_processes(predictions)
    references = gather_from_all_processes(references)
    
    if local_rank == 0:
        if args.task.startswith("commonsense170k-"):
            sub_data_name = args.task.split("-")[-1]
            sub_name_key = {
                "arc_c": "ARC-Challenge",
                "arc_e": "ARC-Easy",
                "siqa": "social_i_qa",
                "obqa": "openbookqa"
            }
            if sub_data_name in sub_name_key.keys():
                sub_data_name = sub_name_key[sub_data_name]
            predictions = [extract_cs_answer(sub_data_name, o) for o in predictions]   
        correct_rate = sum(p == q for p, q in zip(predictions, references)) / len(predictions)
        ans = {
            'task': args.task,
            'model': args.output,
            'acc':  round(correct_rate * 100, 2),
        }
        save_csv(ans, args.result)

    
if __name__ == "__main__":

    test()