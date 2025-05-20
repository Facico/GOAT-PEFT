from datasets import load_dataset, Dataset, DatasetDict
import typing as tp
import functools
import os
import pickle
import random
import re
from .datasets_preprocess import DatasetPreprocessor, preprocess
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import tqdm
from transformers import GenerationConfig

def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper_cache

    return decorator_cache

def make_file_friendly_str(value) -> str:
    """Converts a value to a string that is safe for use in file names."""
    if isinstance(value, (list, tuple)):
        return '_'.join(make_file_friendly_str(v) for v in value)
    elif isinstance(value, dict):
        return '_'.join(f"{make_file_friendly_str(k)}-{make_file_friendly_str(v)}" for k, v in value.items())
    else:
        return str(value).replace(os.sep, '_').replace(':', '_')

def cache_to_disk2(root_datadir: str):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            # Create a unique filename based on the function name and its arguments
            func_name = func.__name__.replace("/", "")
            args_str = '_'.join(make_file_friendly_str(arg)[:5] for arg in args)
            kwargs_str = '_'.join(f"{k}-{make_file_friendly_str(v)}" for k, v in sorted(kwargs.items()))
            parameters_str = f"{args_str}_{kwargs_str}".strip('_')
            cache_file = os.path.join(root_datadir, f"{func_name}_{parameters_str}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper_cache

    return decorator_cache


class WizardLM52k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["instruction"], str):
            raise NotImplementedError
    
        else:
            combined_text = [
                x + " " + y + self.tokenizer.eos_token 
                for (x, y) in zip(example["instruction"], example["output"])
            ]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
            input_text_length = [
                len(self.tokenizer(example["instruction"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["instruction"]))
            ]
            labels = encodings["input_ids"].clone()
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''
        
@cache_to_disk2("data_cache")
def load_wizardlm(max_tokens=1024, max_cnt=52000):
        
    dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    def preprocess(data):
        y = data['output']
        return {
            "instruction": template_wo_input.format(
                instruction=data['instruction']
            ),
            "output": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=max_cnt)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp['output'].lower() or "as an ai" in temp['output'].lower():
            continue
        if len(tokenizer(temp['instruction']+' '+temp['output'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < max_cnt:
            train_samples.append(processed_sample)
        elif count >= max_cnt:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    datasets = DatasetDict({
        "train": train_samples,
    })
    
    return datasets

@cache_to_disk("data_cache")
def load_alpaca_eval(tokenizer,):

    dataset = load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval")["eval"]

    def alpaca_preprocess(instruction, input, y):

        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_wo_input.format(instruction=instruction + ' ' + input)
        
        inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=768)
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": y,
        }
    
    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"]),
        remove_columns=dataset.column_names,
    )
    return dataset


class MetaMathQA100k_Preprocessor(DatasetPreprocessor):
    # [TODO]

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["x"], str):
            # not batched
#             input_text, target_text = self.preprocess(
#                 example["instruction"], example["output"]
#             )
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            labels = encodings["input_ids"].clone()
            if self.tokenizer.padding_side == 'right':
                for i, l in enumerate(input_text_length):
                    labels[i, :l] = -100
                labels[encodings["attention_mask"] == 0] = -100
            else:
                for i, l in enumerate(input_text_length):
                    # Calculate the length of the padding on the left side
                    padding_length = (encodings["input_ids"].size(1) - l - len(self.tokenizer(example["y"][i], return_tensors="pt")["input_ids"][0]))
                    # Set the padding and input text part to -100
                    labels[i, :padding_length + l] = -100
                labels[encodings["attention_mask"] == 0] = -100
            # why: labels[encodings["attention_mask"] == self.tokenizer.pad_token_id] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
#                 "attention_mask": encodings["input_ids"].ne(self.tokenizer.pad_token_id),
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }
            assert len(results["input_ids"]) == len(results["labels"]) == len(results["attention_mask"])
            return results

@cache_to_disk2("data_cache")
def load_meta_math(max_tokens=512, max_cnt=100000):
    
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
#     def preprocess(data):
#         return {
#             "x": f'Q: {data["query"]}\nA: ',
#             "y": data["response"].split("\nThe answer is:")[0]
#         }
    def preprocess(data):
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": data["response"]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=max_cnt)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens or "GSM" not in sample["type"]:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < max_cnt:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        # elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
        #     eval_samples.append(processed_sample)
        elif count >= max_cnt:  # Stop processing after collecting enough samples
            break
        count += 1

    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    datasets = DatasetDict({
        "train": train_samples,
    })
    
    return datasets

ALPACA_PREFIX_CODE_TEMPLATE = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""


ALPACA_PREFIX_MATH_TEMPLATE = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

def extract_gsm_num(text):
    # Regex pattern to find the number following '####'

    # cannot extract negative number
    # pattern = r'####\s*(\d+)'
    pattern = r'####\s*(-?\d+(?:,\d+)*)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        # replace numbers like `x,xxx` with `xxxx`
        # only have int value
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

@cache_to_disk2("data_cache")
def load_gsm8k_eval(tokenizer, max_cnt=None):
    dataset = load_dataset("openai/gsm8k", 'main', split="test")
    if max_cnt is not None:
        indices = random.sample(range(len(dataset)), max_cnt)
        dataset = dataset.select(indices)

    def _preprocess_gsm8k(examples):
        x = template_wo_input.format(instruction=examples['question'])
        y = extract_gsm_num(examples['answer'])
        inputs = tokenizer(x, return_tensors="pt", max_length=768, truncation=True,)
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": y,
        }

    test_dataset = dataset.map(
        _preprocess_gsm8k, 
        remove_columns=dataset.column_names,
        num_proc=1,
        # Notice inner has tensor,with use gpu (because have been dist.init), 
        # so if use process number more than world size, will cause spawn error
    )
    return test_dataset

@dataclass
class CustomDataCollatorWithPadding:
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract labels and remove them from features before passing to tokenizer's padding method
        labels = None
        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]
        
        # Use the tokenizer's padding method to handle input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Add back the labels
        if labels is not None:
            batch["labels"] = labels 
        return batch

@torch.inference_mode()
def infer_gsm8k(test_dataset, test_bz, tokenizer, model, ):

    predictions, references = [], []
    dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_bz, 
        collate_fn=CustomDataCollatorWithPadding(tokenizer, padding=True)
    )
    for batch in tqdm.tqdm(dataloader, desc='eval') if os.getenv('LOCAL_RANK', '0') == '0' else dataloader:
        outputs = model.generate(
            input_ids=batch["input_ids"].to('cuda'), 
            attention_mask=batch["attention_mask"].to('cuda'), # ！important
            max_new_tokens=512, 
            eos_token_id=tokenizer.eos_token_id, 
            top_p=0.95, 
            temperature=0.8,
        )
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([extract_gsm_num(p) for p in pred])
        references.extend(batch["labels"])
    return predictions, references

@torch.inference_mode()
def infer_humaneval(test_dataset, test_bz, tokenizer, model, ):

    def _post_process(text):
        text = text.replace("```", "").replace("\t", "    ")
        text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
        lines = [ll.rstrip() for ll in text.splitlines() if ll.strip()]
        spaces = [len(re.match(r'^( *)', line).group(1)) for line in lines]
        try:
            def_line = next(i for i, line in enumerate(lines) if "def" in line)
            def_space = spaces[def_line]
        except StopIteration:
            def_space = 0
        indentation = {space: (0 if space <= def_space else i + 1) for i, space in enumerate(sorted(set(spaces)))}
        return "\n".join(["    " * indentation[space] + line.lstrip() for line, space in zip(lines, spaces)])

    all_predictions = []
    dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_bz, 
        collate_fn=CustomDataCollatorWithPadding(tokenizer, padding=True)
    )
    for batch in tqdm.tqdm(dataloader, desc='eval') if os.getenv('LOCAL_RANK', '0') == '0' else dataloader:
        outputs = model.generate(
            input_ids=batch["input_ids"].to('cuda'), 
            attention_mask=batch["attention_mask"].to('cuda'), # ！important
            max_new_tokens=512, 
            eos_token_id=tokenizer.eos_token_id, 
            top_p=0.95, 
            temperature=0.8,
        )
        # split base on ### Response: ?
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred = [o.split("### Response:\n```python\n")[-1].strip() for o in pred]
        all_predictions.extend([
            dict(task_id=f"HumanEval/{task_id}", completion=_post_process(pred_text))
            for task_id, pred_text in zip(batch["task_ids"], pred)
        ])
    return all_predictions
    
@torch.inference_mode()
def infer_commonsense(test_dataset, test_bz, tokenizer, model, em=False):

    def _predict_choices(examples):
        inputs = tokenizer(examples["text"], truncation=True, return_tensors="pt", padding=True).to('cuda')
        outputs = model(**inputs)
        predictions = outputs.logits[:, -1, :]
        choices = [chr(ord('A') + i) for i in range(max(examples['num_choices']))]
        choice_ids = [tokenizer.encode(choice, add_special_tokens=False)[-1] for choice in choices]
        predicted_ids = torch.argmax(predictions[:, choice_ids], dim=-1)
        return {'prediction': [choices[predicted_id] for predicted_id in predicted_ids.cpu().numpy()]}
    def _predict_commonsense(examples):
        prompts = examples['text']
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        
        # Generation parameters
        inputs["max_new_tokens"] = 32
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )
        inputs["generation_config"] = generation_config
        inputs["return_dict_in_generate"] = True
        inputs["output_scores"] = True
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        s = outputs.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        predictions = [o.split("### Response:")[-1].strip() for o in outputs]
        return {'prediction': predictions}

    with torch.inference_mode():
        if not em:
            predictions = test_dataset.map(
                lambda x: _predict_commonsense(x), 
                batched=True, 
                batch_size=test_bz
            )['prediction']
        else:
            predictions = test_dataset.map(
                lambda x: _predict_choices(x), 
                batched=True, 
                batch_size=test_bz
            )['prediction']
        references = test_dataset['answer']
    return predictions, references


@cache_to_disk2("data_cache")
def load_human_eval(tokenizer, max_cnt=None):

    import human_eval.data

    def _preprocess_human_eval(examples):

        task_ids = int(examples["task_id"].split("/")[-1])
        input_texts = ALPACA_PREFIX_CODE_TEMPLATE.format(PROMPT=examples["prompt"]) + " "
        encodings = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "task_ids": task_ids,
        }

    dataset = Dataset.from_list(list(human_eval.data.read_problems().values()))
    if max_cnt is not None:
        indices = random.sample(range(len(dataset)), max_cnt)
        dataset = dataset.select(indices)

    test_dataset = dataset.map(
        _preprocess_human_eval, 
        num_proc=1,
        remove_columns=dataset.column_names,
    )
    return test_dataset

class CodeFeedback100k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["x"], str):
            # not batched
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

@cache_to_disk2("data_cache")
def load_codefeedback(max_tokens=1024,max_cnt=100000):
    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    def preprocess(data):
        y = data['answer']
        y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=max_cnt)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < max_cnt:
            train_samples.append(processed_sample)
        elif count >= max_cnt:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    datasets = DatasetDict({
        "train": train_samples,
    })
    
    return datasets
        
#     # convert to hf dataset
#     train_set = Dataset.from_list(train_samples)
#     eval_set = Dataset.from_list(eval_samples)
#     return train_set, eval_set, eval_set





























































