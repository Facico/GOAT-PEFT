"""
Loading and Preprocessing Datasets
"""
import os

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import typing as tp
import functools
import os
import pickle
import logging
import re
log = logging.getLogger(__name__)

def extract_cs_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
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
                    log.info(f"Loading cached data for {func.__name__}")
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
                log.info(f"Cached data for {func.__name__}")
            return result

        return wrapper_cache

    return decorator_cache

@cache_to_disk("data_cache")
def load_rte(mlm=False):
    dataset = load_dataset("glue", "rte")
    instruction = "determine if the second sentence entails the first sentence: "
    label_map = {0: "entailment", 1: "not entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]] if not mlm else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_stsb(mlm=False):
    dataset = load_dataset("glue", "stsb")
    instruction = "compute the semantic similarity between the two sentences: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": e["label"] if not mlm else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_emo():
    dataset = load_dataset("emo")
    label_map = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    instruction = "classify the emotion of the text: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["text"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    test_set = dataset["test"]
    return train_set, test_set, test_set

@cache_to_disk("data_cache")
def load_sst2(mlm=False):
    dataset = load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_cola(mlm=False):
    dataset = load_dataset("glue", "cola")
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_qqp(mlm=False):
    dataset = load_dataset("glue", "qqp")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "duplicate", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question1"]}\n{e["question2"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_mrpc(mlm=False):
    dataset = load_dataset("glue", "mrpc")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_mnli(mlm=False):
    dataset = load_dataset("glue", "mnli")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation_matched"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_squad(mlm=False):
    dataset = load_dataset("rajpurkar/squad")
    instruction = "answer the question: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\ncontext: {e["context"]}\nresult: ',
            "y": ", ".join(e["answers"]["text"]) if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_qnli(mlm=False):
    dataset = load_dataset("glue", "qnli")
    instruction = "classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]] if mlm==False else e["label"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    return train_set, validation_set, test_set


template_with_input = '''### Instruction:
{instruction}

### Input:
{input}

### Response:
'''

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

@cache_to_disk("data_cache")
def load_alpaca():
    dataset = load_dataset("tatsu-lab/alpaca")
    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}
    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)['train']
    validation_set = dataset["train"].train_test_split(test_size=0.1)['test']
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    #x = "Q: " + x[0] + "\n" + "A:"
    dataset = dataset.map(
        lambda e: {
            "x": f'Question: {e["question"]}\nAnswer: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_alpaca_gpt4():
    dataset = load_dataset("tatsu-lab/alpaca")
    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}
    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)['train']
    validation_set = dataset["train"].train_test_split(test_size=0.1)['test']
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_flan():
    dataset = load_dataset("Muennighoff/flan", split='train', streaming=True)
    def preprocess(data):
        return {
            "x": template_wo_input.format(instruction=data['inputs']),
            "y": data['targets'],
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(buffer_size=5000, seed=42)
    from tqdm import tqdm
    for sample in tqdm(dataset, total=110000):
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_meta_math(max_tokens=512):
    dataset = load_dataset("meta-math/MetaMathQA", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        return {
            "x": f'Question: {data["query"]}\nAnswer: ',
            "y": data["response"].split("\nThe answer is:")[0]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
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
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_flan_v2(max_tokens=512):
    dataset = load_dataset("SirNeural/flan_v2", split='train', streaming=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        return {
            "x": data['inputs'],
            "y": data['targets'],
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(buffer_size=5000, seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=1024):
    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
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
    bar = tqdm(dataset, total=110000)
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
        if count < 100000:
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_wizardlm(max_tokens=1024):
    dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        y = data['output']
        return {
            "x": template_wo_input.format(
                instruction=data['instruction']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=70000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp['y'].lower() or "as an ai" in temp['y'].lower():
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < 52000:
            train_samples.append(processed_sample)
        elif 52000 <= count < 70000:
            eval_samples.append(processed_sample)
        elif count >= 70000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set
@cache_to_disk("data_cache")
def load_commonsense170k():
    data = load_dataset("json", data_files="../dataset/commonsense_170k.json")
    dataset = data["train"].train_test_split(test_size=120, shuffle=True, seed=42)
    def preprocess(data):
        if data["input"]:
            return {"x": f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                        ### Instruction:
                        {data["instruction"]}
                        ### Input:
                        {data["input"]}
                        ### Response:""",
                    "y": data["output"]
                    }
        else:
            return {"x": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                    ### Instruction:
                    {data["instruction"]}
                    ### Response:""",
                    "y": data["output"]}
    dataset = dataset.map(preprocess)
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set

def load_commonsense170k_test(sub_data_name):
    sub_name2path = {
        "arc_c": "ARC-Challenge",
        "arc_e": "ARC-Easy",
        "siqa": "social_i_qa",
        "obqa": "openbookqa"
    }
    if sub_data_name in sub_name2path.keys():
        sub_data_name_path = sub_name2path[sub_data_name]
    else:
        sub_data_name_path = sub_data_name.lower()
    dataset = load_dataset("json", data_files=f"../dataset/{sub_data_name_path}/test.json")
    def preprocess(data):
        if data["input"]:
            return {"x": f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                        ### Instruction:
                        {data["instruction"]}
                        ### Input:
                        {data["input"]}
                        ### Response:""",
                    "y": data["output"]
                    }
        else:
            return {"x": f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                    ### Instruction:
                    {data["instruction"]}
                    ### Response:""",
                    "y": data["output"]}
    dataset = dataset.map(preprocess)
    train_set = dataset["train"]
    return train_set, train_set, train_set

def load_glue_mlm(sub_data_name):
    train_set, validation_set, _ = eval(f"load_{sub_data_name}")(mlm=True)
    def preprocess(examples):
        return {"text": examples["x"], "labels": examples["y"]}
    train_set = train_set.map(preprocess)
    validation_set = validation_set.map(preprocess)
    # train_set = dataset["train"]
    return train_set, validation_set, validation_set

DATASET_MAP = {
    "glue-mlm": load_glue_mlm,
    "sst2": load_sst2,
    "cola": load_cola,
    "qqp": load_qqp,
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "emo": load_emo,
    "squad": load_squad,
    "alpaca": load_alpaca,
    "qnli": load_qnli,
    "gsm8k": load_gsm8k,
    "commonsense170k": load_commonsense170k,
    "commonsense170k-test": load_commonsense170k_test,
    "alpaca_gpt4": load_alpaca_gpt4,
    "flan": load_flan,
    "flan_v2": load_flan_v2,
    "meta_math": load_meta_math,
    "codefeedback": load_codefeedback,
    "wizard_lm": load_wizardlm,
}

def format_text(example, data_name: str, prompt_only: bool = True):
    """
    Format an example into one text
    """
    if data_name == 'boolq':
        """
        Passage: Windows Movie Maker -- Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) 
        is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and 
        offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, 
        and Flickr.
        Question: is windows movie maker part of windows essentials
        Choices:
        A. No
        B. Yes
        Answer: B
        """
        text = f"Passage: {example['passage']}\nQuestion: {example['question']}\nChoices:\n"
        text += "A. No\nB. Yes\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'cb':
        """
        Text: It was a complex language. Not written down but handed down. One might say it was peeled down.
        Hypothesis: the language was peeled down
        Question: Does the text entail the hypothesis, contradict it, or is it neutral?
        Choices:
        A. Entailment
        B. Contradiction
        C. Neutral
        Answer: A
        """
        text = f"Text: {example['premise']}\nHypothesis: {example['hypothesis']}\n" \
               f"Question: Does the text entail the hypothesis, contradict it, or is it neutral?\nChoices:\n"
        text += "A. Entailment\nB. Contradiction\nC. Neutral\n"
        text += "Answer: "
        example['answer'] = ['A', 'B', 'C'][example['label']]
        example['num_choices'] = 3

    elif data_name == 'copa':
        """
        Premise: My body cast a shadow over the grass.
        Question: What’s the cause for this?
        Choices:
        A. The sun was rising.
        B. The grass was cut.
        Answer: A
        """
        text = f"Premise: {example['premise']}\nQuestion: What’s the {example['question']} for this?\nChoices:\n"
        text += f"A. {example['choice1']}\nB. {example['choice2']}\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'multirc':
        """
        Paragraph: While this process moved along, diplomacy continued its rounds. Direct pressure on the Taliban had 
        proved unsuccessful. As one NSC staff note put it, "Under the Taliban, Afghanistan is not so much a state 
        sponsor of terrorism as it is a state sponsored by terrorists." ...
        Question: What did the high-level effort to persuade Pakistan include?
        Candidate Answer: Children, Gerd, or Dorian Popa
        Choices:
        A. False
        B. True
        Answer: A
        """
        text = f"Paragraph: {example['paragraph']}\nQuestion: {example['question']}\n" \
               f"Candidate Answer: {example['answer']}\nChoices:\n"
        text += f"A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'record':
        raise NotImplementedError

    elif data_name == 'rte':
        """
        Text: No Weapons of Mass Destruction Found in Iraq Yet.
        Hypothesis: Weapons of Mass Destruction Found in Iraq.
        Question: Does the text entail the hypothesis or not?
        Choices:
        A. Entailment
        B. Not entailment
        Answer: B
        """
        text = f"Text: {example['premise']}\nHypothesis: {example['hypothesis']}\n" \
               f"Question: Does the text entail the hypothesis or not?\nChoices:\n"
        text += "A. Entailment\nB. Not entailment\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'wic':
        """
        Context 1: Do you want to come over to my <place> later?
        Context 2: A political system with no <place> for the less prominent groups.
        Question: Is the word in brackets used with the same meaning in both contexts?
        Choices:
        A. False
        B. True
        Answer: A
        """
        sentence1 = example['sentence1']
        sentence2 = example['sentence2']
        marked_sentence1 = sentence1[:example['start1']] + '<' + sentence1[example['start1']:example['end1']] \
                           + '>' + sentence1[example['end1']:]
        marked_sentence2 = sentence2[:example['start2']] + '<' + sentence2[example['start2']:example['end2']] \
                           + '>' + sentence2[example['end2']:]
        text = f"Context 1: {marked_sentence1}\nContext 2: {marked_sentence2}\n" \
               f"Question: Is the word in brackets used with the same meaning in both contexts?\nChoices:\n"
        text += "A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'wsc.fixed':
        """
        Text: <Mark> told Pete many lies about himself, which Pete included in his book. <He> should have been more 
        skeptical.
        Question: Is the pronoun in brackets referring to the correct entity as intended in the context?
        Choices:
        A. False
        B. True
        Answer: A
        """
        tokens = example['text'].split()
        span1_start = example['span1_index']
        span1_end = example['span1_index'] + len(example['span1_text'].split())
        span2_start = example['span2_index']
        span2_end = example['span2_index'] + len(example['span2_text'].split())
        marked_tokens = tokens[:span1_start] + ['<' + example['span1_text'] + '>'] + tokens[span1_end:span2_start] \
                        + ['<' + example['span2_text'] + '>'] + tokens[span2_end:]
        marked_text = ' '.join(marked_tokens)
        text = f"Text: {marked_text}\n" \
               f"Question: Is the pronoun in brackets referring to the correct entity as intended in the context?\n" \
               f"Choices:\n"
        text += "A. False\nB. True\n"
        text += "Answer: "
        example['answer'] = ['A', 'B'][example['label']]
        example['num_choices'] = 2

    elif data_name == 'commonsense_qa':
        """
        Question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the 
        school had made to change?
        Choices:
        A. ignore
        B. enforce
        C. authoritarian
        D. yell at
        E. avoid
        Answer: A
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 5

    elif data_name == 'cosmos_qa':
        """
        Context: Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . 
        seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to 
        them ; I think it 's the fact that they seemed so happy themselves when they played .
        Question: In the future , will this person go to see other bands play ?
        Choices:
        A. None of the above choices .
        B. This person likes music and likes to see the show , they will see other bands play .
        C. This person only likes Good Old War and Person L , no other bands .
        D. Other Bands is not on tour and this person can not see them .
        Answer: B
        """
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answer0']}\n"
        text += f"B. {example['answer1']}\n"
        text += f"C. {example['answer2']}\n"
        text += f"D. {example['answer3']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['label'])
        example['num_choices'] = 4

    elif data_name == 'social_i_qa':
        """
        Context: Cameron decided to have a barbecue and gathered her friends together.
        Question: How would Others feel as a result?
        Choices:
        A. like attending
        B. like staying home
        C. a good friend to have
        Answer: A
        """
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nChoices:\n"
        text += f"A. {example['answerA']}\n"
        text += f"B. {example['answerB']}\n"
        text += f"C. {example['answerC']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + int(example['label']) - 1)
        example['num_choices'] = 3

    elif data_name == 'piqa':
        """
        Question: When boiling butter, when it's ready, you can
        Choices:
        A. Pour it onto a plate
        B. Pour it into a jar
        Answer: B
        """
        text = f"Question: {example['goal']}\nChoices:\n"
        text += f"A. {example['sol1']}\n"
        text += f"B. {example['sol2']}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['label'])
        example['num_choices'] = 2

    elif data_name == 'openbookqa':
        """
        Fact: the sun is the source of energy for physical cycles on Earth
        Question: The sun is responsible for
        Choices:
        A. puppies learning new tricks
        B. children growing up and getting old
        C. flowers wilting in a vase
        D. plants sprouting, blooming and wilting
        Answer: D
        """
        text = f"Fact: {example['fact1']}\nQuestion: {example['question_stem']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 4

    elif data_name == 'ai2_arc':
        """
        Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most 
        heat?
        Choices:
        A. dry palms
        B. wet palms
        C. palms covered with oil
        D. palms covered with lotion
        Answer: A
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for label, choice in zip(choices['label'], choices['text']):
            text += f"{label}. {choice}\n"
        text += "Answer: "
        example['answer'] = example['answerKey']
        example['num_choices'] = 4

    elif data_name == 'scienceqa':
        """
        Question: Which tense does the sentence use?
        Mona will print her name with care.
        Choices:
        A. present tense
        B. future tense
        C. past tense
        Answer: B
        """
        text = f"Question: {example['question']}\nChoices:\n"
        choices = example['choices']
        for index, choice in enumerate(choices):
            text += f"{chr(ord('A') + index)}. {choice}\n"
        text += "Answer: "
        example['answer'] = chr(ord('A') + example['answer'])
        example['num_choices'] = 5  # undefined
    elif data_name == 'hellaswag':
        labels_id = ["A", "B", "C", "D"]
        text = (
                "Please choose the correct ending to complete the given sentence.\n"
            )
        text += f"Sentence: {example['activity_label']}. {example['ctx']}\nChoices:\n"
        for label, choice in enumerate(example["endings"]):
            text += f"\n({labels_id[label]}) {choice}"
        text += "Answer: "
        example['answer'] = chr(ord('A') + int(example['label'])) if example['label'] != "" else ""
        example['num_choices'] = len(example["endings"])
    elif data_name == "winogrande":
        text = "Please choose the correct answer to fill in the blank to complete the given sentence.\n"
        text += f"Sentence: {example['sentence']}\nChoices:\n"
        text += f"\n(A) {example['option1']}\n(B) {example['option2']}"
        text += "\nAnswer:"
        example['answer'] = chr(ord('A') + int(example["answer"]) - 1) if example['answer'] != "" else ""
        example['num_choices'] = 2
    else:
        raise NotImplementedError

    if not prompt_only:
        text += f"{example['answer']}"
    example['data_name'] = data_name
    example['text'] = text
    return example

def format_text_2(example, data_name: str, prompt_only=True):
    if not data_name.startswith("commonsense170k-"):
        example['answer'] = example["y"]
    example['data_name'] = data_name
    example['text'] = example["x"]
    if not prompt_only:
        example['text'] += f"{example['answer']}"
    return example

def get_formatted_datasets(data_path: str, prompt_only: bool):
    """
    Get formatted datasets
    """
    data_name = os.path.basename(data_path).lower()
    # Load and format datasets
    if data_name.startswith("commonsense170k-"):
        sub_data_name = data_name.split("-")[-1]
        train_set, val_set, test_set = DATASET_MAP["commonsense170k-test"](sub_data_name)
        train_set = train_set.map(
            lambda example: format_text_2(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)
        val_set = val_set.map(
            lambda example: format_text_2(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)
        return DatasetDict({"train": train_set, "validation": val_set})
    if data_name.startswith("glue-mlm-"):
        sub_data_name = data_name.split("-")[-1]
        train_set, val_set, test_set = DATASET_MAP["glue-mlm"](sub_data_name)
        return DatasetDict({"train": train_set, "validation": val_set})
    if data_name in DATASET_MAP.keys():
        train_set, val_set, test_set = DATASET_MAP[data_name]()
        train_set = train_set.map(
            lambda example: format_text_2(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)
        val_set = val_set.map(
            lambda example: format_text_2(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)
        return DatasetDict({"train": train_set, "validation": val_set})
    if data_name == 'super_glue':
        data_names = ['boolq', 'cb', 'copa', 'rte', 'wic']
        splits = ['train', 'validation', 'test']
        formatted_datasets = {split: [] for split in splits}

        # Load and format datasets
        for _data_name in data_names:
            _datasets = load_dataset(path='super_glue', name=_data_name)
            print(f'Datasets: {_datasets}')
            _formatted_datasets = _datasets.map(
                lambda example: format_text(example, _data_name, prompt_only=prompt_only),
                batched=False, load_from_cache_file=False)
            for split in splits:
                formatted_datasets[split].append(
                    _formatted_datasets[split].select_columns(['data_name', 'text', 'num_choices', 'answer']))

        # Concatenate datasets
        for split in splits:
            formatted_datasets[split] = concatenate_datasets(formatted_datasets[split])
        formatted_datasets = DatasetDict(formatted_datasets)
        # print(f'Formatted datasets: {formatted_datasets}')
        # print(f"Text example:\n{formatted_datasets['train']['text'][0]}")
    else:
        # Load datasets
        if data_name in [
            'axb', 'axg', 'boolq', 'cb', 'copa', 'multirc',
            'record', 'rte', 'wic', 'wsc', 'wsc.fixed',
        ]:
            datasets = load_dataset(path='super_glue', name=data_name)
        elif data_name == 'openbookqa':
            datasets = load_dataset(path=data_path, name='additional')
        elif data_name == 'arc_c':
            datasets = load_dataset(path="allenai/ai2_arc", name='ARC-Challenge')
            data_name = "ai2_arc"
        elif data_name == 'arc_e':
            datasets = load_dataset(path="allenai/ai2_arc", name='ARC-Easy')
            data_name = "ai2_arc"
        elif data_name == "hellaswag":
            datasets = load_dataset(path="Rowan/hellaswag")
        elif data_name == "winogrande":
            datasets = load_dataset("winogrande", "winogrande_debiased")
        elif data_name == 'scienceqa':
            datasets = load_dataset(path=data_path)
            datasets = datasets.filter(lambda example: example["image"] is None)
        else:
            datasets = load_dataset(path=data_path)
        # print(f'Datasets: {datasets}')
        # print(f"Example: {datasets['train'][0]}")

        # Format datasets
        formatted_datasets = datasets.map(
            lambda example: format_text(example, data_name, prompt_only=prompt_only),
            batched=False, load_from_cache_file=False)

    return formatted_datasets


if __name__ == '__main__':
    data_path = 'arc_c'
    x = get_formatted_datasets(data_path=data_path, prompt_only=False)
    print(x["train"][0])
    print(len(x["test"]))
