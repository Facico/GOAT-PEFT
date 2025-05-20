from transformers import StoppingCriteria, StoppingCriteriaList
import re
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
        
class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(stop_string in decoded_text for stop_string in self.stop_strings)
    
def trim_output_gsm8k(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Q:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely
    if "\n####" in output:
        output = output.split("\n####")[-1]
    else:
        if "\nA:" in output:
            output = output.split("\nA:")[1]
        for prefix in [instruction_prefix, question_prefix, comment_prefix]:
            if prefix in output:
                output = output.split(prefix)[0]

    return output