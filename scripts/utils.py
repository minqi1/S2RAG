"""
Code Original Source:
- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. pages 1, 3, 7,
8, 11, 13, 16, 24, 25, 36
  - Repository: https://github.com/AkariAsai/self-rag
  - Description: data preprocessing and postprocessing functions, load_jsonlines, load_file functions, prompt dictionary, task instructions

  
- Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023. pages 5, 11
  - Repository: https://github.com/potsawee/selfcheckgpt
  - Description: SelfCheckBERTScore class, expand_list1, expand_list2 functions 
-----------------------------------------------------------------
  
Modifications are from:

Contributors:
- Minqi Xiang (mx716@ic.ac.uk) Equally contributed
- Zihan Zhu (zcabhub@ucl.ac.uk) Equally contributed

Public Source: https://github.com/zhuzihan728/LLM_Adaptive_RAG
Private Source: https://github.com/minqi1/S2RAG
Description: compute_confidence, compute_inv_perplexity, get_retrieval_p_compare functions

"""

import jsonlines
import json
import copy
import re
import spacy
import torch
import bert_score
from typing import Dict, List, Set, Tuple, Union
import numpy as np  
import string

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
}


TASK_INST = {
             "fever": "Is the following claim true or false? Say true if it's true; otherwise say false. Then, provide a brief explanation to support your answer.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice. Then, provide a brief explanation to support your answer.",
             }

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def preprocess_input_data_origin(dataset, task=None):
    """
    Code from Other Sources:
    - Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. pages 1, 3, 7,
    8, 11, 13, 16, 24, 25, 36
    - Repository: https://github.com/AkariAsai/self-rag
    - Description: data preprocessing function    
    """
    new_data = []
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = item["question"] + choices
            item["answers"] = [item["answerKey"]]
        elif task == "fever":
            item["instruction"] = f"Is the claim \"{item['question']}\" true or false?"
        else:
            item["instruction"] = item["question"]
        
        new_data.append({'instruction': item["instruction"]})

    return new_data


def preprocess_input_data(dataset, task=None, is_llama3=False, origin_task=True):
    """
    Code from Other Sources:
    - Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. pages 1, 3, 7,
    8, 11, 13, 16, 24, 25, 36
    - Repository: https://github.com/AkariAsai/self-rag
    - Description: data preprocessing function    
    """
    new_data = []
    cur_task = task
    if not origin_task:
        if cur_task == "arc":
            cur_task = "arc_c"
        elif cur_task == "health":  
            cur_task = "fever"

    if cur_task in TASK_INST:
        instruction = TASK_INST[cur_task]
    else:
        instruction = None
    for item in dataset:
        if cur_task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = item["question"] + choices
            item["answers"] = [item["answerKey"]]
        elif cur_task == "fever" and is_llama3:
            item["instruction"] = f'Is the claim "{item["question"]}" true or false?'
        else:
            item["instruction"] = item["question"]
        item["instruction"] = instruction + "\n\n" + item["instruction"] if instruction is not None else item["instruction"]
        new_data.append(item)
    return new_data

def postprocess_ans(answer):
    """
    Code from Other Sources:
    - Akari Asai
    - Repository: https://github.com/AkariAsai/self-rag
    - Description: data preprocessing and postprocessing functions    
    """

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")
    if type(answer) is str and len(answer) > 0 and (answer[0] == "#" or answer[0] == ":"):
        answer = answer[1:]
        
    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()
    return white_space_fix(handle_punc(lower(answer))).strip()


# SelfCheck - BERTScore utils
def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded


class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self, default_model="en", rescale_with_baseline=True,  model_path=None):
        """
        :default_model: model for BERTScore
        :rescale_with_baseline:
            - whether or not to rescale the score. If False, the values of BERTScore will be very high
            - this issue was observed and later added to the BERTScore package,
            - see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
        """
        if model_path is None:
            model_path = "en_core_web_sm"
        self.nlp = spacy.load(model_path) 
        self.default_model = default_model # en => roberta-large
        self.rescale_with_baseline = rescale_with_baseline
        print("SelfCheck-BERTScore initialized")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 1] # change to len(sent) > 1
            num_sentences_sample  = len(sentences_sample)

            # error handling
            if num_sentences_sample == 0:
                print("warning: empty sentence, skipping...")
                bertscore_array[:,s] = 0 # low bertscore
                continue 

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            P, R, F1 = bert_score.score(
                    cands, refs,
                    lang=self.default_model, verbose=False,
                    rescale_with_baseline=self.rescale_with_baseline,
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent


def compute_confidence(log_probs:List):
    '''
    log_probs: List[float]
    '''
    return np.mean(np.exp(log_probs))

def compute_inv_perplexity(log_probs:List):
    '''
    log_probs: List[float]
    '''
    return np.exp(np.mean(log_probs))


def get_retrieval_p_compare(pred, tokenizer, new_format=False):
    '''modified for 4 options'''
    has_judgment = False
    judge_ind = 0
    retrieve_p_hard = None

    token_ids = pred.outputs[0].token_ids
    if new_format:
        token_ids = token_ids[::-1]

    for ind, id_ in enumerate(token_ids):
        word = tokenizer.decode(id_).strip()
        if word in ['1', '2', '3', '4']:
            has_judgment = True
            judge_ind = len(token_ids) - ind - 1 if new_format else ind
            retrieve_p_hard = int(word)
            break

    log_prob_dc = pred.outputs[0].logprobs[judge_ind] # use the first token if no judgment
    token_1 = tokenizer.encode('1')[-1]
    token_2 = tokenizer.encode('2')[-1]
    token_3 = tokenizer.encode('3')[-1]
    token_4 = tokenizer.encode('4')[-1]
    retrieve_p = [np.exp(log_prob_dc[token])/(np.exp(log_prob_dc[token_1])+np.exp(log_prob_dc[token_2]) + np.exp(log_prob_dc[token_3]) + np.exp(log_prob_dc[token_4])) for token in [token_1, token_2, token_3, token_4]]

    return retrieve_p, retrieve_p_hard, has_judgment

def cal_bertscore(preds, ref, bertscore_model, closed=False):
    # use bertscore to choose better answer
    bert_score = 1 - bertscore_model.predict(preds, [ref])
    return bert_score

# choose better answers
def choose_better_prompt(context, a, cfd_a, b, cfd_b, question, closed=False):
    start_str = "Given a question and two possible answers, you need to determine which answer is better, equally good or equally bad. "
    ctx_str = """Some context is provided, but please note that this context might be irrelevant or misleading (sourced from similar names): \n{context}""".format(context=context)
    
    ins_str = """
    
    The good answer should be concise, factual, correct and strictly follow the answer format (no special tokens or emojis). 
    - If the first answer is better, type "1", with short explanation.
    - If the second answer is better, type "2", with short explanation.
    - If both answers are equal or similar, and you are sure they are both correct, type "3", with short explanation.
    - If two answers have conflicts and you need more retrieved passages to determine their correctness, type "4".
    
    Question: {question}

    Answer 1: {a}
    Answer 1 confidence score: {cfd_a:.3f}

    Answer 2: {b}
    Answer 2 confidence score: {cfd_b:.3f}
    
    Please read the provided context, question, and possible answers carefully before making a judgement.
    - Ensure your decision is based on a thorough understanding of the context and question.

    When checking the answer, prioritize as follows:
    1. The context is not misleading, and the answer is supported by the context. The format is exactly as expected. -- This is the best answer.
    2. The answer does not use the context, but you believe the answer is correct. The format is exactly as expected. -- This is the second-best answer.
    3. The context is misleading, and the answer is misled by the context, hence it is wrong. Or the answer includes any emojis or special tokens -- This is the worst answer.
    4. The answer does not directly address the question or challenge the premise of the question. -- This is the worst answer.
    """.format(question=question, a=a, b=b, cfd_a=cfd_a, cfd_b=cfd_b)

    
    return start_str + '\n' + ctx_str + '\n' + ins_str

# choose better
def train_instruction(ans_a, ans_b, question): # ans_a: (ctx, str, cfd)
    start_str = "Given a question and two possible answers, you need to determine which answer is better, equally good or equally bad. "
    context = ans_a[0] + '\n' + ans_b[0]
    ctx_str = """Some context is provided, but please note that this context might be irrelevant or misleading (sourced from similar names): \n{context}""".format(context=context)
    
    ins_str = """
    
    The good answer should be concise, factual, correct and strictly follow the answer format (no special tokens or emojis). 
    - If the first answer is better, type "1"
    - If the second answer is better, type "2"
    - If both answers correct, type "3"
    - If two answers are both wrong, type "4".
    
    Question: {question}
    
    Answer 1: {ans_a[1]}
    Answer 1 confidence score: {ans_a[2]:.3f}

    Answer 2: {ans_b[1]}
    Answer 2 confidence score: {ans_b[2]:.3f}
    
    Please read the provided context, question, and possible answers carefully before making a judgement.

    When checking the answer, prioritize as follows:
    a. The context is not misleading, and the answer is supported by the context. The format is exactly as expected. -- This is the best answer.
    b. The answer does not use the context, but you believe the answer is correct. The format is exactly as expected. -- This is the second-best answer.
    c. The context is misleading, and the answer is misled by the context, hence it is wrong. Or the answer includes any emojis or special tokens -- This is the worst answer.
    d. The answer does not directly address the question or challenge the premise of the question. -- This is the worst answer.
    """.format(question=question, ans_a=ans_a, ans_b=ans_b)

    
    return start_str + '\n' + ctx_str + '\n' + ins_str