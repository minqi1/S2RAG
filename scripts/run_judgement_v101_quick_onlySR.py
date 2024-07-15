from metrics import *
from typing import List, Dict, Any, Tuple, Literal
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import torch
from tqdm import tqdm
import utils
import numpy as np
import os
import sys
import argparse
from utils import TASK_INST
import jsonlines
import random 
import copy

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

parser = argparse.ArgumentParser()
parser.add_argument("--choice", type=int, default=3, help="Choose the dataset to evaluate")
parser.add_argument("--model", type=str, default="llama3", help="Model name")
args = parser.parse_args()

choice = args.choice

if 'lama2' in args.model and '7' in args.model:
    model_name = "/apdcephfs_qy3/share_4983883/ping_test/ping/hf_model/Llama-2-7b-chat-hf"
elif 'lama2' in args.model and '13' in args.model:
    model_name = "/apdcephfs_qy3/share_4983883/ping_test/ping/hf_model/Llama-2-13b-chat-hf"
elif 'lama3' in args.model:
    model_name = "/apdcephfs_qy3/share_4983883/ping_test/ping/hf_model/Meta-Llama-3-8B-Instruct"
elif 'selfrag' in args.model:
    model_name = "/apdcephfs_qy3/share_4983883/ping_test/ping/hf_model/selfrag_llama2_7b"
else:
    raise ValueError('Model name not found!')

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

max_new_tokens = 20
model = MyModel(model, tokenizer, max_new_tokens=max_new_tokens)
data_list = ['pqa', 'tqa', 'arc', 'health']
task = data_list[choice]
print(f"Current task: {task}...")
if "Llama-3" in model_name:
    file_path = f'/apdcephfs_qy3/share_4983883/ping_test/rag/minqi_dev/self-rag-main/retrieval_lm/minqi_inf_output/llama3Ins-{task}.json'
    # file_path = f'/home/minqi/code/self-rag/retrieval_lm/minqi_inf_output/llama3Ins-{task}.json'
elif 'Llama-2-7b' in model_name:
    file_path = f'/apdcephfs_qy3/share_4983883/ping_test/rag/minqi_dev/self-rag-main/retrieval_lm/minqi_inf_output/llama2chat-{task}.json'  # w exp has different format
    # file_path = f'/home/minqi/code/self-rag/retrieval_lm/minqi_inf_output/llama3Ins-{task}-fullspan-w_exp.json'
elif 'Llama-2-13b' in model_name:
    file_path = f'/apdcephfs_qy3/share_4983883/ping_test/rag/minqi_dev/self-rag-main/retrieval_lm/minqi_inf_output/llama2chat13b-{task}.json'
elif 'selfrag' in model_name:
    file_path = f'/apdcephfs_qy3/share_4983883/ping_test/rag/minqi_dev/self-rag-main/retrieval_lm/minqi_inf_output/selfrag-{task}.json'

with open(file_path) as f:
    json_file = json.load(f)

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

response_prefix = ""
answer_prefix = ""
closed = False
is_llama3 = False  
if "Llama-3" in model_name:
    response_prefix = tokenizer.decode(128006) + tokenizer.decode(78191) + tokenizer.decode(128007) + tokenizer.decode(271)
    is_llama3 = True
if task == "arc":
    answer_prefix = 'The best answer choice is ' 
    closed = True
if task == "health":
    answer_prefix = 'The claim is '
    closed = True
cur_prefix = response_prefix + answer_prefix

print(f"Current prefix: {cur_prefix}")

def preprocess_input_data(dataset, task=None, is_llama3=False):
    new_data = []
    cur_task = task
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


def score_answer(context, a, question):
    start_str = "Given a question and a possible answers, you need to score it from 0-10 based on the context and answer quality. "
    ctx_str = """Some context is provided, but please note that this context might be irrelevant or misleading (sourced from similar names): \n{context}""".format(context=context)
    
    ins_str = """
    
    The good answer should be concise, factual, correct and strictly follow the answer format (no special tokens or emojis). 
    
    Question: {question}

    Answer: {a}

    Please read the provided context, question, and possible answers carefully before scoring (from 0-10).
    
    """.format(question=question, a=a)
    
    return start_str + '\n' + ctx_str + '\n' + ins_str
    
prefix = ['no_retrieval', 'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4']

scores = [] # list of list, len == len(json_file)
best = [] # list of options, len == len(json_file)
best_pred = [] # list of predictions, len == len(json_file)
performance = [] # list of scores, len == len(json_file)    
inf_performance = [] # list of scores, len == len(json_file)
question = []  # list of questions, len == len(json_file)
golds = [] # list of gold answers, len == len(json_file)
cnt = 0
stage1_cnt = 0 # count of stage 1: direct judgment
stage2_cnt = 0 # count of stage 2: confidence judgment
stage3_cnt = 0 # count of stage 3: direct judgment w ref
stage4_cnt = 0 # count of stage 4: no more ref, stop
bad_pair_cnt = 0 # count of bad pairs
all_rec = {'num_generations': [], 'cfd': [], 'score_after_cfd': [], 'judge_p': [], 'score_after_judge': [], 'score_after_voting': [],
           'global_record': [], 'chosen_ans': [], 'performance': [] , 'possible_correct': [], 'question': [], 'golds': []} # record

if closed:
    if task == 'arc':
        auxilliary_data_path = '/apdcephfs_qy3/share_4983883/ping_test/rag/eval_data/arc_challenge_processed.jsonl'
    elif task == 'health':
        auxilliary_data_path = '/apdcephfs_qy3/share_4983883/ping_test/rag/eval_data/health_claims_processed.jsonl'
    auxilliary_data = load_jsonlines(auxilliary_data_path)
    auxilliary_data = preprocess_input_data(auxilliary_data, task=task, is_llama3=is_llama3)
    assert len(json_file) == len(auxilliary_data), f'Length of json file {len(json_file)} and auxilliary data {len(auxilliary_data)} do not match!'

def filter_score(l:str):
    for word in l[:3]: # only consider the first 3 words
        if word.isdigit():
            return int(word)
    return 5 # default score

# ref method
for i, item in enumerate(tqdm(json_file, desc="Processing items")):
    print('='*80)
    if closed:
        cur_question = auxilliary_data[i]['instruction']
    else:
        cur_question = item['question']

    question.append(cur_question)
    golds.append(item['gold'])

    all_rec['cfd'].append([]) # list of float (element level)

    cur_pos = 0
    ctxs = []
    cur_ans = []
    format_prompts = []
    for j in range(len(prefix)):
        all_rec['cfd'][-1].append(compute_confidence(item[prefix[j] + '_log_probs']))
        ctxs.append(item[prefix[j] + '_ctx']) if j > 0 else ''
        cur_ans.append(item[prefix[j]])
        prompt = score_answer(context=ctxs[-1], a=cur_ans[-1].replace('assistant', '').strip(), question=cur_question)
        if 'Llama-3' in model_name or 'Llama-2' in model_name:
            chat = [{"role": "user", "content": prompt}]
            prompt_fmt = tokenizer.apply_chat_template(chat, tokenize=False)
            prompt_fmt = prompt_fmt + response_prefix + 'The score is : '

        elif 'selfrag' in model_name:
            prompt_fmt = '### Instruction: ' + prompt + '### Response:\n The score is: '
        else:
            raise ValueError('Model name not found!')
        format_prompts.append(prompt_fmt)
        
    cur_preds = model.generate(format_prompts, max_new_tokens=10)
    # only need the text part
    cur_preds_scores = [filter_score(pred.outputs[0].text) for pred in cur_preds]
    
    all_rec['global_record'].append(copy.deepcopy(cur_preds_scores))

    # output the best one
    max_score = max(cur_preds_scores)
    best_pos = cur_preds_scores.index(max_score) # find the (first) best one including the bertscore
    best.append(prefix[best_pos])
    best_pred.append(item[prefix[best_pos]])
    scores.append(cur_preds_scores)
    cur_f1 = metric_max_over_ground_truths(f1_score, best_pred[-1], item['gold'])
    if closed:
        score_ = metric_max_over_ground_truths(loose_acc, best_pred[-1], item['gold'])
    else:
        score_ = metric_max_over_ground_truths(loose_match, best_pred[-1], item['gold'])
    performance.append(score_)
    print(f'Task: {task} \nQuestion: {question[-1]} \nGold: {golds[-1]} \nBest pred {best_pred[-1]}\nBest: {best[-1]} Internal score: {scores[-1]} \nCurrent metric: {performance[-1]}')

    all_rec['chosen_ans'].append(prefix[best_pos])
        
    if i % 10 == 0:
        print('*'*80)
        print(f'Avg performance: {np.mean(performance)}')
        print('*'*80)

# update the all_rec
all_rec['performance'] = performance
all_rec['question'] = question
all_rec['golds'] = golds

# record the basic info (task, model, cfd, s_cfd, proceed_threshold)
final_rec = {'basic info': f'Task: {task}, {args}', 
             'results': all_rec}

# save the records
with open(file_path.replace('.json', '-final_rec_SRnCF.json'), 'w') as f:
    json.dump(final_rec, f)

# print out the final results
print(f'Final results: {task} \nAvg Score: {np.mean(performance)}')
# count the number of each option
count = [0 for _ in range(len(prefix))]
for i in range(len(best)):
    count[prefix.index(best[i])] += 1

for i in range(len(count)):
    print(f'Count of {prefix[i]}: {count[i]}')

# print args
print(args)
