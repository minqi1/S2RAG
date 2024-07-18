from metrics import *
from typing import List
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import torch
from tqdm import tqdm
import numpy as np
import argparse
from utils import TASK_INST, preprocess_input_data, load_jsonlines, compute_confidence, compute_inv_perplexity
import jsonlines
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--choice", type=int, default=3, help="Choose the dataset to evaluate")
parser.add_argument("--model_name", type=str, default="llama3", help="Model name")
args = parser.parse_args()

choice = args.choice
model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

max_new_tokens = 20
model = MyModel(model, tokenizer, max_new_tokens=max_new_tokens)
data_list = ['pqa', 'tqa', 'health', 'arc']
task = data_list[choice]
print(f"Current task: {task}...")
if "Llama-3" in model_name:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/llama3Ins-{task}.json'
elif 'Llama-2-7b' in model_name:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/llama2chat-{task}.json' 
elif 'selfrag' in model_name:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/selfrag-{task}.json'

with open(file_path) as f:
    json_file = json.load(f)

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
question = []  # list of questions, len == len(json_file)
golds = [] # list of gold answers, len == len(json_file)

all_rec = {'cfd': [], 'global_record': [], 'chosen_ans': [], 'performance': [] , 'question': [], 'golds': []} # record

if closed:
    if task == 'arc':
        auxilliary_data_path = '/home/minqi/code/S2RAG/data_eval/arc_challenge_processed_example.jsonl' # change to real path
    elif task == 'health':
        auxilliary_data_path = '/home/minqi/code/S2RAG/data_eval/health_claims_processed_example.jsonl'
    auxilliary_data = load_jsonlines(auxilliary_data_path)
    auxilliary_data = preprocess_input_data(auxilliary_data, task=task, is_llama3=is_llama3, origin_task=False)
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
        ctxs.append(item[prefix[j] + '_ctx'] if j > 0 else '')
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
    best_pos = cur_preds_scores.index(max_score) # find the (first) best one
    best.append(prefix[best_pos])
    best_pred.append(item[prefix[best_pos]])
    scores.append(cur_preds_scores)
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
with open(file_path.replace('.json', '-final_rec_SEnCFD.json'), 'w') as f:
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
