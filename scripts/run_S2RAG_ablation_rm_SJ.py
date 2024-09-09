from metrics import *
from typing import List, Dict, Any, Tuple, Literal
import json
import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import torch
from tqdm import tqdm
import utils
import numpy as np
import os
import sys
import argparse
from utils import TASK_INST, SelfCheckBERTScore, preprocess_input_data, load_jsonlines, \
    compute_confidence, get_retrieval_p_compare, cal_bertscore, choose_better_prompt 
import jsonlines
import random 
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--choice", type=int, default=3, help="Choose the dataset to evaluate")
parser.add_argument("--cfd", type=float, default=0.95, help="Confidence threshold for no ret")
parser.add_argument("--s_cfd", type=float, default=1.1, help="Cfd scale for rest")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name") # judge model, here we don't use any judge model since we remove self-judgment
parser.add_argument("--tau", type=float, default=0.45, help="Cautiousness") #
parser.add_argument("--bertmodel", type=str, default=None, help="BERT model path")
parser.add_argument("--base", type=str, default=None, help="Base dir") # base generatorargs = parser.parse_args()
args = parser.parse_args()

# choice = args.choice

en_core_web_sm_path = args.bertmodel

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True, model_path=en_core_web_sm_path) # non-factual score
# test
print('Test BertScore')
print(1 - selfcheck_bertscore.predict(['This is a test'], ['This is a test'])[0])

model_name = args.model_name

data_list = ['pqa', 'tqa', 'arc', 'health']
task = data_list[args.choice]
print(f"Current task: {task}...")

if args.base is None:
    args.base = model_name
else:
    if 'lama2' in args.base and '7' in args.base:
        args.base = 'Llama-2-7b'
    elif 'lama3' in args.base:
        args.base = 'Llama-3'

if "Llama-3" in args.base:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/llama3Ins-{task}.json'
elif 'Llama-2-7b' in args.base:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/llama2chat-{task}.json'  # w exp has different format
elif 'selfrag' in args.base:
    file_path = f'/home/minqi/code/S2RAG/minqi_inf_output/selfrag-{task}.json'
else:
    raise ValueError(f'Base model: {args.base} not found!')

with open(file_path) as f:
    json_file = json.load(f)
    

response_prefix = ""
answer_prefix = ""
closed = False
is_llama3 = False  
    
prefix = ['no_retrieval', 'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4']

scores = [] # list of list, len == len(json_file)
best = [] # list of options, len == len(json_file)
best_pred = [] # list of predictions, len == len(json_file)
performance = [] # list of scores, len == len(json_file)    
question = []  # list of questions, len == len(json_file)
golds = [] # list of gold answers, len == len(json_file)
all_rec = {'num_generations': [], 'cfd': [], 'score_after_cfd': [], 'judge_p': [], 'score_after_judge': [], 'score_after_voting': [], 
        'global_record': [], 'chosen_ans': [], 'performance': [] , 'possible_correct': [], 'question': [], 'golds': []} # record

if closed:
    if task == 'arc':
        auxilliary_data_path = '/home/minqi/code/S2RAG/data_eval/arc_challenge_processed_example.jsonl' # change to real path
    elif task == 'health':
        auxilliary_data_path = '/home/minqi/code/S2RAG/data_eval/health_claims_processed_example.jsonl'
    auxilliary_data = load_jsonlines(auxilliary_data_path)
    auxilliary_data = preprocess_input_data(auxilliary_data, task=task, is_llama3=is_llama3, origin_task=False)
    assert len(json_file) == len(auxilliary_data), f'Length of json file {len(json_file)} and auxilliary data {len(auxilliary_data)} do not match!'

# ref method
for i, item in enumerate(tqdm(json_file, desc="Processing items")):
    score_record = [0 for _ in range(len(prefix))]
    print('='*80)
    if closed:
        cur_question = auxilliary_data[i]['instruction']
    else:
        cur_question = item['question']

    question.append(cur_question)
    golds.append(item['gold'])

    # record the error records for further analysis ONLY DELETE LATER
    possible_correct_pred = []
    for i_0 in range(len(prefix)):
        # check whether there exists a better answer
        tmp_pred = item[prefix[i_0]]
        if closed:
            score_ = metric_max_over_ground_truths(loose_acc, tmp_pred, item['gold'])
        else:
            score_ = metric_max_over_ground_truths(loose_match, tmp_pred, item['gold'])
        if score_ == 1.0:
            possible_correct_pred.append((prefix[i_0], tmp_pred, score_))
    all_rec['possible_correct'].append(possible_correct_pred) # record the possible correct answers
    all_rec['cfd'].append([]) # list of float (element level)
    all_rec['score_after_cfd'].append([]) # list of list (round level) 
    all_rec['score_after_judge'].append([]) # list of list (round level)
    all_rec['score_after_voting'].append([]) # list of list (round level)

    for cur_pos in range(len(prefix)):
        ans_cfd = compute_confidence(item[prefix[cur_pos] + '_log_probs'])
        all_rec['cfd'][-1].append(ans_cfd) # append the cfd scores
    # award to the highest cfd
    max_cfd = max(all_rec['cfd'][-1])
    max_cfd_pos = all_rec['cfd'][-1].index(max_cfd)
    score_record[max_cfd_pos] += 1
    all_rec['score_after_cfd'][-1].append(copy.deepcopy(score_record)) # append the cfd scores
    all_rec['score_after_judge'][-1].append(copy.deepcopy(score_record)) # append the judge scores

    # 3. Voting, 1 score
    tmp_bertscore = [0 for _ in range(len(prefix))]
    # quick check
    quick_check = False # quick check all answers vs current ref, if supported, +1 tmp score (remove after this round)
    if (closed and cur_pos < 1 and 'lama' in model_name) or (closed and 'selfrag' in model_name):
        for i_ref in range(len(prefix)):
            ref_ans = item[prefix[i_ref]]
            ref_norm = normalize_answer(ref_ans)
            for i_ in range(len(prefix)):
                cur_norm = normalize_answer(item[prefix[i_]])
                try:
                    if cur_norm.split()[0] == ref_norm.split()[0] and i_ref != i_: 
                        tmp_bertscore[i_] += 1
                        quick_check = True
                except:
                    pass
            
    # use bertscore to choose better answer
    preds = [item[prefix[i_]] for i_ in range(len(prefix))] 
    if not quick_check:
        for i_ref in range(len(prefix)):
            ref_ans = item[prefix[i_ref]]
            bert_score = cal_bertscore(preds, ref_ans, selfcheck_bertscore, closed=closed)
            bert_score[i_ref] = 0 # remove the self ref score
            for ii, it in enumerate(bert_score):
                nearest_float = round(it, 1)
                tmp_bertscore[ii] += nearest_float
    
    # normalize the tmp_bertscore (norm to 1)
    tmp_bertscore = [it/(cur_pos + 3) for it in tmp_bertscore] # avg the bertscore
    score_record_bert = [score_record[ii] + tmp_bertscore[ii] for ii in range(len(prefix))]
    all_rec['score_after_voting'][-1].append(copy.deepcopy(score_record_bert)) # append the voting scores

    # output the best one
    max_score = max(score_record_bert)
    best_pos = score_record_bert.index(max_score) # find the (first) best one including the bertscore
    best.append(prefix[best_pos])
    best_pred.append(item[prefix[best_pos]])
    scores.append(score_record_bert)
    cur_f1 = metric_max_over_ground_truths(f1_score, best_pred[-1], item['gold'])
    if closed:
        score_ = metric_max_over_ground_truths(loose_acc, best_pred[-1], item['gold'])
    else:
        score_ = metric_max_over_ground_truths(loose_match, best_pred[-1], item['gold'])
    performance.append(score_)
    print('Best:', prefix[best_pos])
    print(f'Task: {task} \nQuestion: {question[-1]} \nGold: {golds[-1]} \nBest pred {best_pred[-1]}\nBest: {best[-1]} Internal score: {scores[-1]} \nCurrent metric: {performance[-1]}')
    if len(possible_correct_pred) > 0:
        print('Possible correct pred:', possible_correct_pred[0])
    print('='*80)

    all_rec['global_record'].append(copy.deepcopy(score_record_bert))
    all_rec['chosen_ans'].append(prefix[best_pos])
    all_rec['num_generations'].append(len(prefix)) #  
        
    if i % 10 == 0:
        print('*'*80)
        print(f'Avg performance: {np.mean(performance)}')
        print('*'*80)

# update the all_rec
all_rec['performance'] = performance
all_rec['question'] = question
all_rec['golds'] = golds

# record the basic info
final_rec = {'basic info': f'Task: {task}, {args}', 
             'results': all_rec}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.default(value) if isinstance(value, (np.ndarray, dict)) else value for key, value in obj.items()}
        return super(NumpyEncoder, self).default(obj)

# save the records
with open(file_path.replace('.json', '-final_rec_rm_SJ.json'), 'w') as f:
    json.dump(final_rec, f, cls=NumpyEncoder)


# print out the final results
print(f'Final results: {task} \nAvg Score: {np.mean(performance)}')
count = [0 for _ in range(len(prefix))]
for i in range(len(best)):
    count[prefix.index(best[i])] += 1

for i in range(len(count)):
    print(f'Count of {prefix[i]}: {count[i]}')

print(args)