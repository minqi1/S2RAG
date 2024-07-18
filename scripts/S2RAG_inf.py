# For higher efficiency, we directly load the responses from previously stored files, and apply S2RAG to choose for the best response.

from metrics import *
from typing import List, Dict, Any, Tuple, Literal
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import torch
from tqdm import tqdm
import numpy as np
import os
import sys
import argparse
from utils import TASK_INST, SelfCheckBERTScore, preprocess_input_data, load_jsonlines, \
    compute_confidence, get_retrieval_p_compare, cal_bertscore, choose_better_prompt
import jsonlines
import random 
import copy
from peft import PeftModel, LoraConfig

parser = argparse.ArgumentParser()
parser.add_argument("--choice", type=int, default=3, help="Choose the dataset to evaluate")
parser.add_argument("--cfd", type=float, default=0.95, help="Confidence threshold for no ret")
parser.add_argument("--s_cfd", type=float, default=1.1, help="Cfd scale for rest")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name") # judge model, normally the same as the base model, except for the last chapter: trained judge
parser.add_argument("--tau", type=float, default=0.45, help="Cautiousness") #
parser.add_argument("--use_trained", action="store_true", help="Use trained model")
parser.add_argument("--trained_path", type=str, default="weights_minqi/llama3_8B_256", help="Trained model adapter path")
parser.add_argument("--test", action="store_true", help="run on test data")
parser.add_argument("--bertmodel", type=str, default=None, help="BERT model path")
parser.add_argument("--train_idx_path", type=str, default="data_training/train_data_fusion.jsonl", help="Training data idx path")
parser.add_argument("--base", type=str, default=None, help="Base dir") # base generator
args = parser.parse_args()

choice = args.choice
model_name = args.model_name

en_core_web_sm_path = args.bertmodel

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True, model_path=en_core_web_sm_path) # non-factual score
# test
print('Test BertScore')
print(1 - selfcheck_bertscore.predict(['This is a test'], ['This is a test'])[0])

if args.use_trained: # use trained judge model
    base_dir = args.trained_path
    tokenizer = AutoTokenizer.from_pretrained(base_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, base_dir)
else: # use untrained model as judge
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

max_new_tokens = 100
model = MyModel(model, tokenizer, max_new_tokens=max_new_tokens)
data_list = ['pqa', 'tqa', 'health', 'arc']
task = data_list[choice]
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

# load test data
trained_idx = []
if args.test:
    idx_path = args.train_idx_path
    idx_data = load_jsonlines(idx_path)
    tmp_df = pd.DataFrame(idx_data)
    tmp_df = tmp_df[tmp_df['task'] == task]
    trained_idx = tmp_df['data_idx'].unique().tolist()

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

    
prefix = ['no_retrieval', 'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4']

scores = [] # list of list, len == len(json_file)
best = [] # list of options, len == len(json_file)
best_pred = [] # list of predictions, len == len(json_file)
performance = [] # list of scores, len == len(json_file)    
inf_performance = [] # list of scores, len == len(json_file)
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
    if args.test and  i in trained_idx:
        if i % 10 == 0:
            print('*'*80)
            print('Perfect inf performance: ', np.mean(inf_performance))
            print(f'Avg performance: {np.mean(performance)}')
            print('*'*80)
        print(f'Skip trained data: {i}')
        continue # skip the trained data

    score_record = [0 for _ in range(len(prefix))]
    print('='*80)
    if closed:
        cur_question = auxilliary_data[i]['instruction']
    else:
        cur_question = item['question']

    question.append(cur_question)
    golds.append(item['gold'])

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
    if len(possible_correct_pred) > 0:
        inf_performance.append(1)
    else:
        inf_performance.append(0)

    all_rec['possible_correct'].append(possible_correct_pred) # record the possible correct answers
    all_rec['cfd'].append([]) # list of float (element level)
    all_rec['score_after_cfd'].append([]) # list of list (round level) 
    all_rec['judge_p'].append([]) # list of list (round level)
    all_rec['score_after_judge'].append([]) # list of list (round level)
    all_rec['score_after_voting'].append([]) # list of list (round level)

    cur_pos = 0
    while True and cur_pos <= len(prefix) - 2: 
        ans1_cfd = compute_confidence(item[prefix[cur_pos] + '_log_probs'])
        ans2_cfd = compute_confidence(item[prefix[cur_pos+1] + '_log_probs'])
        if all_rec['cfd'][-1] == []:
            all_rec['cfd'][-1].extend([ans1_cfd, ans2_cfd]) # append no retrieval and ret_0
        else:
            all_rec['cfd'][-1].append(ans2_cfd) # append the second one

        # 1. score from confidence: noticeable difference 1 score
        if prefix[cur_pos] != 'no_retrieval': # fair comparison
            if ans1_cfd > args.s_cfd * ans2_cfd:
                score_record[cur_pos] += 1
            elif ans2_cfd > args.s_cfd  * ans1_cfd:
                score_record[cur_pos + 1] += 1
        else: 
            if ans1_cfd > args.cfd * ans2_cfd:
                score_record[cur_pos] += 1 
            elif ans2_cfd > ans1_cfd / args.cfd * args.s_cfd:
                score_record[cur_pos + 1] += 1

        all_rec['score_after_cfd'][-1].append(copy.deepcopy(score_record)) # append the cfd scores

        # 2. score from direct judgement (w/o ref) 1 score
        if cur_pos > 0:
            cur_ctx = item[prefix[cur_pos] + "_ctx"] + '\n' + item[prefix[cur_pos+1] + "_ctx"]
        else:
            cur_ctx = item[prefix[cur_pos+1] + "_ctx"] # no retrieval does not have ctx

         # add quick check
        quick_check = False # quick check
        both_good = False
        retrieve_p = [0, 0, 0, 0]
        if (closed and 'selfrag' in model_name) or (task=='arc' and 'lama' in model_name and cur_pos < 1): # quick check the answer for the first pair (assumption that they will not be equally bad)
            ans1_norm = normalize_answer(item[prefix[cur_pos]])
            ans2_norm = normalize_answer(item[prefix[cur_pos+1]])
            try:
                if ans1_norm.split()[0] == ans2_norm.split()[0]: 
                    retrieve_p[2] = 1
                    quick_check = True
                    print('Quick check: both are equally good, skip the prompt judgment!')
            except:
                pass

        if not quick_check:
            a = item[prefix[cur_pos]].replace('assistant', '').strip()
            b = item[prefix[cur_pos+1]].replace('assistant', '').strip()
            prompt = choose_better_prompt(context=cur_ctx, a=a, cfd_a=ans1_cfd, b=b, cfd_b=ans2_cfd, question=cur_question, closed=closed)
            if 'Llama-3' in model_name or 'Llama-2' in model_name:
                if not args.use_trained:
                    chat = [{"role": "user", "content": prompt}]
                    prompt_fmt = tokenizer.apply_chat_template(chat, tokenize=False)
                    prompt_fmt = prompt_fmt + response_prefix + 'My judgement is: '
                else:
                    prompt_fmt = prompt + '\nJudgement: '

            elif 'selfrag' in model_name:
                prompt_fmt = '### Instruction: ' + prompt + '### Response:\n My judgement is: '
            else:
                raise ValueError('Model name not found!')
            
            cur_pred = model.generate([prompt_fmt], max_new_tokens=10)
            
            # update score using the logits
            retrieve_p, retrieve_p_hard, has_judgment = get_retrieval_p_compare(cur_pred[0], tokenizer)
            print('Judgment:', retrieve_p, retrieve_p_hard, has_judgment)
        
        all_rec['judge_p'][-1].append(copy.deepcopy(retrieve_p)) # append the judgment
        
        abs_diff = abs(retrieve_p[0] - retrieve_p[1]) # absolute difference between two answers
        max_pos = retrieve_p.index(max(retrieve_p))
        if ((abs_diff < args.tau and retrieve_p[2] < args.tau) or max_pos==3) and cur_pos < len(prefix) - 2: # proceed to the next pair
            score_record[cur_pos] += max(retrieve_p[0] - retrieve_p[1], 0)
            score_record[cur_pos + 1] += max(retrieve_p[1] - retrieve_p[0], 0)
            cur_pos += 1 # move to the next pair: ret_0 and ret_1...
            continue
        elif max_pos == 0:
            score_record[cur_pos] += 1
        elif max_pos == 1:
            score_record[cur_pos + 1] += 1
        elif max_pos == 2: # both are equally good and correct, break the loop
            score_record[cur_pos] += 1
            score_record[cur_pos + 1] += 1
            if cur_pos < 1: # only for the first two answers
                both_good = True # both are equally good skip condition when comparing the first two answers
        else:
            if cur_pos >= len(prefix) - 2:
                pass
            else:
                raise ValueError('Invalid judgment!')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        
        all_rec['score_after_judge'][-1].append(copy.deepcopy(score_record)) # append the judgment scores

        # check whether there is only one best answer (higher in both confidence and judgment)
        max_score, second_max = sorted(score_record)[-1], sorted(score_record)[-2]
        cur_score_diff = abs(max_score - second_max)
        if (score_record.count(max_score) == 1 and cur_score_diff > args.tau) or both_good: # only best answer is noticeable better, OR first two are equally good
            best.append(prefix[score_record.index(max_score)])
            best_pred.append(item[prefix[score_record.index(max_score)]])
            scores.append(score_record)
            cur_f1 = metric_max_over_ground_truths(f1_score, best_pred[-1], item['gold'])
            if closed:
                score_ = metric_max_over_ground_truths(loose_acc, best_pred[-1], item['gold'])
            else:
                score_ = metric_max_over_ground_truths(loose_match, best_pred[-1], item['gold'])
            performance.append(score_)
            print('Best:', prefix[score_record.index(max_score)])
            print(f'Task: {task} \nQuestion: {question[-1]} \nGold: {golds[-1]} \nBest pred {best_pred[-1]}\nBest: {best[-1]} Internal score: {scores[-1]} \nCurrent metric: {performance[-1]}')
            if len(possible_correct_pred) > 0:  
                print('Possible correct pred:', possible_correct_pred[0])
            print('='*80)
            all_rec['global_record'].append(copy.deepcopy(score_record))
            all_rec['chosen_ans'].append(prefix[score_record.index(max_score)])
            all_rec['num_generations'].append(cur_pos + 2)
            break
        
        # 3. Voting, 1 score
        tmp_bertscore = [0 for _ in range(len(prefix))]
        if cur_pos < len(prefix) - 2: # not the last pair
            # quick check
            quick_check = False # quick check all answers vs current ref, if supported, +1 tmp score (remove after this round)
            if (closed and cur_pos < 1 and 'lama' in model_name) or (closed and 'selfrag' in model_name):
                for i_ref in range(cur_pos + 3):
                    ref_ans = item[prefix[i_ref]]
                    ref_norm = normalize_answer(ref_ans)
                    for i_ in range(cur_pos + 2):
                        cur_norm = normalize_answer(item[prefix[i_]])
                        try:
                            if cur_norm.split()[0] == ref_norm.split()[0] and i_ != i_ref: 
                                tmp_bertscore[i_] += 1
                                quick_check = True
                        except:
                            pass
            
            # use bertscore to choose better answer
            preds = [item[prefix[i_]] for i_ in range(cur_pos + 2)] 
            if not quick_check:
                for i_ref in range(cur_pos + 3):
                    ref_ans = item[prefix[i_ref]]
                    bert_score = cal_bertscore(preds, ref_ans, selfcheck_bertscore, closed=closed)
                    if i_ref < len(bert_score):
                        bert_score[i_ref] = 0 # remove the ref answer
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
        all_rec['num_generations'].append(cur_pos + 3) # +3 since used ref answer 
        break
        
    if i % 10 == 0:
        print('*'*80)
        print('Perfect inf performance: ', np.mean(inf_performance))
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
if not args.use_trained: # use trained judge model
    if args.test:
        with open(file_path.replace('.json', '-final_rec_test.json'), 'w') as f:
            json.dump(final_rec, f)
    else:
        with open(file_path.replace('.json', '-final_rec.json'), 'w') as f:
            json.dump(final_rec, f)
else:
    if args.test:
        if args.base == model_name: 
            with open(file_path.replace('.json', f'-final_rec_test_trained_{args.tau}.json'), 'w') as f:
                json.dump(final_rec, f)
        else: # different inf model and judge model
            with open(file_path.replace('.json', f'-final_rec_test_trained_{args.tau}_{args.base}.json'), 'w') as f:
                json.dump(final_rec, f)
    else:
        with open(file_path.replace('.json', f'-final_rec_trained_{args.tau}.json'), 'w') as f:
            json.dump(final_rec, f)

# print out the final results
print(f'Final results: {task} \nAvg Score: {np.mean(performance)}')
count = [0 for _ in range(len(prefix))]
for i in range(len(best)):
    count[prefix.index(best[i])] += 1

for i in range(len(count)):
    print(f'Count of {prefix[i]}: {count[i]}')

print(args)
