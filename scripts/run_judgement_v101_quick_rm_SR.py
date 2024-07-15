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
from utils import TASK_INST
import jsonlines
import random 
import copy

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

# the one with bertscore as stage 3
# cfd 1.1 arc best, 1 health best
# note: this should be put under retrieval_lm to import the metrics.py and tsfm_wrapper.py
parser = argparse.ArgumentParser()
parser.add_argument("--choice", type=int, default=3, help="Choose the dataset to evaluate")
parser.add_argument("--devices", type=str, default="0,1", help="Device to use")
parser.add_argument("--cfd", type=float, default=0.95, help="Confidence threshold for no ret")
parser.add_argument("--s_cfd", type=float, default=1.1, help="Cfd scale for rest")
parser.add_argument("--model", type=str, default="llama2", help="Model name")
parser.add_argument("--proceed_threshold", type=float, default=0.15, help="Threshold for proceeding") # set different val for different model with different confidence (llama2: 0.4, llama3: 0.2, selfrag: 0.2)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
# choice = args.choice

# add: bert and spacy
en_core_web_sm_path = '/apdcephfs_qy3/share_4983883/ping_test/ping/hf_model/en_core_web_sm-3.6.0/en_core_web_sm/en_core_web_sm-3.6.0' 
selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True) # non-factual score
# test
print('Test BertScore')
print(1 - selfcheck_bertscore.predict(['This is a test'], ['This is a test'])[0])

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

data_list = ['arc', 'health']
for choice in range(len(data_list)):
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

    response_prefix = ""
    answer_prefix = ""
    closed = False
    is_llama3 = False  

    if task == "arc":
        answer_prefix = 'The best answer choice is ' 
        closed = True
    if task == "health":
        answer_prefix = 'The claim is '
        closed = True
    cur_prefix = response_prefix + answer_prefix
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

    def get_retrieval_p_compare(pred, tokenizer):
        '''modified for 4 options'''
        has_judgment = False
        judge_ind = 0
        retrieve_p_hard = None
        for ind, id_ in enumerate(pred.outputs[0].token_ids):
            # print(f'ID: {id_}|{tokenizer.decode(id_)}')
            word = tokenizer.decode(id_).strip().lower()
            if word in ['1', '2', '3', '4']:
                has_judgment = True
                judge_ind = ind
                if word == '1':
                    retrieve_p_hard = 1
                elif word == '2':
                    retrieve_p_hard = 2
                elif word == '3':
                    retrieve_p_hard = 3
                else:
                    retrieve_p_hard = 4
                break
        log_prob_dc = pred.outputs[0].logprobs[judge_ind] # use the first token if no judgment
        global token_1, token_2, token_3, token_4
        retrieve_p = [np.exp(log_prob_dc[token])/(np.exp(log_prob_dc[token_1])+np.exp(log_prob_dc[token_2]) + np.exp(log_prob_dc[token_3]) + np.exp(log_prob_dc[token_4])) for token in [token_1, token_2, token_3, token_4]]
        # if retrieve_p_hard is None:
        #     retrieve_p_hard = float(retrieve_p>0.)
        return retrieve_p, retrieve_p_hard, has_judgment

    # dict_keys(['no_retrieval', 'no_retrieval_ids', 'no_retrieval_log_probs', 'all_doc_retrieval', 'all_doc_retrieval_ids', 'all_doc_retrieval_log_probs', 'retrieval', 'retrieval_token_ids', 'retrieval_log_probs', 
    # 'retrieval_res', 'question_id', 'gold', 'question', 'ret_0', 'ret_0_log_probs', 'ret_0_ctx', 'ret_0_ctx_score', 'ret_0_scores', 'ret_1', 'ret_1_log_probs', 'ret_1_ctx', 'ret_1_ctx_score', 'ret_1_scores', 'ret_2', 
    # 'ret_2_log_probs', 'ret_2_ctx', 'ret_2_ctx_score', 'ret_2_scores', 'ret_3', 'ret_3_log_probs', 'ret_3_ctx', 'ret_3_ctx_score', 'ret_3_scores', 'ret_4', 'ret_4_log_probs', 'ret_4_ctx', 'ret_4_ctx_score', 'ret_4_scores', 
    # 'all_doc_retrieval_ctx', 'all_doc_retrieval_ctx_score', 'all_doc_retrieval_scores', 'no_retrieval_scores'])

    def cal_bertscore(preds, ref, closed=False):
        # use bertscore to choose better answer
        res = []
        bert_score = 1 - selfcheck_bertscore.predict(preds, [ref])
        return bert_score
        
    prefix = ['no_retrieval', 'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4']
    
    scores = [] # list of list, len == len(json_file)
    best = [] # list of options, len == len(json_file)
    best_pred = [] # list of predictions, len == len(json_file)
    performance = [] # list of scores, len == len(json_file)    
    question = []  # list of questions, len == len(json_file)
    golds = [] # list of gold answers, len == len(json_file)
    cnt = 0
    stage1_cnt = 0 # count of stage 1: direct judgment
    stage2_cnt = 0 # count of stage 2: confidence judgment
    stage3_cnt = 0 # count of stage 3: direct judgment w ref
    stage4_cnt = 0 # count of stage 4: no more ref, stop
    bad_pair_cnt = 0 # count of bad pairs
    all_rec = {'num_generations': [], 'cfd': [], 'score_after_cfd': [], 'judge_p': [], 'score_after_judge': [], 'score_after_voting': [], 'voting': [],
            'global_record': [], 'chosen_ans': [], 'performance': [] , 'possible_correct': [], 'question': [], 'golds': []} # record
    def load_jsonlines(file):
        with jsonlines.open(file, 'r') as jsonl_f:
            lst = [obj for obj in jsonl_f]
        return lst

    if closed:
        if task == 'arc':
            auxilliary_data_path = '/apdcephfs_qy3/share_4983883/ping_test/rag/eval_data/arc_challenge_processed.jsonl'
        elif task == 'health':
            auxilliary_data_path = '/apdcephfs_qy3/share_4983883/ping_test/rag/eval_data/health_claims_processed.jsonl'
        auxilliary_data = load_jsonlines(auxilliary_data_path)
        auxilliary_data = preprocess_input_data(auxilliary_data, task=task, is_llama3=is_llama3)
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
        all_rec['judge_p'].append([]) # list of list (round level)
        all_rec['score_after_judge'].append([]) # list of list (round level)
        all_rec['score_after_voting'].append([]) # list of list (round level)
        all_rec['voting'].append([]) # list of list 

        # 'global_record': [], 'chosen_ans': [], 'chosen_metric': []  are only one for each item
        for cur_pos in range(len(prefix)):
            ans_cfd = compute_confidence(item[prefix[cur_pos] + '_log_probs'])
            all_rec['cfd'][-1].append(ans_cfd) # append the cfd scores
        # award to the highest cfd
        max_cfd = max(all_rec['cfd'][-1])
        max_cfd_pos = all_rec['cfd'][-1].index(max_cfd)
        score_record[max_cfd_pos] += 1
        all_rec['score_after_cfd'][-1].append(copy.deepcopy(score_record)) # append the cfd scores

        # 3. Voting, 1 score
        tmp_bertscore = [0 for _ in range(len(prefix))]
        # quick check
        quick_check = False # quick check all answers vs current ref, if supported, +1 tmp score (remove after this round)
        if (closed and cur_pos < 1 and 'lama' in model_name) or (closed and 'selfrag' in model_name):
            for i_ref in range(len(prefix)):
                ref_ans = item[prefix[i_ref]]
                ref_norm = normalize_answer(ref_ans)
                tmp_tmp_bertscore = [0 for _ in range(len(prefix))]
                for i_ in range(len(prefix)):
                    cur_norm = normalize_answer(item[prefix[i_]])
                    try:
                        if cur_norm.split()[0] == ref_norm.split()[0] and i_ref != i_: 
                            tmp_bertscore[i_] += 1
                            tmp_tmp_bertscore[i_] += 1
                            quick_check = True
                    except:
                        pass
                if quick_check:
                    all_rec['voting'][-1].append(copy.deepcopy(tmp_tmp_bertscore))
              
        # use bertscore to choose better answer
        preds = [item[prefix[i_]] for i_ in range(len(prefix))] 
        if not quick_check:
            for i_ref in range(len(prefix)):
                ref_ans = item[prefix[i_ref]]
                bert_score = cal_bertscore(preds, ref_ans, closed=closed)
                bert_score[i_ref] = 0 # remove the self ref
                for ii, it in enumerate(bert_score):
                    if ii == i_ref:
                        # skip the self ref
                        continue
                    nearest_float = round(it, 1)
                    tmp_bertscore[ii] += nearest_float
                all_rec['voting'][-1].append(copy.deepcopy(bert_score.tolist()))

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
        print('Best (used ref):', prefix[best_pos])
        print(f'Task: {task} \nQuestion: {question[-1]} \nGold: {golds[-1]} \nBest pred {best_pred[-1]}\nBest: {best[-1]} Internal score: {scores[-1]} \nCurrent metric: {performance[-1]}')
        if len(possible_correct_pred) > 0:
            print('Possible correct pred:', possible_correct_pred[0])
        print('='*80)
        stage3_cnt += 1 # count of stage 3: direct judgment w ref

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

    # record the basic info (task, model, cfd, s_cfd, proceed_threshold)
    final_rec = {'basic info': f'Task: {task}, Model: {model_name}, cfd: {args.cfd}, s_cfd: {args.s_cfd}, proceed_threshold: {args.proceed_threshold}', 
                'results': all_rec}
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: self.default(value) if isinstance(value, (np.ndarray, dict)) else value for key, value in obj.items()}
            return super(NumpyEncoder, self).default(obj)
    
    # save the records
    with open(file_path.replace('.json', '-final_rec_rm_SR.json'), 'w') as f:
        json.dump(final_rec, f, cls=NumpyEncoder)


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