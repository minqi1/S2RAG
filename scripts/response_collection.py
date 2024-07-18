"""
This script is co-written by:

Contributors:
- Minqi Xiang (mx716@ic.ac.uk) Equally contributed
- Zihan Zhu (zcabhub@ucl.ac.uk) Equally contributed

Public Source: https://github.com/zhuzihan728/LLM_Adaptive_RAG
Private Source: https://github.com/minqi1/S2RAG
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import random
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
import string
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, preprocess_input_data, postprocess_ans 

seed = 123
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
def format_prompt_plain(prompt, evidences=None, instruction=None):
    if evidences is None:
        prompts = [prompt]
    else:
        ctxs = []
        ctxs.append('\n'.join(["{0}\n{1}\n".format(para["title"], para["text"]) for para in evidences]))
        for i in evidences:
            ctxs.append("{0}\n{1}\n".format(i["title"], i["text"]))
        prompts = ["{i}\n{p}".format(i=i, p=prompt) for i in ctxs]
    
    prompts = [[{"role": "user", "content": i}] for i in prompts]
    return prompts


def results_rec_func(preds, evidences):
    results = {}
    overall_scores = {}
    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text = pred.outputs[0].text
        pred_id_log_probs = pred.outputs[0].id_log_probs

        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred_id_log_probs), 1)
        final_score = np.exp(seq_score)
        overall_scores[p_idx] = {"final_score": final_score}
        results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score, "id_log_probs": pred_id_log_probs, "token_ids": pred_token_ids, "evidence": evidences[p_idx]}
    return results

def call_model_generate(prompt, evidences, model, rec_method):
    final_results = {}
    prompt_no_ret, prompt_with_ret = prompt[0], prompt[1]
    prompt_use_all_ret = [prompt_with_ret[0]]
    prompt_use_one_ret = prompt_with_ret[1:]
    final_results["prompts"] = {"no_retrieval": prompt_no_ret, "all_doc_retrieval": prompt_use_all_ret, "one_doc_retrieval": prompt_use_one_ret}
    
    preds = model.generate(prompt_no_ret+prompt_use_all_ret+prompt_use_one_ret)
    
    # index 0 is the no retrieval case
    final_results["no_retrieval"] = preds[0].outputs[0].text
    final_results["no_retrieval_ids"] = preds[0].outputs[0].token_ids
    final_results["no_retrieval_log_probs"] = preds[0].outputs[0].id_log_probs

    # index 1 is the all retrieval case
    final_results["all_doc_retrieval"] = preds[1].outputs[0].text
    final_results["all_doc_retrieval_ids"] = preds[1].outputs[0].token_ids
    final_results["all_doc_retrieval_log_probs"] = preds[1].outputs[0].id_log_probs

    # index 2: is the one retrieval case
    results = rec_method(preds[2:], evidences)

    # results for one doc retrieval
    final_results["retrieval_res"] = results    
    return final_results

def process_data_evidences(demonstration, top_n):
    """
    Quote from selfrag to ensure consistency
    """
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str) # input model path
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--ndocs', type=int, default=5)
    # Decoding hyperparams
    parser.add_argument("--use_default_prompt", action="store_true", help="use default prompt as selfrag")
    
    args = parser.parse_args()
    
    if 'arc' in args.input_file.lower():
        args.task = 'arc_c'
    elif 'health' in args.input_file.lower():
        args.task = 'fever'
    
    model_name_ = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    if args.task in TASK_INST:
        instruction = TASK_INST[args.task]
    else:
        instruction = None
        
    model_ = AutoModelForCausalLM.from_pretrained(model_name_, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_, padding_side="left")
    max_new_tokens = args.max_new_tokens
    model = MyModel(model_, tokenizer, max_new_tokens=max_new_tokens)
    rec_method = results_rec_func

    if not args.output_file:
        raise ValueError("Output file not specified")

    def generate(prompt, evidences):
        return call_model_generate(prompt, evidences=evidences, model=model, rec_method=rec_method)

    all_results = []
    final_results = {}
    final_results["dataset"] = args.input_file
    
    prompt_fn = None

    if args.use_default_prompt:
        print("Using default prompt")
        prompt_fn = format_prompt_plain
    else:
        raise ValueError("Prompt function not implemented")
    
    input_data = preprocess_input_data(input_data, task=args.task, is_llama3="Llama-3" in args.model_name)
   
    for i, row in tqdm(enumerate(input_data)):
        
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        
        chats_no_ret = prompt_fn(prompt=row['instruction'], evidences=None, instruction=instruction)
        chats_with_ret = prompt_fn(prompt=row['instruction'], evidences=evidences, instruction=instruction)
        
        prompt_no_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_no_ret]
        prompt_with_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_with_ret]
        
        if "Llama-3" in args.model_name:
            response_prefix = tokenizer.decode(128006) + tokenizer.decode(78191) + tokenizer.decode(128007) + tokenizer.decode(271)

            prompt_no_ret = [i + response_prefix for i in prompt_no_ret]
            prompt_with_ret = [i + response_prefix for i in prompt_with_ret]
        if args.task == "arc_c":
            prompt_no_ret = [i + 'The best answer choice is ' for i in prompt_no_ret]
            prompt_with_ret = [i + 'The best answer choice is ' for i in prompt_with_ret]
        if args.task == "fever":
            prompt_no_ret = [i + 'The claim is ' for i in prompt_no_ret]
            prompt_with_ret = [i + 'The claim is ' for i in prompt_with_ret] 
        res = generate([prompt_no_ret, prompt_with_ret], evidences)
        
        if 'id' in row:
            res['question_id'] = row['id']
        else:
            res['question_id'] = i # for pub health
    
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        
        res['gold'] = row["answers"]
        
        all_results.append(res)
        
        # print out first example
        if i == 0:
            print("=========Prompt no retrieval=========")
            print(prompt_no_ret[0])
            print("====Prompt with all doc retrieval====")
            print(prompt_with_ret[0])
            print("====Prompt with one doc retrieval====")
            for p in prompt_with_ret[1:]:
                print(p)
                print()
            
    final_results["results"] = all_results
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)
    

if __name__ == "__main__":
    main()