#!/usr/bin/python
# -*- coding: UTF-8 -*-
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForCausalLM
from tsfm_wrapper import MyModel
import random
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
import string
import textwrap
from colorama import Fore, Style, init
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, SelfCheckBERTScore, preprocess_input_data, postprocess_ans, \
    compute_confidence, get_retrieval_p_compare, cal_bertscore, choose_better_prompt, choose_better_prompt_fantasy
from peft import PeftModel, LoraConfig


seed = 123
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
def format_prompt_plain(prompt, evidences=None):
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

def format_float(number):
    return f"{number:.3f}"

def wrap_text_with_hash(text, width=80):
    wrapped_text = textwrap.fill(text, width=width)
    return '\n'.join([f'#   {line}' for line in wrapped_text.split('\n')])

def s2rag_gen(prompt, evidences, model, tokenizer, args, cur_question, response_prefix, selfcheck_bertscore):
    print("\n======================================================================================================================")
    print(f"{Fore.YELLOW}QUERY: {cur_question.strip()}{Style.RESET_ALL}")
    print("======================================================================================================================\n")
    model_name = args.model_name
    final_results = {}
    prompt_no_ret, prompt_with_ret = prompt[0], prompt[1]
    prompt_use_one_ret = prompt_with_ret[1:]
    final_results["prompts"] = {"no_retrieval": prompt_no_ret, "one_doc_retrieval": prompt_use_one_ret}
    
    # start S2RAG
    preds = model.generate(prompt_no_ret + [prompt_use_one_ret[0]]) # a_0, a_1
    score_record = [0 for _ in range(len(prompt_use_one_ret)+ 1)]
    cur_pos = 0
    ans1_cfd = compute_confidence(preds[cur_pos].outputs[0].id_log_probs)
    print(f"-------------------------------------------------Candidate Answer {0}-------------------------------------------------")
    print(f'#   Candidate answer: {cur_pos} | Confidence: {format_float(ans1_cfd)} \n#   ', preds[cur_pos].outputs[0].text.replace('assistant', '').strip())
    print(f'#   - Evidence: {None}')

    while True and cur_pos <= len(prompt_use_one_ret) - 1:
        ans1_cfd = compute_confidence(preds[cur_pos].outputs[0].id_log_probs)
        ans2_cfd = compute_confidence(preds[cur_pos+1].outputs[0].id_log_probs)

        print(f"-------------------------------------------------Candidate Answer {cur_pos+1}-------------------------------------------------")
        print(f'#   Candidate answer: {cur_pos + 1} | Confidence: {format_float(ans2_cfd)}  \n#   ', preds[cur_pos + 1].outputs[0].text.replace('assistant', '').strip())
        evidence_text = wrap_text_with_hash('- Evidence: ' + evidences[cur_pos]["title"] + ": " + evidences[cur_pos]["text"].replace("\n", " "))
        print(evidence_text)

        # 1. score from confidence: noticeable difference 1 score
        if cur_pos != 0: # fair comparison
            if ans1_cfd > args.s_cfd * ans2_cfd:
                score_record[cur_pos] += 1
            elif ans2_cfd > args.s_cfd  * ans1_cfd:
                score_record[cur_pos + 1] += 1
        else: 
            if ans1_cfd > args.cfd * ans2_cfd:
                score_record[cur_pos] += 1 
            elif ans2_cfd > ans1_cfd / args.cfd * args.s_cfd:
                score_record[cur_pos + 1] += 1

        # 2. score from direct judgement (w/o ref) 1 score
        if cur_pos > 0:
            cur_ctx = evidences[cur_pos-1]['title'] + '\n'+ evidences[cur_pos-1]['text'] + '\n' + evidences[cur_pos]['title'] + '\n' + evidences[cur_pos]['text']
        else:
            cur_ctx = evidences[cur_pos]['title'] + '\n' + evidences[cur_pos]['text']# no retrieval does not have ctx
        
        both_good = False
        retrieve_p = [0, 0, 0, 0]
        a = preds[cur_pos].outputs[0].text.replace('assistant', '').strip()
        b = preds[cur_pos + 1].outputs[0].text.replace('assistant', '').strip()

        sys = None
        if args.demo_task == 'fantasy':
            prompt = choose_better_prompt_fantasy(context=cur_ctx, a=a, cfd_a=ans1_cfd, b=b, cfd_b=ans2_cfd, question=cur_question)
        else:
            prompt = choose_better_prompt(context=cur_ctx, a=a, cfd_a=ans1_cfd, b=b, cfd_b=ans2_cfd, question=cur_question)
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
        
        cur_pred = model.generate([prompt_fmt], max_new_tokens=5)

        # update score using the logits
        retrieve_p, retrieve_p_hard, has_judgment = get_retrieval_p_compare(cur_pred[0], tokenizer)
        # print(f"-------------------------------------------------Judgement on Ans {cur_pos} and Ans {cur_pos+1}-------------------------------------------------")
        retrieve_p = [round(it, 3) for it in retrieve_p]
        print(f'\n{Fore.MAGENTA}Judgement on Ans {cur_pos} and Ans {cur_pos+1}: {retrieve_p} \n{Style.RESET_ALL}') 

        abs_diff = abs(retrieve_p[0] - retrieve_p[1]) # absolute difference between two answers
        max_pos = retrieve_p.index(max(retrieve_p))
        if ((abs_diff < args.tau and retrieve_p[2] < args.tau) or max_pos==3) and cur_pos < len(prompt_use_one_ret) + 1: # proceed to the next pair
            score_record[cur_pos] += max(retrieve_p[0] - retrieve_p[1], 0)
            score_record[cur_pos + 1] += max(retrieve_p[1] - retrieve_p[0], 0)
            cur_pos += 1 # move to the next pair: ret_0 and ret_1...
            # generate the next answer 
            if cur_pos < len(prompt_use_one_ret):
                new_pred = model.generate(prompt_use_one_ret[cur_pos])
                preds.extend(new_pred) 
                continue
            else:
                score_record = [round(it, 2) for it in score_record]
                print("====================================================Final Answer====================================================")
                print(f'Answer Score: {score_record}')
                print(f'{Fore.RED}NO FINAL ANSWER FOUND SINCE ALL ANSWERS ARE BAD!{Style.RESET_ALL}')
                break
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
            if cur_pos >= len(prompt_use_one_ret) + 1:
                pass
            else:
                raise ValueError('Invalid judgment!')                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        max_score, second_max = sorted(score_record)[-1], sorted(score_record)[-2]
        cur_score_diff = abs(max_score - second_max)
        if (score_record.count(max_score) == 1 and cur_score_diff > args.tau) or both_good: # only best answer is noticeable better, OR first two are equally good
            best_ans = preds[score_record.index(max_score)].outputs[0].text
            # print("\n====================================================Answer Score====================================================")
            score_record = [round(it, 2) for it in score_record]
            best_pos = score_record.index(max_score)
            print("======================================================================================================================")
            print(f'{Fore.LIGHTBLUE_EX}Answer Score: {score_record}{Style.RESET_ALL}' + f' | Candidate Answer {best_pos} is the best.')
            print(f"   {Fore.GREEN}===>>>{Style.RESET_ALL}")
            print(f"{Fore.LIGHTGREEN_EX}FINAL ANSWER: {best_ans}{Style.RESET_ALL}")
            print("======================================================================================================================")
            break

        # 3. Voting, 1 score
        tmp_bertscore = [0 for _ in range(len(prompt_use_one_ret) + 1)]
        if cur_pos < len(prompt_use_one_ret) + 1: # not the last pair                
            # use bertscore to choose better answer
            bert_preds = [preds[i].outputs[0].text for i in range(cur_pos + 2)]
            for i_ref in range(cur_pos + 3):
                # ref_ans = item[prefix[i_ref]]
                ref_ans = model.generate(prompt_use_one_ret[cur_pos + 1])[0].outputs[0].text
                bert_score = cal_bertscore(bert_preds, ref_ans, selfcheck_bertscore)
                if i_ref < len(bert_score):
                    bert_score[i_ref] = 0 # remove the ref answer
                for ii, it in enumerate(bert_score):
                    nearest_float = round(it, 1)
                    tmp_bertscore[ii] += nearest_float
        # normalize the tmp_bertscore (norm to 1)
        tmp_bertscore = [it/(cur_pos + 3) for it in tmp_bertscore] # avg the bertscore
        score_record_bert = [score_record[ii] + tmp_bertscore[ii] for ii in range(len(prompt_use_one_ret) + 1)]
        print("-------------------------------------------------Voting Score-------------------------------------------------")
        print(f'Voting score: {tmp_bertscore}')

        # output the best one
        max_score = max(score_record_bert)
        best_pos = score_record_bert.index(max_score) # find the (first) best one including the bertscor
        best_ans = preds[best_pos].outputs[0].text
        
        score_record_bert = [round(it, 2) for it in score_record_bert]
        print("======================================================================================================================")
        print(f'{Fore.LIGHTBLUE_EX}Answer Score: {score_record}{Style.RESET_ALL}' + f' | Candidate Answer {best_pos} is the best.')
        print(f"   {Fore.GREEN}===>>>{Style.RESET_ALL}")
        print(f"{Fore.LIGHTGREEN_EX}FINAL ANSWER: {best_ans}{Style.RESET_ALL}")
        print("======================================================================================================================")
        break
        

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
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") # input model path
    parser.add_argument('--input_q_ctx', type=str) # input query and ctx {question: "xxx", ctxs: [{title: "xxx", text: "xxx"}]} or path to json file
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--demo_task', type=str, default='fact') # fact or fantacy
    # Decoding hyperparams
    parser.add_argument("--use_default_prompt", action="store_true", help="use default prompt as selfrag")
    
    # s2rag
    parser.add_argument("--cfd", type=float, default=0.95, help="Confidence threshold for no ret")
    parser.add_argument("--s_cfd", type=float, default=1.1, help="Cfd scale for rest")
    parser.add_argument("--tau", type=float, default=0.45, help="Cautiousness") #
    parser.add_argument("--use_trained", action="store_true", help="Use trained model")
    parser.add_argument("--trained_path", type=str, default="weights_minqi/llama3_8B_256", help="Trained model adapter path")
    parser.add_argument("--bertmodel", type=str, default=None, help="BERT model path")
    # for demo purpose, we use the same model for both base and judge (so that only one model is loaded)
    parser.add_argument("--base", type=str, default=None, help="Base dir") # base generator

    args = parser.parse_args()
    
    model_name = args.model_name
    input_q_ctx = args.input_q_ctx

    en_core_web_sm_path = args.bertmodel
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True, model_path=en_core_web_sm_path) 

    if input_q_ctx.endswith(".json"):
        input_data = json.load(open(input_q_ctx))
    else:
        input_data = load_jsonlines(input_q_ctx)

    if args.use_trained: # use trained judge model
        base_dir = args.trained_path
        tokenizer = AutoTokenizer.from_pretrained(base_dir, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, base_dir)
        print("Use trained model as judge.")
    else: # use untrained model as judge
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        print("Use untrained model as judge.")

    max_new_tokens = args.max_new_tokens
    model = MyModel(model, tokenizer, max_new_tokens=max_new_tokens)

    if args.base is None:
        args.base = model_name
    else:
        if 'lama2' in args.base and '7' in args.base:
            args.base = 'Llama-2-7b'
        elif 'lama3' in args.base:
            args.base = 'Llama-3'
    
    prompt_fn = None

    if args.use_default_prompt:
        # print("Using default prompt")
        prompt_fn = format_prompt_plain
    else:
        raise ValueError("Prompt function not implemented")
    
    def preprocess_input_data_demo(dataset):
        if type(dataset) == dict:
            dataset = [dataset]
        new_data = []
        if args.demo_task == "fact":
            # instruction = "You are asked to provide a fact-based answer to the following question. Please provide a short answer (within 5 words), it is better to be supported by the context. If you do not know the answer, respond with: I do not know."
            instruction = "You are asked to provide a fact-based answer to the following question. Please provide a short answer (within 5 words), it is better to be supported by the context."

        elif args.demo_task == "fantasy":
            instruction = """
            You are asked to provide a fantasy-based answer to the following question. Please provide a very short and concise answer within 10 words. 

            Example:
            Question: What is the name of the legendary sword in the story?
            Context: ["The legendary sword, Excalibur, is known for its magical properties.", "King Arthur wielded the sword Excalibur in many battles."]
            Answer: Excalibur
            """        
        else:
            instruction = None

        for item in dataset:
            item["instruction"] = item["question"]
            item["instruction"] = instruction + "\n\n" + item["instruction"] if instruction is not None else item["instruction"]
            new_data.append(item)
        return new_data
    
    input_data = preprocess_input_data_demo(input_data)
   
    for i, row in enumerate(input_data):
        if i > 0:
            input(f"{Fore.MAGENTA}\nPress Enter to continue...{Style.RESET_ALL}")
            print('\n\n')
        _, evidences = process_data_evidences(row, top_n=5)
        
        chats_no_ret = prompt_fn(prompt=row['instruction'], evidences=None)
        chats_with_ret = prompt_fn(prompt=row['instruction'], evidences=evidences)
        
        prompt_no_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_no_ret]
        prompt_with_ret = [tokenizer.apply_chat_template(i, tokenize=False) for i in chats_with_ret]

        response_prefix = ''
        if "Llama-3" in args.model_name:
            response_prefix = tokenizer.decode(128006) + tokenizer.decode(78191) + tokenizer.decode(128007) + tokenizer.decode(271)
            prompt_no_ret = [i + response_prefix for i in prompt_no_ret]
            prompt_with_ret = [i + response_prefix for i in prompt_with_ret]

        s2rag_gen([prompt_no_ret, prompt_with_ret], evidences=evidences, model=model, tokenizer=tokenizer, \
                args=args, cur_question=row['question'], \
                    response_prefix=response_prefix, selfcheck_bertscore=selfcheck_bertscore)

if __name__ == "__main__":
    main()