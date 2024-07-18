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
from utils import TASK_INST, load_jsonlines
import jsonlines
import random 
import copy
from accelerate import Accelerator
from accelerate.utils import gather_object

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name") # we only use Llama-3 for explanation collection
parser.add_argument("--input_file", type=str, default="/home/minqi/code/S2RAG/data_training/train_data_fusion.jsonl", help="Input file")
parser.add_argument("--output_file", type=str, default="/home/minqi/code/S2RAG/data_training/train_data_fusion_w_exp.jsonl", help="Output file")
args = parser.parse_args()
batch_size = args.batch_size 

accelerator = Accelerator()
accelerator.wait_for_everyone()

model_name = args.model_name

num_gpus = torch.cuda.device_count()
model_ = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": accelerator.process_index}, torch_dtype=torch.bfloat16) # do not use "auto" as model is shred, which makes inference slow.
 
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = MyModel(model_, tokenizer, max_new_tokens=60)

# Prepare the model with the accelerator
model = accelerator.prepare(model)
print(accelerator.process_index)

# load data
writing_path =  '/'.join(os.getcwd().split('/')[:-1])
file = args.input_file
input_data = load_jsonlines(file) # list of dict


with accelerator.split_between_processes(input_data) as input_data_split:
    input_data = input_data_split
    all_results = []
    for start_idx in tqdm(range(0, len(input_data), batch_size)):
        batch = input_data[start_idx:start_idx+batch_size]
        new_instructions = [row['new_instruction'] for row in batch]
        results = model.generate(new_instructions, temperature=0.7, do_sample=True) # list
        preds = [pred.outputs[0].text for pred in results]
        expls = [pred.replace('assistant', '') for pred in preds]
        train_instructions = [row['instruction'] + '\nJudgement: '  for row in batch]
        train_outputs = [batch[idx]['output'].strip() + '\n' + 'Explanation: ' + expls[idx].strip() for idx in range(len(batch))]
        tmp_dict = [{'instruction': train_instruction, 'output': train_output} for train_instruction, train_output in zip(train_instructions, train_outputs)]
        # print every 10 batches
        if start_idx % 10 == 0:
            print('='*80)
            print('Example train instruction: ', tmp_dict[0]['instruction'])
            print('Example train output: ', tmp_dict[0]['output'])
            print('='*80)
        all_results.extend(tmp_dict)

# save results
results_gathered = gather_object(all_results)
if accelerator.is_main_process:
    with jsonlines.open(args.output_file, 'w') as writer:
        for row in results_gathered:
            writer.write(row)

    print(f'Results saved to {args.output_file}!')