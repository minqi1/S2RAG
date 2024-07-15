import jsonlines
import json
import copy
import re

# Quote and modify from selfrag to ensure consistency

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

FEW_SHOT = {
"fever": "[INST]Kathy Niakan\nInstitute she is investigating the mechanisms of lineage specification in human embryos and stem cells. In February 2016 Niakan was given the go-ahead by the UK Human Fertilisation and Embryology Authority to genetically modify human embryos. The embryos were to be destroyed after seven days. She planned to use the CRISPR technique to answer questions like what genetic faults cause some women to miscarry, what causes infertility and what is crucial for a healthy embryo. In 2017 her lab published the first major study using CRISPR-Cas9 in human embryos in Nature, demonstrating that the transcription factor Oct4 is essential for\n\nIs the following statement correct or not? Say true if it's correct; otherwise say false.\n\n## Input:\n\nBritain gives scientist go-ahead to genetically modify human embryos.[/INST]true</s>",
"fever_no_ret" : "[INST]Is the following statement correct or not? Say true if it's correct; otherwise say false.\n\n## Input:\n\nBritain gives scientist go-ahead to genetically modify human embryos.[/INST]true</s>",
"arc_c": "[INST]History of Earth\nAnother 30 million years (310 Ma) saw the divergence of the synapsids (including mammals) from the sauropsids (including birds and reptiles). Other groups of organisms continued to evolve, and lines diverged—in fish, insects, bacteria, and so on—but less is known of the details. After yet another, the most severe extinction of the period (251~250 Ma), around 230 Ma, dinosaurs split off from their reptilian ancestors. The Triassic–Jurassic extinction event at 200 Ma spared many of the dinosaurs, and they soon became dominant among the vertebrates. Though some mammalian lines began to separate during this period, existing mammals were probably small\n\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\nWhich of the following animals have been on Earth for the longest period of time?\nA: Humans\nB: Birds\nC: Fish\nD: Reptiles[/INST]The best option is C</s>",
"arc_c_no_ret": "[INST]Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\nWhich of the following animals have been on Earth for the longest period of time?\nA: Humans\nB: Birds\nC: Fish\nD: Reptiles[/INST]The best option is C</s>"
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
