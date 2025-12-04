import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def GPT4(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
MODEL_ID = "/home/a00164/oscar50513.ii13/mg/genai/cache_models/Llama-2-13b-chat-hf" 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()


import spacy
nlp = spacy.load("en_core_web_sm")

def _split_regional_prompts(regional_prompt: str):
    return [p.strip() for p in regional_prompt.split("BREAK") if p.strip()]

def _extract_keywords(text: str):
    doc = nlp(text)
    # 名詞/形容詞的 lemma，避開 stop words，順便把名詞片語的 head 也納入
    lemmas = {t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "ADJ"} and not t.is_stop}
    chunks = {chunk.root.lemma_.lower() for chunk in doc.noun_chunks}
    return {w for w in (lemmas | chunks) if len(w) > 1}


def _extract_bindings(text: str):
    doc = nlp(text)
    bindings = set()
    for chunk in doc.noun_chunks:
        noun = chunk.root.lemma_.lower()
        adjs = [t.lemma_.lower() for t in chunk if t.pos_ == "ADJ" and not t.is_stop]
        for adj in adjs:
            if len(adj) > 1 and len(noun) > 1:
                bindings.add((adj, noun))
    return bindings


def _binding_hit(binding, text: str, max_gap: int = 3):
    adj, noun = binding
    doc = nlp(text)
    lemmas = [t.lemma_.lower() for t in doc]
    for i in range(len(lemmas) - 1):
        if lemmas[i] == adj and lemmas[i + 1] == noun:
            return True
    for sent in doc.sents:
        s = [t.lemma_.lower() for t in sent]
        if adj in s and noun in s:
            if abs(s.index(adj) - s.index(noun)) <= max_gap:
                return True
    return False


def _binding_missing(required_bindings, regional_prompt: str):
    regional_prompts = _split_regional_prompts(regional_prompt)
    missing = []
    for binding in required_bindings:
        if not any(_binding_hit(binding, rp) for rp in regional_prompts):
            missing.append(binding)
    return missing


def _coverage_check(user_prompt: str, regional_prompt: str):
    gold = _extract_keywords(user_prompt)
    generated = set()
    for rp in _split_regional_prompts(regional_prompt):
        generated |= _extract_keywords(rp)
    missing = gold - generated
    return missing, gold, generated


def local_llm(prompt, missing_kw=None, missing_bind=None):
    with open('template/template.txt', 'r') as f:
        template = f.readlines()

    feedback = ""
    if missing_kw:
        miss_text = ", ".join(sorted(missing_kw))
        feedback += (
            f"\nYou must explicitly include these keywords in the plan: {miss_text}."
            " Do not change the Final split ratio format. Keep Regional Prompt in the same schema."
        )
    if missing_bind:
        bind_text = "; ".join([f"{a} {n}" for a, n in missing_bind])
        feedback += (
            f"\nYou must keep these adjective-noun bindings together: {bind_text}."
            " Do not change the Final split ratio format. Keep Regional Prompt in the same schema."
        )

    user_textprompt = f"Caption:{prompt}\nLet's think step by step:{feedback}"
    textprompt = f"{' '.join(template)} \n {user_textprompt}"
    inputs = tokenizer(textprompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        res = model.generate(**inputs, max_new_tokens=2048)[0]
    output = tokenizer.decode(res, skip_special_tokens=True).replace(textprompt, "")
    return get_params_dict(output, fallback_prompt=prompt)  # 把原始 prompt 傳進去做備援
    
def guarded_plan(prompt: str, max_retry: int = 3):
    required_bindings = _extract_bindings(prompt)
    missing_kw = None
    missing_bind = None
    last_plan = None
    for attempt in range(max_retry):
        plan = local_llm(prompt, missing_kw=missing_kw, missing_bind=missing_bind)
        last_plan = plan  # 先記起來
        missing_kw, _, _ = _coverage_check(prompt, plan["Regional Prompt"])
        missing_bind = _binding_missing(required_bindings, plan["Regional Prompt"])
        if not missing_kw and not missing_bind:
            print("------------------plan PASS -------------------")
            return plan  # PASS
        print(
            f"[plan guard] retry {attempt+1}/{max_retry}, "
            f"missing keywords: {sorted(missing_kw)}; "
            f"missing bindings: {[f'{a} {n}' for a, n in missing_bind]}"
        )
    # 超過上限仍缺詞或缺綁定，直接用最後一次的結果，避免無窮迴圈
    print(
        f"[plan guard] still missing after {max_retry} retries, use last plan as-is: "
        f"missing keywords {sorted(missing_kw)}, "
        f"missing bindings {[f'{a} {n}' for a, n in missing_bind]}"
    )
    return last_plan


# def local_llm(prompt, version=None, model_path=None, hf_token=None):
#     model_id = model_path or "meta-llama/Llama-2-13b-chat-hf"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_id,
#         use_fast=False,          # Llama 通常需 slow tokenizer
#         token=hf_token,          # 若是 gated/private 模型需帶 token
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#         token=hf_token,
#     )
#     with open('template/template.txt', 'r') as f:
#         template=f.readlines()
#     user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
#     textprompt= f"{' '.join(template)} \n {user_textprompt}"
#     model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
#     model.eval()
#     with torch.no_grad():
#         print('waiting for LLM response')
#         res = model.generate(**model_input, max_new_tokens=4096)[0]
#         output=tokenizer.decode(res, skip_special_tokens=True)
#         output = output.replace(textprompt,'')
#     # print(f"local_LLM output:{output}")
#     return get_params_dict(output)

def get_params_dict(output_text, fallback_prompt):
    response = output_text
    split_ratio_match = re.search(r"Final split ratio[:：]\s*([\d.,;]+)", response)
    prompt_match = re.search(r"Regional Prompt[:：]\s*(.*?)(?=\n\n|\Z)", response, re.DOTALL)

    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1).strip()
        if final_split_ratio.replace(" ","")=="0,0":
            final_split_ratio="0.5,0.5"
        elif final_split_ratio.replace(" ","")=="0":
            final_split_ratio="0.5,0.5"
        print("Final split ratio:", final_split_ratio)
    else:
        final_split_ratio = "1"  # 預設單區塊
        print("Final split ratio not found, fallback to 1.")

    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        regional_prompt = fallback_prompt  # 用原始 caption 當區域描述
        print("Regional Prompt not found, fallback to original prompt.")

    return {"Final split ratio": final_split_ratio, "Regional Prompt": regional_prompt}

# def get_params_dict(output_text):
#     response = output_text
#     # Find Final split ratio
#     split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
#     if split_ratio_match:
#         final_split_ratio = split_ratio_match.group(1)
#         print("Final split ratio:", final_split_ratio)
#     else:
#         print("Final split ratio not found.")
#     # Find Regioanl Prompt
#     prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
#     if prompt_match:
#         regional_prompt = prompt_match.group(1).strip()
#         print("Regional Prompt:", regional_prompt)
#     else:
#         print("Regional Prompt not found.")

#     image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
#     return image_region_dict
