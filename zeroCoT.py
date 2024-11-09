from transformers import pipeline
import torch
from datasets import load_dataset
import json
import os
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

fix_seed(42)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",)

ds = load_dataset("rajpurkar/squad", split="validation")

# randomly sample 100 examples
sampled_indices = torch.randperm(len(ds))[:500].tolist()
sampled_ds = ds.select(sampled_indices)

BATCH_SIZE = 128

def create_prompt(context, question):
    return f"""You are a precise question-answering assistant. Given the context below, answer the question by extracting the minimal span of text that completely answers the question. Only include the exact answer text, with no additional explanation.

Context: {context}

Question: {question}

Let's think step by step:

Answer: """

predictions = {}
no_answer_probs = {} 

for i in range(0, len(sampled_ds), BATCH_SIZE):
    batch_indices = list(range(i, min(i + BATCH_SIZE, len(sampled_ds))))
    batch = sampled_ds[i:i + BATCH_SIZE]
    
    prompts = [
        create_prompt(
            context=batch['context'][j],
            question=batch['question'][j]
        ) for j in range(len(batch_indices))
    ]
    
    outputs = pipe(
        prompts,
        max_new_tokens=50,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )
    
    for j, generation in enumerate(outputs):
        full_response = generation[0]['generated_text']
        answer = full_response[len(prompts[j]):].strip()
        
        question_id = batch['id'][j]
        predictions[question_id] = answer
        no_answer_probs[question_id] = 0.0  
        
with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

with open('na_prob.json', 'w') as f:
    json.dump(no_answer_probs, f)

eval_dataset = {'data': [], 'version': 'v2.0'}
current_article = {'title': 'sampled_data', 'paragraphs': []}

for idx in range(len(sampled_ds)):
    paragraph = {
        'context': sampled_ds[idx]['context'],
        'qas': [{
            'answers': [{'answer_start': ans_start, 'text': text} 
                       for ans_start, text in zip(sampled_ds[idx]['answers']['answer_start'], 
                                                sampled_ds[idx]['answers']['text'])],
            'question': sampled_ds[idx]['question'],
            'id': sampled_ds[idx]['id']
        }]
    }
    current_article['paragraphs'].append(paragraph)

eval_dataset['data'].append(current_article)

with open('eval_dataset.json', 'w') as f:
    json.dump(eval_dataset, f)
