from transformers import pipeline
import torch
from datasets import load_dataset
import json
import os
import random
import numpy as np
from retriever import Retriever
from tqdm import tqdm
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot learning for Question Answering')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                      help='Name of the model to use')
    parser.add_argument('--num_shots', type=int, default=0,
                      help='Number of shots for few-shot learning')
    return parser.parse_args()

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def create_base_prompt(context, question):
    return f"""You are a precise question-answering assistant. Given the context below, answer the question by extracting the minimal span of text that completely answers the question. Only include the exact answer text, with no additional explanation.

Context: {context}

Question: {question}

Answer: """

def create_few_shot_prompt(retrieved_examples, context, question):
    return f"""You are a precise question-answering assistant. Answer the questions based on the given contexts. Only include the exact answer text, with no additional explanation.

{retrieved_examples}Context: {context}

Question: {question}

Answer: """

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    
    fix_seed(42)
    
    retriever = Retriever()
    
    print(f"Loading model: {args.model_name}")
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",)
    
    train_ds = load_dataset("rajpurkar/squad", split="train")
    eval_ds = load_dataset("rajpurkar/squad", split="validation")
    
    train_indices = torch.randperm(len(train_ds))[:50].tolist()
    demo_ds = train_ds.select(train_indices)
    
    print("Generating answers for demonstration examples...")
    DEMO_BATCH_SIZE = 128
    
    for i in tqdm(range(0, len(demo_ds), DEMO_BATCH_SIZE), desc="Processing demonstration examples"):
        batch_indices = list(range(i, min(i + DEMO_BATCH_SIZE, len(demo_ds))))
        batch = demo_ds[i:i + DEMO_BATCH_SIZE]
        
        prompts = [
            create_base_prompt(
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
            
            example = {
                'context': batch['context'][j],
                'question': batch['question'][j],
                'answer': answer
            }
            retriever.add_example(example)
    
    print("Finished generating demonstration answers")
    
    eval_indices = torch.randperm(len(eval_ds))[:500].tolist()
    sampled_ds = eval_ds.select(eval_indices)
    
    BATCH_SIZE = 256
    predictions = {}
    no_answer_probs = {}
    
    for i in tqdm(range(0, len(sampled_ds), BATCH_SIZE), desc="Processing evaluation examples"):
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(sampled_ds))))
        batch = sampled_ds[i:i + BATCH_SIZE]
        
        prompts = []
        for j in range(len(batch_indices)):
            query = {
                'context': batch['context'][j],
                'question': batch['question'][j]
            }
            
          
            retrieved = retriever.retrieve_topk(query, k=args.num_shots)
            
                
            retrieved_text = "\n\n".join(retrieved) + "\n\n" if retrieved else ""
            
            prompt = create_few_shot_prompt(
                retrieved_examples=retrieved_text,
                context=batch['context'][j],
                question=batch['question'][j]
            )
            prompts.append(prompt)
        
        if not prompts:
            continue
            
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
    
    print("Saving results...")
    
    model_name = args.model_name.split('/')[-1]  
    output_dir = f"{model_name}_{args.num_shots}shots_topk"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f)
    
    with open(os.path.join(output_dir, 'na_prob.json'), 'w') as f:
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
    
    with open(os.path.join(output_dir, 'eval_dataset.json'), 'w') as f:
        json.dump(eval_dataset, f)
    
    print(f"Done! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
