import torch
from sentence_transformers import SentenceTransformer
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Retriever:
    def __init__(self):
        self.memory = []
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    @staticmethod
    def normalize(x):
        """Simpler normalization using min-max scaling"""
        x_min = torch.min(x)
        x_max = torch.max(x)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def build_prompt(self, query, k=4, method='topk'):
        """Build prompt with retrieved examples using specified method"""
        if method == 'topk':
            retrieved = self.retrieve_topk(query, k)
        elif method == 'random':
            retrieved = self.retrieve_random(k)
        else:
            raise ValueError("Invalid retrieval method. Use 'topk' or 'random'.")
            
        context = "\n\n".join(retrieved) + "\n\n" if retrieved else ""
        return context + query['question']

    def retrieve_topk(self, query, k):
        """Retrieve k most similar examples based on embedding similarity"""
        k = min(len(self.memory), k)
        if k == 0:
            return []
            
        query_text = query['context'] + " " + query['question']
        query_embedding = torch.tensor(self.model.encode(query_text))
        
        memory_embeddings = torch.stack([item['embedding'] for item in self.memory], dim=0)
        
        similarity_scores = torch.cosine_similarity(memory_embeddings, query_embedding, dim=-1)
        sim_normalized = self.normalize(similarity_scores)
        
        _, indices = torch.topk(sim_normalized, k, largest=True)
        indices = indices.tolist()
        indices.reverse()
        
        return [self.format_example(self.memory[i]) for i in indices]

    def format_example(self, example):
        return f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}"

    def add_example(self, example):
        text = example['context'] + " " + example['question']
        embedding = torch.tensor(self.model.encode(text))
        
        self.memory.append({
            'context': example['context'],
            'question': example['question'],
            'answer': example['answer'],  
            'embedding': embedding
        })
