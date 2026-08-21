import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import bitsandbytes as bnb
import time
from torch.utils.data import DataLoader
import math
from typing import List, Tuple
from tqdm import tqdm

class BnBModelEvaluator:
    def __init__(self, model_name: str = 'gpt2-medium'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Keep original model for comparison
        self.original_model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
    def load_evaluation_data(self, dataset_name: str = 'wikitext', split: str = 'test', num_samples: int = 3000) -> List[str]:
        """Load evaluation dataset."""
        if dataset_name == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        elif dataset_name == 'ptb':
            dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        
        texts = [text for text in dataset['text'] if text.strip()][:num_samples]
        return texts

    def measure_inference_latency(self, model: nn.Module, input_text: str, num_runs: int = 100) -> float:
        """Measure average inference latency."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Warm-up run
        with torch.no_grad():
            model(**inputs)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                model(**inputs)
        end_time = time.time()
        
        return (end_time - start_time) / num_runs

    def compute_perplexity(self, model: nn.Module, texts: List[str], batch_size: int = 8) -> float:
        """Compute perplexity on given texts."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
                batch_texts = texts[i:i + batch_size]
                encodings = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                         max_length=512, return_tensors="pt").to(self.device)
                
                labels = encodings.input_ids.clone()
                outputs = model(**encodings, labels=labels)
                
                loss = outputs.loss * labels.ne(self.tokenizer.pad_token_id).sum()
                total_loss += loss.item()
                total_tokens += labels.ne(self.tokenizer.pad_token_id).sum().item()
        
        return math.exp(total_loss / total_tokens)

    def get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 / 1024  # Convert to MB

    def quantize_model_8bit(self) -> nn.Module:
        """Quantize model to 8-bit using bitsandbytes."""
        model_8bit = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map='auto'
        )
        return model_8bit

    def quantize_model_4bit(self) -> nn.Module:
        """Quantize model to 4-bit using bitsandbytes."""
        model_4bit = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            device_map='auto'
        )
        return model_4bit

    def quantize_model_nf4(self) -> nn.Module:
        """Quantize model to NF4 format using bitsandbytes."""
        model_nf4 = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            device_map='auto'
        )
        return model_nf4

    def evaluate_all_quantization_methods(self):
        """Evaluate all quantization methods and compare results."""
        eval_texts = self.load_evaluation_data(num_samples=3000)
        sample_text = "The quick brown fox jumps over the lazy dog."
        
        # Original model metrics
        original_size = self.get_model_size(self.original_model)
        original_latency = self.measure_inference_latency(self.original_model, sample_text)
        original_perplexity = self.compute_perplexity(self.original_model, eval_texts)
        
        results = {
            'original': {
                'size': original_size,
                'latency': original_latency,
                'perplexity': original_perplexity
            }
        }
        
        # Evaluate 8-bit quantization
        print("\nEvaluating 8-bit quantization...")
        model_8bit = self.quantize_model_8bit()
        results['8bit'] = {
            'size': self.get_model_size(model_8bit),
            'latency': self.measure_inference_latency(model_8bit, sample_text),
            'perplexity': self.compute_perplexity(model_8bit, eval_texts)
        }
        del model_8bit
        torch.cuda.empty_cache()
        
        # Evaluate 4-bit quantization
        print("\nEvaluating 4-bit quantization...")
        model_4bit = self.quantize_model_4bit()
        results['4bit'] = {
            'size': self.get_model_size(model_4bit),
            'latency': self.measure_inference_latency(model_4bit, sample_text),
            'perplexity': self.compute_perplexity(model_4bit, eval_texts)
        }
        del model_4bit
        torch.cuda.empty_cache()
        
        # Evaluate NF4 quantization
        print("\nEvaluating NF4 quantization...")
        model_nf4 = self.quantize_model_nf4()
        results['nf4'] = {
            'size': self.get_model_size(model_nf4),
            'latency': self.measure_inference_latency(model_nf4, sample_text),
            'perplexity': self.compute_perplexity(model_nf4, eval_texts)
        }
        del model_nf4
        torch.cuda.empty_cache()
        
        return results

def run_bnb_evaluation():
    print("Starting BitsAndBytes quantization evaluation...")
    evaluator = BnBModelEvaluator('gpt2-medium')
    results = evaluator.evaluate_all_quantization_methods()
    
    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    for method, metrics in results.items():
        print(f"\n{method.upper()} Quantization:")
        print(f"Model Size: {metrics['size']:.2f} MB")
        print(f"Inference Latency: {metrics['latency']*1000:.2f}ms")
        print(f"Perplexity: {metrics['perplexity']:.2f}")
        if method != 'original':
            size_reduction = (results['original']['size'] - metrics['size']) / results['original']['size'] * 100
            latency_change = (metrics['latency'] - results['original']['latency']) / results['original']['latency'] * 100
            perplexity_change = (metrics['perplexity'] - results['original']['perplexity']) / results['original']['perplexity'] * 100
            print(f"Size Reduction: {size_reduction:.1f}%")
            print(f"Latency Change: {latency_change:+.1f}%")
            print(f"Perplexity Change: {perplexity_change:+.1f}%")

if __name__ == '__main__':
    run_bnb_evaluation()