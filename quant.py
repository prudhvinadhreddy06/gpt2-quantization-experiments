import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
import copy
from datasets import load_dataset
import time
from torch.utils.data import DataLoader
import math
from tqdm import tqdm

class ModelQuantizer:
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.model = copy.deepcopy(model)
        self.quantization_stats = defaultdict(dict)
        
    def _calculate_scaling_factor(self, tensor: torch.Tensor, num_bits: int = 8) -> float:
        max_val = torch.max(torch.abs(tensor))
        return max_val / (2 ** (num_bits - 1) - 1)

    def _quantize_tensor(self, 
                        tensor: torch.Tensor, 
                        num_bits: int = 8, 
                        scaling_factor: Optional[float] = None) -> torch.Tensor:
        if scaling_factor is None:
            scaling_factor = self._calculate_scaling_factor(tensor, num_bits)
            
        
        clamp_val = 2 ** (num_bits - 1) - 1
        quantized = torch.clamp(torch.round(tensor / scaling_factor), -clamp_val, clamp_val)
        
        
        dequantized = quantized * scaling_factor
        return dequantized

    def quantize_whole_model(self, num_bits: int = 8) -> nn.Module:
        for name, param in self.model.named_parameters():
            
            if param.dtype in [torch.float32, torch.float16]:
                quantized_param = self._quantize_tensor(param.data, num_bits)
                param.data = quantized_param
                
                
                self.quantization_stats[name] = {
                    'original_dtype': param.dtype,
                    'num_bits': num_bits,
                    'scaling_factor': self._calculate_scaling_factor(param.data, num_bits)
                }
        
        return self.model

    def quantize_selected_components(self, 
                                   component_names: List[str], 
                                   num_bits: int = 8) -> nn.Module:
        for name, param in self.model.named_parameters():
            if any(comp_name in name for comp_name in component_names):
                
                if param.dtype in [torch.float32, torch.float16]:
                    quantized_param = self._quantize_tensor(param.data, num_bits)
                    param.data = quantized_param
                    
                    
                    self.quantization_stats[name] = {
                        'original_dtype': param.dtype,
                        'num_bits': num_bits,
                        'scaling_factor': self._calculate_scaling_factor(param.data, num_bits)
                    }
        
        return self.model

def calculate_model_size(model: nn.Module, quantization_bits: Optional[Dict[str, int]] = None) -> float:
    total_bits = 0
    
    for name, param in model.named_parameters():
        
        if quantization_bits and name in quantization_bits:
            bits_per_param = quantization_bits[name]
        else:
            bits_per_param = 32  
        
        
        num_params = param.numel()
        total_bits += num_params * bits_per_param
    
    return total_bits


def run_quantization_experiments_mlp():
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    original_size_bits = calculate_model_size(model)
    original_size_mb = original_size_bits / 8 / 1024 / 1024  
    
    
    quantizer = ModelQuantizer(model)
    
    
    ffn_components = ['mlp']  
    selective_quantized_model = quantizer.quantize_selected_components(
        component_names=ffn_components,
        num_bits=8
    )
    
    
    quantization_bits = {name: quantizer.quantization_stats[name]['num_bits'] 
                         for name in quantizer.quantization_stats}
    quantized_size_bits = calculate_model_size(selective_quantized_model, quantization_bits)
    quantized_size_mb = quantized_size_bits / 8 / 1024 / 1024  
    
    
    print(f"Original Model Size: {original_size_mb:.2f} MB")
    print(f"Quantized Model Size: {quantized_size_mb:.2f} MB")
    print(f"Compression Ratio: {original_size_bits / quantized_size_bits:.2f}x")

def run_quantization_experiments_attn():
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    original_size_bits = calculate_model_size(model)
    original_size_mb = original_size_bits / 8 / 1024 / 1024  
    
    
    quantizer = ModelQuantizer(model)
    
    
    ffn_components = ['attn']  
    selective_quantized_model = quantizer.quantize_selected_components(
        component_names=ffn_components,
        num_bits=8
    )
    
    
    quantization_bits = {name: quantizer.quantization_stats[name]['num_bits'] 
                         for name in quantizer.quantization_stats}
    quantized_size_bits = calculate_model_size(selective_quantized_model, quantization_bits)
    quantized_size_mb = quantized_size_bits / 8 / 1024 / 1024  
    
    
    print(f"Original Model Size: {original_size_mb:.2f} MB")
    print(f"Quantized Model Size: {quantized_size_mb:.2f} MB")
    print(f"Compression Ratio: {original_size_bits / quantized_size_bits:.2f}x")


def run_quantization_experiments_fixed():
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    original_size_bits = calculate_model_size(model)
    original_size_mb = original_size_bits / 8 / 1024 / 1024  
    
    
    quantizer = ModelQuantizer(model)
    
    quantized_model = quantizer.quantize_whole_model(
        num_bits=8
    )
    
    
    quantization_bits = {name: quantizer.quantization_stats[name]['num_bits'] 
                         for name in quantizer.quantization_stats}
    quantized_size_bits = calculate_model_size(quantized_model, quantization_bits)
    quantized_size_mb = quantized_size_bits / 8 / 1024 / 1024  
    
    
    print(f"Original Model Size: {original_size_mb:.2f} MB")
    print(f"Quantized Model Size: {quantized_size_mb:.2f} MB")
    print(f"Compression Ratio: {original_size_bits / quantized_size_bits:.2f}x")

if __name__ == '__main__':
    print("whole model")
    run_quantization_experiments_fixed()
    print("Only MLP")
    run_quantization_experiments_mlp()
    print("Attention")
    run_quantization_experiments_attn()
    
    
class ModelEvaluator:
    def __init__(self, model_name: str = 'gpt2-medium'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.original_model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.quantizer = ModelQuantizer(self.original_model)
        
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
        
        
        with torch.no_grad():
            model(**inputs)
        
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                model(**inputs)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_runs
        return avg_latency

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

    def evaluate_model(self, quantize_components: List[str] = None, num_bits: int = 8) -> Tuple[dict, dict]:
        """Evaluate original and quantized models."""
        
        eval_texts = self.load_evaluation_data(num_samples=3000)
        sample_text = "The quick brown fox jumps over the lazy dog."
        
        
        original_metrics = {
            'latency': self.measure_inference_latency(self.original_model, sample_text),
            'perplexity': self.compute_perplexity(self.original_model, eval_texts)
        }
        
        
        if quantize_components:
            quantized_model = self.quantizer.quantize_selected_components(quantize_components, num_bits)
        else:
            quantized_model = self.quantizer.quantize_whole_model(num_bits)
            
        quantized_metrics = {
            'latency': self.measure_inference_latency(quantized_model, sample_text),
            'perplexity': self.compute_perplexity(quantized_model, eval_texts)
        }
        
        return original_metrics, quantized_metrics

def run_evaluation_experiments():
    evaluator = ModelEvaluator('gpt2-medium')
    
    
    print("\nEvaluating full model quantization...")
    orig_metrics, quant_metrics = evaluator.evaluate_model()
    print(f"Original Model - Latency: {orig_metrics['latency']*1000:.2f}ms, Perplexity: {orig_metrics['perplexity']:.2f}")
    print(f"Quantized Model - Latency: {quant_metrics['latency']*1000:.2f}ms, Perplexity: {quant_metrics['perplexity']:.2f}")

    evaluator1 = ModelEvaluator('gpt2-medium')
    
    print("\nEvaluating MLP-only quantization...")
    orig_metrics, quant_metrics = evaluator1.evaluate_model(quantize_components=['mlp'])
    print(f"Original Model - Latency: {orig_metrics['latency']*1000:.2f}ms, Perplexity: {orig_metrics['perplexity']:.2f}")
    print(f"MLP Quantized - Latency: {quant_metrics['latency']*1000:.2f}ms, Perplexity: {quant_metrics['perplexity']:.2f}")

    evaluator2 = ModelEvaluator('gpt2-medium')
    
    print("\nEvaluating attention-only quantization...")
    orig_metrics, quant_metrics = evaluator2.evaluate_model(quantize_components=['attn'])
    print(f"Original Model - Latency: {orig_metrics['latency']*1000:.2f}ms, Perplexity: {orig_metrics['perplexity']:.2f}")
    print(f"Attention Quantized - Latency: {quant_metrics['latency']*1000:.2f}ms, Perplexity: {quant_metrics['perplexity']:.2f}")

if __name__ == '__main__':
    run_evaluation_experiments()