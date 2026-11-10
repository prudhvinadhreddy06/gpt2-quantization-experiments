# Model Quantization Experiments on GPT-2

Empirical implementations and benchmarks of neural network quantization techniques for Transformer language models (**GPT-2 Medium**) comparing **custom fixed-point quantization** algorithms against **BitsAndBytes (BnB)** 8-bit, 4-bit, and NormalFloat4 (NF4) quantization.

## Overview

Quantizing large language models is essential for deploying state-of-the-art architectures onto resource-constrained edge accelerators. This project investigates the trade-offs between **model memory footprint**, **inference latency**, and **perplexity preservation** across multiple quantization paradigms:

1. **Custom Fixed-Point Quantization** (`quant.py`):
   - Dynamic scaling factor calculation per tensor: $S = \frac{2^{b-1} - 1}{\max(|X|)}$.
   - Symmetric and asymmetric affine mapping to fixed integer representations.
   - Selective sub-module quantization targeting MLP feed-forward blocks vs. Multi-Head Attention projections independently.
2. **BitsAndBytes (BnB) Quantization** (`bnb.py`):
   - LLM.int8() vector-wise outlier decomposition.
   - 4-bit Linear quantization and NormalFloat4 (NF4) theoretical information-optimal distribution.
   - Perplexity benchmarking evaluated on WikiText-2 and Penn Treebank (PTB).

---

## Empirical Benchmark Results

Evaluated on **GPT-2 Medium** (WikiText-2 test set):

| Quantization Method | Model Size (MB) | Size Reduction (%) | Inference Latency (ms) | Perplexity | Perplexity Degradation (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Original)** | 1,353.54 MB | — | **16.14 ms** | 1,427.59 | Baseline |
| **8-Bit (BnB)** | 388.77 MB | **71.3%** | 98.70 ms | 1,440.05 | +0.9% (Near-lossless) |
| **4-Bit FP4 (BnB)** | 244.77 MB | **81.9%** | 38.53 ms | 1,531.03 | +7.2% |
| **NF4 (NormalFloat4)** | 244.77 MB | **81.9%** | 38.05 ms | **1,358.93** | **-4.8% (Optimal fit)** |

### Key Findings
- **81.9% Memory Reduction**: 4-bit and NF4 quantization compress the model from 1.35 GB down to ~245 MB, enabling execution on edge hardware with limited on-chip SRAM/DRAM.
- **NF4 Superiority**: NormalFloat4 yields better perplexity than standard 4-bit quantization because transformer weights naturally follow a zero-centered normal distribution.
- **Component Selectivity**: Quantizing MLP layers achieves ~65% of the total size reduction while having minimal impact on attention map integrity.

---

## Repository Structure

```
├── bnb.py                      # BitsAndBytes evaluation pipeline (8-bit, 4-bit, NF4)
├── quant.py                    # Custom fixed-point quantization & selective layer quantizer
├── bits.ipynb                  # Interactive notebook with BnB quantization benchmarks & plots
├── qunat.ipynb                 # Interactive notebook with custom tensor quantization experiments
├── ANLP_Assignment4_report.pdf # Comprehensive technical analysis & report
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run BitsAndBytes Experiments
```bash
python bnb.py
```

### 3. Run Custom Fixed-Point & Selective Quantization
```bash
python quant.py
```
Evaluates full-model, MLP-only, and Attention-only quantization schemes, reporting compression ratio, memory delta, latency, and perplexity.
