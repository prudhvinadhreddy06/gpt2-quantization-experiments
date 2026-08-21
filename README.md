# Model Quantization Experiments

This repository contains implementations of various model quantization techniques for GPT-2 using both custom quantization methods and the Bits and Bytes (BnB) library. The project includes tools for evaluating model performance, size reduction, and inference latency across different quantization approaches.

## Features

- Custom fixed-point quantization implementation
- Selective component quantization (MLP, Attention layers)
- Integration with Bits and Bytes (BnB) library for 8-bit, 4-bit, and NF4 quantization
- Comprehensive evaluation metrics (perplexity, latency, model size)
- Support for different evaluation datasets (WikiText, Penn Treebank)

## Requirements

Create a virtual environment and install the requirements:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Project Structure

- `bnb.py`: Implementation of quantization using the Bits and Bytes library
- `quant.py`: Custom quantization implementation with selective component quantization

## Usage

### Bits and Bytes Quantization

To run the BnB quantization experiments:

```bash
python bnb.py
```

This will evaluate:
- 8-bit quantization
- 4-bit quantization
- NF4 quantization

The script will output:
- Model size reduction
- Inference latency changes
- Perplexity impact
- Comparative metrics against the original model

### Custom Quantization

To run the custom quantization experiments:

```bash
python quant.py
```

This will evaluate:
- Full model quantization
- MLP-only quantization
- Attention-only quantization

Each experiment will output:
- Model size comparison
- Compression ratios
- Inference latency
- Perplexity metrics

## Evaluation Metrics

The code measures several key metrics:

1. Model Size
   - Calculated in MB
   - Compression ratio compared to original model

2. Inference Latency
   - Measured in milliseconds
   - Averaged over multiple runs
   - Includes warm-up runs to ensure accurate measurement

3. Perplexity
   - Computed on WikiText-2 test set
   - Used as a proxy for model quality preservation
   - Lower values indicate better performance

## Hardware Requirements

- CUDA-capable GPU recommended for optimal performance
- Minimum 8GB GPU memory for GPT-2 Medium
- CPU execution supported but significantly slower

## Notes and Limitations

1. Memory Usage
   - Keep in mind that loading multiple quantized versions simultaneously may require significant GPU memory
   - The code includes memory management through explicit garbage collection

2. Dataset Considerations
   - Default evaluation uses WikiText-2
   - Can be modified to use Penn Treebank or custom datasets
   - Sample size can be adjusted through the `num_samples` parameter

3. Model Support
   - Currently tested with GPT-2 Medium
   - Can be adapted for other transformer models
   - Some quantization methods may be model-specific


## Models link
`https://iiitaphyd-my.sharepoint.com/:f:/g/personal/prudhvinadh_k_students_iiit_ac_in/Eu8p9F3DTshErCB38uOs438BPzxjN54kELwj6R4GobJIeg?e=QN6qgL`
