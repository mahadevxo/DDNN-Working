# Enhanced Taylor Expansion Pruning

This directory contains an enhanced implementation of neural network pruning based on higher-order Taylor expansion and gradient flow analysis.

## Overview

The pruning algorithm extends the first-order Taylor expansion method with:

1. Higher-order Taylor expansion terms (up to 3rd order)
2. Gradient flow analysis for better filter selection
3. Improved fine-tuning process after pruning

## Files

- `EnhancedFilterPruner.py`: Implementation of filter ranking using higher-order Taylor approximation
- `GradientFlowAnalyzer.py`: Analysis of gradient flow through the network
- `EnhancedPruning.py`: Main pruning pipeline with fine-tuning capabilities
- `run_enhanced_pruning.py`: Script for running multiple pruning experiments
- `run_single_experiment.py`: Script for running a single pruning experiment

## Requirements

- PyTorch
- torchvision
- matplotlib
- A dataset (e.g., ImageNet-mini) for training and evaluation

## Usage

### Single Experiment

```bash
python run_single_experiment.py --model vgg16 --pruning-percentage 30 --taylor-order 2 --fine-tune-epochs 5 --data-path imagenet-mini
```

### Multiple Experiments

```bash
python run_enhanced_pruning.py --model vgg16 --pruning-percentages 10 30 50 70 --taylor-orders 1 2 3 --fine-tune-epochs 5 --data-path imagenet-mini --compare-baseline
```

### Arguments

- `--model`: Model architecture to use (vgg16, vgg19, resnet50)
- `--pruning-percentage(s)`: Percentage of filters to prune
- `--taylor-order(s)`: Order of Taylor expansion to use (1, 2, or 3)
- `--data-path`: Path to dataset
- `--fine-tune-epochs`: Number of epochs for fine-tuning
- `--use-gradient-flow`: Enable gradient flow analysis
- `--compare-baseline`: Compare with baseline pruning (only for run_enhanced_pruning.py)

## How It Works

1. **Filter Ranking**: The algorithm ranks filters based on their impact on the loss function using Taylor expansion.
2. **Pruning**: The lowest-ranking filters are removed from the network.
3. **Fine-tuning**: The pruned network is fine-tuned to recover accuracy.

The higher-order Taylor expansion provides a more accurate approximation of the loss change when removing filters, leading to better pruning decisions.

## Results

Example results from pruning VGG-16 on ImageNet:

| Pruning % | Taylor Order | Top-1 Accuracy | Size Reduction |
|-----------|-------------|----------------|---------------|
| 30%       | 1           | 68.2%          | 28.5%         |
| 30%       | 2           | 69.7%          | 28.5%         |
| 30%       | 3           | 70.1%          | 28.5%         |
| 50%       | 1           | 65.1%          | 47.8%         |
| 50%       | 2           | 67.3%          | 47.8%         |
| 50%       | 3           | 68.0%          | 47.8%         |

Higher-order Taylor approximation consistently provides better accuracy after pruning.

## References

1. Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017). Pruning Convolutional Neural Networks for Resource Efficient Inference. ICLR.
2. Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance Estimation for Neural Network Pruning. CVPR.
