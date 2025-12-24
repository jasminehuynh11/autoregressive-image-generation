# PixelCNN: Autoregressive Image Generation on CIFAR-10

A PyTorch implementation of PixelCNN for autoregressive image generation, featuring class-conditional generation, masked convolutions, and comprehensive evaluation on CIFAR-10.

## Overview

This project implements a class-conditional PixelCNN model for generating images from the CIFAR-10 dataset. The implementation includes masked convolutions to enforce autoregressive dependencies, gated residual blocks for improved expressivity, and comprehensive training and evaluation pipelines.

## Features

- **Class-Conditional Generation**: Generate images conditioned on class labels (10 CIFAR-10 classes)
- **Masked Convolutions**: Type-A and Type-B masked convolutions enforcing proper autoregressive ordering
- **Gated Residual Blocks**: Enhanced expressivity through gated activation functions
- **Comprehensive Evaluation**: Metrics including NLL, BPD (bits-per-dimension), and perplexity
- **Reproducible Training**: Deterministic splits, fixed seeds, and checkpoint management
- **Visual Analysis**: Sample generation, temperature sweeps, and feature visualizations

## Architecture

The model architecture consists of:

- **Type-A Masked Convolution** (7×7): First layer that blocks the current pixel
- **15 Gated Residual Blocks**: Type-B masked convolutions with gated activations
- **Class Conditioning**: Additive label embeddings injected after the first layer
- **Output Head**: Two 1×1 masked convolutions producing 256-way categorical distributions per channel/pixel

**Model Size**: ~7.0M trainable parameters

## Results

### Performance Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **BPD** | 4.1832 | 4.7517 | **4.7507** |
| **NLL (nats/dim)** | 2.8996 | 3.2936 | **3.2929** |
| **Perplexity** | 18.2 | 27.0 | **26.94** |

**Best Validation BPD**: 4.7505 (epoch 49)  
**Generalization Gap**: 0.5685 bits/dim

The model achieves performance on-benchmark for compact PixelCNN architectures on CIFAR-10 (typical range: 4.7–5.0 BPD).

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+ (with CUDA support recommended)
- torchvision
- numpy
- matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/autoregressive-image-generation.git
cd autoregressive-image-generation

# Install dependencies
pip install torch torchvision numpy matplotlib
```

## Usage

### Training

The notebook `pixelcnn_cifar10.ipynb` contains the complete implementation. To train the model:

1. Open the Jupyter notebook
2. Run all cells sequentially
3. Training will automatically:
   - Download and preprocess CIFAR-10
   - Initialize the PixelCNN model
   - Train for 50 epochs with EMA and mixed precision
   - Save checkpoints to `./checkpoints/`

### Generating Samples

After training, generate samples using the model:

```python
# Unconditional generation
samples = model.sample(batch_size=16, temperature=1.0)

# Class-conditional generation
labels = torch.tensor([0, 1, 2, ...])  # Class indices
samples = model.sample(batch_size=16, labels=labels, temperature=0.9)
```

### Evaluation

The training pipeline automatically evaluates on validation and test sets. Key metrics are logged per epoch and visualized in training curves.

## Project Structure

```
autoregressive-image-generation/
├── pixelcnn_cifar10.ipynb    # Main implementation notebook
├── README.md                  # This file
├── checkpoints/               # Saved model checkpoints (created during training)
└── data/                      # CIFAR-10 dataset (auto-downloaded)
```

## Technical Details

### Training Configuration

- **Optimizer**: Adam (lr=5×10⁻⁴, β₁=0.9, β₂=0.999, weight_decay=10⁻⁵)
- **Batch Size**: 128
- **Learning Rate Schedule**: 5-epoch warm-up → cosine annealing
- **Gradient Clipping**: L₂ norm ≤ 1.0
- **Mixed Precision**: Enabled (FP32 loss computation for stability)
- **EMA**: Decay rate 0.997 (used for evaluation)

### Data Preprocessing

- **Normalization**: Inputs normalized to [-1, 1] using per-channel mean=0.5, std=0.5
- **Augmentations**: RandomCrop(32, padding=4), RandomHorizontalFlip(0.5), ColorJitter
- **Split**: Deterministic 45k/5k/10k train/validation/test split
- **Reproducibility**: Fixed seeds for all random operations

### Model Components

1. **MaskedConv2d**: Implements Type-A and Type-B spatial masking
2. **ResidualBlock**: Gated residual blocks with optional gated activation
3. **CausalSelfAttention**: Optional masked self-attention layers
4. **ConditionalPixelCNN**: Main model class with class conditioning
5. **PixelCNNLoss**: Computes NLL, BPD, and perplexity metrics

## Key Implementation Highlights

- **Causality Validation**: Automated tests ensure no future-pixel leakage
- **Stable Training**: FP32 loss computation, gradient clipping, and BatchNorm
- **Efficient Training**: Mixed precision (AMP) and EMA for faster convergence
- **Comprehensive Analysis**: Feature visualizations, activation statistics, and sample quality metrics

## Limitations & Future Work

### Current Limitations

- Texture-heavy, object-light samples due to limited non-local context
- 256-way softmax output less expressive than mixture-based alternatives
- Sequential sampling is computationally expensive (O(HW) complexity)

### Potential Improvements

1. **Causal Self-Attention**: Add attention layers at mid-depth for non-local context
2. **DMoL Output Head**: Replace softmax with Discretized Mixture of Logistics
3. **Enhanced Regularization**: Dropout, increased weight decay, Cutout augmentation
4. **Sampling Optimizations**: Row-caching, temperature tuning, noise seeding

## Reproducibility

The implementation ensures full reproducibility:

- Fixed random seeds for Python, NumPy, and PyTorch
- Deterministic data splits and DataLoader workers
- Training configurations saved with checkpoints
- Per-epoch metrics and visualizations logged

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{pixelcnn_cifar10,
  author = {Huynh, Phuong Thao (Jasmine)},
  title = {PixelCNN: Autoregressive Image Generation on CIFAR-10},
  year = {2025},
  url = {https://github.com/yourusername/autoregressive-image-generation}
}
```

## References

- [PixelCNN](https://arxiv.org/abs/1601.06759) - Original PixelCNN paper
- [Conditional PixelCNN](https://arxiv.org/abs/1606.05328) - Class-conditional extension
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Dataset reference

## Author

**Phuong Thao (Jasmine) Huynh**

## License

This project is open source and available under the MIT License.
