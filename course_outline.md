.# ML 101 — PyTorch Course Outline

## Module 1: Foundations

### 1.1 — The Perceptron & Linear Models
- What a neuron is, dot products, weights & biases
- Linear regression from scratch (no PyTorch yet, just NumPy)
- Loss functions (MSE, cross-entropy), gradient descent by hand

### 1.2 — PyTorch Basics
- Tensors, autograd, `backward()`
- Writing a training loop: forward pass → loss → backward → optimizer step
- `Dataset` / `DataLoader`, transforms

### 1.3 — MNIST with an MLP
- Fully-connected network on MNIST
- Activation functions (ReLU, sigmoid, softmax)
- Overfitting, train/val/test split, `nn.Module`

---

## Module 2: Computer Vision

### 2.1 — Convolutional Neural Networks
- Convolutions, pooling, receptive field
- LeNet → AlexNet architecture walk-through
- MNIST / CIFAR-10 with a CNN

### 2.2 — Modern CNN Tricks
- Batch norm, dropout, data augmentation
- Residual connections (ResNet intuition)
- Transfer learning: fine-tune a pretrained ResNet on a small dataset

---

## Module 3: Optimization & Training Craft

### 3.1 — Optimizers
- SGD, momentum, Adam — what the math actually does
- Learning rate schedules, warmup
- Gradient clipping, weight decay

### 3.2 — Regularization & Debugging
- Bias-variance tradeoff
- Reading loss curves, common failure modes
- `torch.utils.tensorboard` for experiment tracking

---

## Module 4: Sequence Models

### 4.1 — Recurrent Neural Networks
- Vanilla RNN, backprop through time, vanishing gradients
- LSTM & GRU — the gating intuition
- Character-level language model

### 4.2 — Embeddings & Word Vectors
- One-hot → dense embeddings, `nn.Embedding`
- Word2Vec intuition, cosine similarity
- Sentiment classification with an LSTM

---

## Module 5: Attention & Transformers

### 5.1 — Attention Mechanism
- The problem RNNs have at long range
- Scaled dot-product attention from scratch
- Multi-head attention

### 5.2 — The Transformer
- Encoder/decoder architecture (Vaswani et al.)
- Positional encodings, layer norm, feed-forward blocks
- Build a small transformer for sequence-to-sequence translation

### 5.3 — Modern LLM Architecture
- GPT-style decoder-only transformer
- Tokenization (BPE), causal masking
- Train a tiny GPT on a text corpus

---

## Module 6: Generative Models

### 6.1 — Variational Autoencoders
- Autoencoders → latent space → VAE
- The reparameterization trick, ELBO loss
- Generate MNIST digits from a VAE

### 6.2 — GANs (brief)
- Generator vs. discriminator, minimax game
- Mode collapse, training instability — why diffusion won

### 6.3 — Diffusion Models
- Forward noising process, reverse denoising, score matching
- DDPM: the noise prediction U-Net
- DDIM sampling, classifier-free guidance
- Train a small diffusion model on MNIST / CIFAR-10

---

## Capstone Options
- Fine-tune a small LLM (GPT-2) on a custom dataset
- Build an image captioning model (CNN encoder + transformer decoder)
- Implement a LoRA fine-tune for a diffusion model