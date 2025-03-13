# Fine-tuned Phi-2 Assistant

This repository contains code and instructions for fine-tuning Microsoft's Phi-2 model using the OpenAssistant dataset (OASST1) with advanced training techniques.

# Huggingface
- The fine tuned model is deployed to Huggingface Spaces. Below is the link
- https://huggingface.co/spaces/satyanayak/PHI2-SFT-OASST1

## Training Details

### Training Environment
- Fine tuning carried in Kaggle envirnment
- Used GPU T4 x2
- Trained one 1 epoch in multiple sessions

### [Training Logs here](./training_logs_final.jsonl)
- Initial few logs missed
- Training step 300 to 2455 captured

### Model Configuration
- Base Model: microsoft/phi-2
- Dataset: OpenAssistant/oasst1 (filtered for English conversations)
- Training Type: Supervised Fine-tuning (SFT)
- Hardware Requirements: GPU with at least 16GB VRAM

### Training Techniques Used
1. **Quantization**:
   - 4-bit quantization using BitsAndBytes
   - NF4 quantization type
   - Double quantization enabled
   - Float16 compute dtype

2. **LoRA Configuration**:
   - Rank (r): 16
   - Alpha: 32
   - Target Modules: q_proj, k_proj, v_proj, dense
   - Dropout: 0.05
   - Task Type: Causal Language Modeling

3. **Training Parameters**:
   - Epochs: 1
   - Batch Size: 4
   - Gradient Accumulation Steps: 4
   - Learning Rate: 2e-4
   - Optimizer: paged_adamw_32bit
   - LR Scheduler: Cosine
   - Warmup Ratio: 0.03
   - Max Sequence Length: 2048

### Checkpointing and Logging
- Checkpoints saved every 100 steps
- Keeps last 2 checkpoints
- Training logs saved in JSONL format
- Automatic checkpoint archiving to ZIP format
- Resumable training support

## Dataset Processing
The OpenAssistant dataset is processed with the following steps:
1. Filtered for English conversations
2. Formatted as Human/Assistant conversations
3. Tokenized with truncation and padding
4. Maximum sequence length of 2048 tokens
