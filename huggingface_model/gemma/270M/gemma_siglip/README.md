# Gemma-3-Nano + SigLIP-2 Vision-Language Pipeline

This repository contains a complete, end-to-end pipeline for training and interacting with a custom Vision-Language Model (VLM). It fuses Google's **SigLIP-2** (vision encoder) with **Gemma-3-Nano** (text decoder) using a "sandwich" embedding architecture.



The pipeline includes a training script with strict data-leakage prevention, a single-turn CLI inference tool, and a multi-turn conversational chat interface.

**Author / Lead Contributor:** Phanthavong Douangphachanh (@phandouang)

---

## ⚙️ Installation & Setup

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install the required packages. The setup relies on `accelerate` for mixed-precision hardware optimization and `climage` for rendering validation images in the terminal.

```bash
pip install -r requirements.txt

```

### 2. Hugging Face Authentication

Gemma 3 is a gated model. To download the weights, you must accept the license agreement on the Hugging Face model page.

You do **not** need to manually set environment variables. The scripts utilize an interactive, secure login block. On your first run, the CLI will prompt you to securely enter your HF token, which will be cached locally for all future runs.

---

## 🚀 Usage Guide

### Phase 1: Training and Validation (`gemma_siglip2_finetune.py`)

This script handles the alignment of the SigLIP vision tower with the Gemma 3 LLM. It downloads a 30k COCO subset, splits it securely (to prevent validation leakage), trains the projector, and outputs a visual validation report directly in your terminal.

**Run both training and validation (Default):**

```bash
python gemma_siglip2_finetune.py

```

**Run only the training loop:**

```bash
python gemma_siglip2_finetune.py --mode train

```

**Run only the validation loop (Requires pre-trained weights):**

```bash
python gemma_siglip2_finetune.py --mode validate

```

### Phase 2: Single-Turn Inference (`gemma_siglip2_inference.py`)

Once your model is trained and weights are saved to `./gemma-nano-siglip2/best_pytorch_model.bin`, you can test it on any local image. The model will ingest the image and generate a single, descriptive caption.

**Command:**

```bash
python gemma_siglip2_inference.py path/to/your/image.jpg

```

**Example Output:**

```text
🚀 Using device: CUDA
📦 Loading tokenizer and image processor...
🖼️  Processing image: sample_dog.jpg...

✨ Generating description...
--------------------------------------------------
🤖 Prediction: A golden retriever sitting in a green grassy field holding a yellow tennis ball in its mouth.

```

### Phase 3: Multi-Turn Conversational Chat (`gemma_siglip2_chat.py`)

This script loads the model into an interactive chat loop. It processes the heavy image embeddings only once at the start, allowing for rapid, memory-efficient follow-up questions about the image.

**Command:**

```bash
python gemma_siglip2_chat.py path/to/your/image.jpg

```

**Example Output:**

```text
🚀 Initializing on CUDA...
🖼️  Analyzing image: sample_dog.jpg...

💬 Chat session started! (Type 'quit' or 'exit' to end)
------------------------------------------------------------
You: What kind of dog is this?
🤖 Model: This appears to be a Golden Retriever.

You: What does it have in its mouth?
🤖 Model: The dog is holding a yellow tennis ball.

```

---

## 📁 Repository Structure

* `gemma_siglip2_finetune.py`: Core architecture definition and `Trainer` setup.
* `gemma_siglip2_inference.py`: Lightweight script for one-shot image captioning.
* `gemma_siglip2_chat.py`: Interactive loop maintaining conversation history tensors.
* `requirements.txt`: Project dependencies.
