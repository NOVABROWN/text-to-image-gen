# Text-to-Image Generation Platform
### Generative AI · Stable Diffusion + CGAN + LoRA + Attention + Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Gradio](https://img.shields.io/badge/UI-Gradio-purple)

---

## Overview

This is a small personal project for experimenting with text-to-image generation using modern AI tools. It brings together a few different ideas in one place: diffusion models, LoRA fine-tuning, prompt embedding, and a simple GAN-based approach.

The goal is to make it easy to try out image generation from text prompts and see how different pieces of the pipeline work together.

---

## Key Features

- Generate images from text prompts using Stable Diffusion
- Fine-tune models with LoRA on custom image datasets
- Preprocess text and create CLIP-based embeddings
- Explore datasets such as Oxford-102 Flowers and SBU Captions
- Train and evaluate an attention-enhanced GAN
- Run a multi-model image generation workflow from one project
- Save outputs, metrics, and experiment visuals automatically

---

## Core Features

### Stable Diffusion Generation

- Load Stable Diffusion 1.5 from HuggingFace
- Generate 512x512 images from prompts
- Support Euler-A, Euler, DDIM, DPM Solver, and LMS schedulers
- Use classifier-free guidance (CFG) for stronger prompt control
- Work with GPU or CPU fallback support

Key file: app.py

### LoRA Fine-Tuning

- Upload a custom image dataset as a ZIP archive
- Train a LoRA adapter for style or domain-specific generation
- Support presets like artwork, medical, product, portrait, and custom
- Track training loss and save adapter weights

Key file: fine_tuner.py

### Text Preprocessing and Embeddings

- Tokenize prompts using CLIP tokenizers
- Create 768-dimensional text embeddings
- Analyze prompt similarity and token statistics
- Export embeddings to .npy and .json files

Key file: prompt_encoder.py

### Dataset Exploration

- Explore the Oxford-102 Flowers dataset
- Look at SBU Captions stats and diversity
- Generate charts and sample galleries

Key file: explore.py

### Attention-Enhanced GAN

- Add self-attention and cross-attention layers to improve GAN quality
- Improve spatial focus and text-conditioned image generation
- Produce cleaner and more stable images than a basic CGAN

Key file: attention_gan.py

### End-to-End Pipeline

- Combine text preprocessing, embeddings, and image generation
- Support multiple generation approaches in one workflow
- Process single prompts or batches
- Save generated outputs and metadata automatically

Key file: pipeline.py

---

## Architecture Summary

### Stable Diffusion Flow

Text Prompt -> CLIP Text Encoder -> UNet Denoising -> VAE Decoder -> Image

### GAN Flow

Noise + Label -> Generator -> Image
Image + Label -> Discriminator -> Real/Fake Score

### Attention-Enhanced GAN

- Self-attention refines spatial features
- Cross-attention uses text embeddings to guide generation
- Residual connections enable smoother learning and sharper images

---

## Project Structure

```text
text-to-image-gen/
├── app.py                          # Main Gradio UI and Stable Diffusion generator
├── cgan.py                         # Conditional GAN implementation
├── fine_tuner.py                   # LoRA fine-tuning pipeline
├── prompt_encoder.py               # Prompt tokenization and embedding generation
├── explore.py                      # Dataset exploration and visualization
├── attention_gan.py                # Attention-enhanced GAN
├── pipeline.py                     # End-to-end generation pipeline
├── text_to_image_generation_complete.ipynb  # Full self-contained notebook for all tasks
├── README.md                       # Project documentation
├── outputs/                        # Generated images and experiment outputs
└── data/                           # Dataset folders
```

---

## Setup and Usage

### Option 1: Google Colab (Recommended)

1. Open `text_to_image_generation_complete.ipynb` in Colab
2. Run the cells in order
3. Let the notebook complete all tasks and launch the Gradio interface

### Option 2: Local Run

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate peft bitsandbytes transformers scipy gradio matplotlib
python app.py
```

The UI will be available at http://localhost:7860.

### Run Individual Components

```bash
python explore.py
python attention_gan.py
python pipeline.py
```

---

## Example Usage

### Stable Diffusion Generation

```python
from app import StableDiffusionGenerator

gen = StableDiffusionGenerator()
image, metadata = gen.generate_image(
    prompt="a serene mountain landscape",
    width=512,
    height=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    scheduler="euler_a"
)
image.save("output.png")
```

### Attention GAN Training

```python
from attention_gan import AttentionGANModel

model = AttentionGANModel()
history = model.train(num_epochs=50)
image = model.generate(label_idx=0)
image.save("sample.png")
```

### Full Pipeline

```python
from pipeline import ImageGenerationPipeline

pipeline = ImageGenerationPipeline()
result = pipeline.analyze_and_generate(
    prompt="a beautiful sunset",
    model_choice="sd"
)
print(result["image_path"])
```

---

## Outputs

Generated files are stored in the outputs folder, including:

- images created by Stable Diffusion
- CGAN training plots and grids
- attention GAN samples
- pipeline metadata and batch reports

---

## Requirements

Recommended hardware:
- GPU: NVIDIA RTX 3060 or better
- RAM: 16GB or more

Software:
- Python 3.10+
- PyTorch 2.x
- Diffusers
- Transformers
- PEFT
- Gradio
- Matplotlib
- NumPy
- Pillow

---

## Wrap-Up

This project is now set up as a simple, hands-on playground for experimenting with text-to-image generation.

It includes a mix of generation styles and tools, so it can be used for learning, testing ideas, and building on top of the current setup.

The main notebook for the full workflow is complete_internship_tasks.ipynb.
