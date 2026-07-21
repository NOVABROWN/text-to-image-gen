# Internship Report
## Text-to-Image Generation Using Deep Learning

**Name:** Nishita Solanki
**Program:** AI/ML Internship
**Duration:** [Start Date] – [End Date]
**Submission Date:** July 2026

---

## 1. Introduction

This report documents my internship work on building a **Text-to-Image Generation Platform** using deep learning techniques. Over the course of the internship, I designed and implemented six interconnected components — ranging from fine-tuning pre-trained diffusion models with LoRA, to building Conditional GANs, encoding text with CLIP, analyzing public datasets, constructing attention-enhanced neural networks, and integrating everything into a comprehensive end-to-end pipeline.

The goal was to gain hands-on experience with modern generative AI methods and to produce a working software system that demonstrates the full lifecycle of a text-to-image generation project: from raw text input to a generated image output.

---

## 2. Background

### 2.1 Text-to-Image Generation
Text-to-image generation is the task of producing a visual image that matches a natural language description. Recent advances in **diffusion models** (e.g., Stable Diffusion) and **Generative Adversarial Networks (GANs)** have made this feasible at high quality.

### 2.2 Stable Diffusion and LoRA
Stable Diffusion is a latent diffusion model trained on billions of image-text pairs. **Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that injects trainable rank-decomposition matrices into attention layers, enabling domain-specific adaptation with minimal compute and memory.

### 2.3 Conditional GANs (CGANs)
GANs consist of a Generator that creates images and a Discriminator that distinguishes real from fake. **Conditional GANs** extend this by conditioning both networks on additional information (e.g., class labels), enabling targeted generation of specific categories.

### 2.4 CLIP Text Encoder
**CLIP (Contrastive Language–Image Pretraining)** from OpenAI learns joint embeddings of images and text. Its text encoder produces rich 768-dimensional vectors that capture semantic meaning — making it the backbone of modern text-conditioned image generation.

### 2.5 Attention Mechanisms in GANs
Self-attention and cross-attention allow the network to model long-range dependencies in images. Incorporating attention into a GAN generator improves global coherence and allows conditioning on text features.

---

## 3. Learning Objectives

By completing this internship, I aimed to:
1. Understand and apply **parameter-efficient fine-tuning** (LoRA/PEFT) to large pre-trained models.
2. Implement a **Conditional GAN** architecture from scratch and train it on a custom task.
3. Use **HuggingFace Transformers** to preprocess, tokenize, and encode natural language.
4. Perform **dataset exploration and statistical analysis** on a real-world public dataset.
5. Incorporate **self-attention and cross-attention** mechanisms into a deep learning model.
6. Integrate multiple components into a **unified end-to-end pipeline** with a user interface.

---

## 4. Activities and Tasks

### Task 1 — LoRA Fine-Tuning of Stable Diffusion (`fine_tuner.py`)

**Objective:** Fine-tune a pre-trained text-to-image model on a custom dataset to produce domain-specific images (e.g., artwork style).

**Approach:**
- Loaded `runwayml/stable-diffusion-v1-5` with the `diffusers` library.
- Applied LoRA adapters via the `peft` library to the UNet's attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`), reducing the number of trainable parameters from ~860M to ~2.4M.
- Built a `CustomImageDataset` that reads images and their paired `.txt` caption files.
- Trained the adapted model for up to 100 steps with configurable rank, alpha, and learning rate.
- Saved loss history as a PNG chart and training metrics as JSON.

**Outcome:** A `LoRAFineTuner` class capable of adapting Stable Diffusion to any custom image domain given a folder of captioned images.

**Key Learning:** LoRA enables efficient fine-tuning of billion-parameter models on consumer hardware by only updating low-rank weight deltas in the attention layers.

---

### Task 2 — Conditional GAN for Shape Generation (`cgan.py`)

**Objective:** Build a CGAN that takes a text label (e.g., "circle", "square") and generates a corresponding image.

**Approach:**
- Defined 5 shape classes: Square, Circle, Triangle, Rectangle, Ellipse.
- Built a `ShapeDataset` that synthetically generates 64×64 colored shape images on the fly.
- Implemented a `Generator` (label embedding → FC → upsampling convolutions) and `Discriminator` (convolutional + label embedding → binary classification).
- Trained using the standard GAN loss (Binary Cross-Entropy) with the Adam optimizer.
- Saved `cgan_generator.pth` weights after training; saved loss curves and sample grids to `outputs/experiments/`.

**Outcome:** A trained CGAN model capable of generating accurate shape images conditioned on a text category label. Generator weights are persisted for immediate inference without re-training.

**Key Learning:** Conditional inputs (label embeddings) allow GANs to target specific output categories — a fundamental building block of text-conditioned image generation.

---

### Task 3 — Text Preprocessing and Encoding (`prompt_encoder.py`)

**Objective:** Build a software module that tokenizes and encodes text descriptions using HuggingFace Transformers.

**Approach:**
- Wrapped `openai/clip-vit-large-patch14` using `CLIPTokenizer` and `CLIPTextModel` from the `transformers` library.
- Implemented `tokenize()` — returns token IDs, decoded token list, attention masks, and a truncation flag.
- Implemented `encode()` — produces per-token 768-dimensional embeddings (`last_hidden_state`) and a pooled sentence embedding.
- Implemented `compare()` — computes cosine similarity between two prompts to measure semantic distance.
- Implemented `export()` — saves embeddings as `.npy` arrays and metadata as `.json`.
- Integrated a lightweight fallback mode for offline/no-GPU environments.

**Outcome:** A `PromptEncoder` class fully integrated into the Gradio UI with real-time tokenization, embedding visualization, and prompt comparison.

**Key Learning:** CLIP's text encoder maps natural language into a high-dimensional embedding space aligned with visual semantics — this is what makes text-conditioned image generation possible.

---

### Task 4 — Dataset Exploration and Analysis (`explore.py`)

**Objective:** Load a public image-text dataset, analyze its statistics, and visualize text + image combinations.

**Approach:**
- Loaded the **Oxford-102 Flowers** dataset via `torchvision.datasets.Flowers102` (102 flower species, 8,189 images, with automatic download).
- Streamed the large-scale **`jackyhate/text-to-image-2M`** dataset from Hugging Face using streaming mode to analyze real-world prompt structures without high disk overhead.
- Generated statistical analysis: class distribution, images per class, resolution uniformity (128×128), and text caption metrics (average length: 49.7 words; std: 21.0; range: 10–111 words).
- Built robust multi-key extraction to handle varied WebDataset JSON schema layouts (`caption`, `prompt`, `txt`).
- Produced 4 visualizations saved to `outputs/task4/`:
  - `task4_flowers_class_dist.png` — class distribution bar chart
  - `task4_flowers_gallery.png` — image gallery with label descriptions
  - `task4_flickr_stats.png` — prompt length distribution histogram + boxplot
  - `task4_dashboard.png` — combined statistics summary dashboard

**Outcome:** Four high-quality visualizations demonstrating proficiency in dataset loading, streaming large-scale WebDatasets, and statistical profiling with PyTorch and Matplotlib.

**Key Learning:** Understanding dataset characteristics (class balance, prompt length, resolution uniformity) and handling modern webdataset streaming are essential prerequisites for training and fine-tuning text-to-image models.

---

### Task 5 — Attention-Enhanced GAN (`attention_gan.py`)

**Objective:** Use self-attention and cross-attention mechanisms to improve a GAN and produce higher-quality images.

**Approach:**
- Built a `SelfAttentionLayer` using Q, K, V projections over 2D feature maps with softmax attention and a residual connection. This captures long-range spatial dependencies within the feature map.
- Built a `CrossAttentionLayer` where the image features act as Query and the text/label embedding acts as Key and Value — enabling explicit text conditioning of the feature map.
- Assembled an `AttentionGenerator` that incorporates both attention types at mid-resolution upsampling stages.
- Built an `AttentionDiscriminator` with self-attention at the bottleneck for global context evaluation.
- Trained with spectral normalization, hinge loss, and the Adam optimizer with learning rate 0.0002.
- Saved trained weights to `task5_attention_gan_weights.pth` and generated a sample grid to `outputs/task5/`.

**Outcome:** An attention-enhanced GAN that demonstrates improved global coherence over the basic CGAN baseline, with persistent weights for immediate inference.

**Key Learning:** Attention mechanisms allow generative models to relate distant parts of an image to each other and to the conditioning signal — directly improving image realism and text alignment.

---

### Task 6 — End-to-End Generation Pipeline (`pipeline.py`)

**Objective:** Build a comprehensive pipeline that combines text preprocessing, embedding, and multi-model generation into a single unified system.

**Approach:**
- Built `TextPreprocessor`: cleans raw text, tokenizes with CLIP, produces 768-dim embeddings, and computes descriptive statistics (word count, special character ratio, vocabulary diversity).
- Built `ImageGenerationPipeline`: a unified interface supporting Stable Diffusion (`sd`), CGAN (`cgan`), and Attention-GAN (`attention_gan`) as generation backends.
- Implemented `analyze_and_generate()`: full flow — text analysis → model selection → image generation → output saved with metadata JSON.
- Implemented `BatchPipelineProcessor`: processes a list of prompts, generates all images, and writes a batch summary JSON report.
- All outputs are saved to `outputs/` with timestamped filenames and rich metadata.

**Outcome:** A production-ready pipeline that serves as the integration layer for the entire project, demonstrable through the Gradio interface.

**Key Learning:** Building end-to-end systems requires careful interface design so that independent modules (encoder, GAN, diffusion model) can be orchestrated cleanly without tight coupling.

---

## 5. Skills Acquired

| Domain | Skills |
|--------|--------|
| Deep Learning | LoRA fine-tuning, GAN training, attention mechanisms |
| NLP | Tokenization, CLIP embeddings, cosine similarity |
| Computer Vision | Image generation, dataset loading, resolution analysis |
| Software Engineering | Modular Python architecture, Gradio UI development |
| Data Analysis | Statistical visualization with Matplotlib, dataset profiling |
| Tools & Frameworks | PyTorch, HuggingFace `transformers`/`diffusers`/`peft`, Gradio |

---

## 6. Evidence of Work

| Task | Evidence |
|------|----------|
| Task 1 | `fine_tuner.py`, LoRA adapter integration, training loop |
| Task 2 | `cgan.py`, `cgan_generator.pth` (saved weights), `outputs/experiments/cgan_loss_curves.png`, `cgan_shape_grid.png` |
| Task 3 | `prompt_encoder.py`, CLIP tokenization, embedding export |
| Task 4 | `explore.py`, `jackyhate/text-to-image-2M` streaming, `outputs/task4/task4_flowers_class_dist.png`, `task4_flowers_gallery.png`, `task4_flickr_stats.png`, `task4_dashboard.png` |
| Task 5 | `attention_gan.py`, `task5_attention_gan_weights.pth`, `outputs/task5/task5_attention_gan_grid.png` |
| Task 6 | `pipeline.py`, batch processing, metadata JSON outputs |
| Integration | `app.py` — Gradio UI with 6 task tabs |
| Notebook | `text_to_image_generation_complete.ipynb` — self-contained walkthrough |

---

## 7. Challenges Faced

1. **VRAM constraints:** Stable Diffusion requires ~6GB VRAM for inference and ~10GB for fine-tuning. Mitigated by using LoRA (reduces trainable parameters by ~99%) and setting `torch_dtype=float16`.

2. **Windows encoding issues:** Several Unicode characters in print statements caused `UnicodeEncodeError` on Windows due to the `cp1252` codepage. Resolved by replacing special characters with ASCII equivalents.

3. **WebDataset schema heterogeneity:** Large-scale datasets like `jackyhate/text-to-image-2M` contain heterogeneous metadata across shards (some shards store prompts in `json['caption']`, others in `json['prompt']` or raw `txt`). Resolved by building a resilient multi-key extraction function with fallback handling.

4. **Model download time:** First-run downloads of CLIP and Stable Diffusion models total ~4GB. Mitigated by implementing offline fallback modes and caching.

5. **GAN training instability:** GANs can suffer from mode collapse and non-convergence. Used spectral normalization, gradient clipping, and a learning rate of 0.0002 to stabilize training.

---

## 8. Outcomes and Results

- **Functional Gradio Application:** `app.py` launches a multi-tab interface covering all 6 tasks.
- **Trained Models:** CGAN (`cgan_generator.pth`) and Attention GAN (`task5_attention_gan_weights.pth`) are trained and saved, enabling inference without re-training.
- **Generated Outputs:** Sample images in `outputs/`, visualizations in `outputs/task4/` and `outputs/experiments/`.
- **Complete Notebook:** `text_to_image_generation_complete.ipynb` provides a self-contained Colab-ready walkthrough of all tasks.
- **Documentation:** `README.md` documents architecture, setup, and usage.

---

## 9. Conclusion

This internship provided me with deep, practical experience across the full stack of modern text-to-image AI: from the mathematics of attention mechanisms and LoRA adaptation, to the practical engineering challenges of building a modular, deployable system. I developed proficiency with the HuggingFace ecosystem (`transformers`, `diffusers`, `peft`), PyTorch, and Gradio — tools used by the leading AI research and product teams worldwide.

The most significant insight gained was how tightly coupled text understanding and image generation must be: producing high-quality, semantically aligned images requires every component — tokenization, embedding, attention, and generation — to work in concert. This end-to-end perspective is a skill I will carry forward in my career in AI/ML.

---

*Report submitted as part of the AI/ML Internship Program.*
*All work is original and independently completed.*
