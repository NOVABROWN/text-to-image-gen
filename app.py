# ==============================================================================
# Text-to-Image Generation Platform - Standalone Unified Entry Point
# ==============================================================================
# This script integrates:
#   1. Stable Diffusion Image Generation (with switchable schedulers)
#   2. LoRA Fine-Tuning on custom datasets (with caption & loss tracking)
#   3. Conditional GAN Shape Generation (from scratch)
#   4. HuggingFace CLIP Prompt Encoder & Inspector
#   5. Reproducible experiments (CGAN, Scheduler, CFG Ablation)
#
# Run this file with: python app.py
# ==============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add current dir to path to ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the main modules
try:
    from cgan import CGANModel, SHAPES, NUM_CLASSES
    from fine_tuner import LoRAFineTuner
    from prompt_encoder import PromptEncoder
    print("OK: Imported cgan, fine_tuner, and prompt_encoder successfully.")
except ImportError as e:
    print(f"Import Warning: {e}. Falling back to internal classes or dependency warnings.")

# --- Stable Diffusion & Gradio UI Backend -------------------------------------
# Main Application and Gradio UI

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import autocast
import numpy as np
from PIL import Image
import os
import time
import gc
import zipfile
import shutil
from typing import Optional, Tuple, List
from datetime import datetime

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler
)
import gradio as gr

# CELL 3: Core Generator Class - Part 1 (Initialization)
class StableDiffusionGenerator:    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "auto"):
        try:
            self.device = self._setup_device(device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            print(f"Initializing Stable Diffusion on {self.device}")
            print(f"Using precision: {self.dtype}")
            
            self.pipe = self._load_pipeline(model_id)
            self.current_scheduler = "euler_a"
            self.schedulers = {
                "euler_a": ("Euler Ancestral", "Fast, good for creative images"),
                "euler": ("Euler", "Deterministic, consistent results"),
                "ddim": ("DDIM", "Classic, good quality, slower"),
                "dpm_solver": ("DPM Solver", "High quality, efficient"),
                "lms": ("LMS", "Linear multistep, stable")
            }
            print("Stable Diffusion Generator Ready!")
        except Exception as e:
            print(f"Initialization Error: {str(e)}")
            raise

# CELL 4: Core Generator Class - Part 2 (Setup Methods)
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"VRAM: {vram_gb:.1f}GB")
            else:
                device = "cpu"
                print("Using CPU (GPU not available)")
        return torch.device(device)
    
    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            print("Applying Memory Optimizations...")
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("XFormers Attention: Enabled")
            except Exception as e:
                print(f"XFormers: Not available ({e})")
            
            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                    print("Full GPU Loading: Success")
                except RuntimeError as e:
                    print("GPU Memory Limited: Using CPU Offload")
                    pipe.enable_model_cpu_offload()
            else:
                print("Model loaded on CPU.")
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

# CELL 5: Core Generator Class - Part 3 (Scheduler & Generation)
    def set_scheduler(self, scheduler_name: str) -> bool:
        if scheduler_name not in self.schedulers:
            print(f"Unknown scheduler: {scheduler_name}")
            return False
        if scheduler_name == self.current_scheduler:
            return True
            
        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler
        }
        try:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            self.current_scheduler = scheduler_name
            name, desc = self.schedulers[scheduler_name]
            print(f"Scheduler Changed: {name} ({desc})")
            return True
        except Exception as e:
            print(f"Scheduler Error: {e}")
            return False

# CELL 6: Core Generator Class - Part 4 (Image Generation)
    def load_lora_weights(self, lora_name: str):
        if not hasattr(self, 'pipe') or self.pipe is None:
            return "Model not loaded"
        if lora_name == "None" or not lora_name:
            self.pipe.unload_lora_weights()
            return "LoRA unloaded"
        try:
            self.pipe.load_lora_weights(os.path.join("lora_output", lora_name))
            return f"Loaded LoRA: {lora_name}"
        except Exception as e:
            return f"Error loading LoRA: {e}"

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a"
    ) -> Tuple[Image.Image, dict]:        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        self.set_scheduler(scheduler)
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        print(f"Generating: '{prompt[:50]}...'")
        print(f"Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        print(f"Seed: {seed}, Scheduler: {scheduler}")
        
        start_time = time.time()
        try:
            with torch.inference_mode():
                if self.device.type == "cuda" and self.dtype == torch.float16:
                    with autocast(self.device.type):
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
            
            generation_time = time.time() - start_time
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype)
            }
            print(f"Generated in {generation_time:.2f}s")
            return result.images[0], metadata
            
        except torch.cuda.OutOfMemoryError:
            self._cleanup_memory()
            raise RuntimeError(
                "GPU Out of Memory! Try: reducing image size, fewer steps, "
                "or use CPU mode. Current settings may be too demanding."
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
        finally:
            self._cleanup_memory()

# CELL 7: Core Generator Class - Part 5 (Utility Methods)
    def _cleanup_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> dict:
        memory_info = {}
        if self.device.type == "cuda":
            memory_info = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            memory_info = {"device": "cpu", "note": "CPU memory tracking not available"}
        return memory_info
    
    def save_image(self, image: Image.Image, metadata: dict, output_dir: str = "outputs") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd_gen_{timestamp}_s{metadata['seed']}_{metadata['width']}x{metadata['height']}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        
        metadata_file = filepath.replace('.png', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write("Stable Diffusion Generation Metadata\n")
            f.write("=" * 40 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved: {filepath}")
        return filepath

# CELL 8: UI Class - Part 1 (Initialization & Generator Setup)
class StableDiffusionUI:
    def __init__(self):
        self.generator = None
        self.gallery_images = []
        self.generation_history = []
        self.cgan_model = None
    
    def initialize_generator(self, model_choice: str, device_choice: str) -> str:
        try:
            model_map = {
                "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
                "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
                "Realistic Vision (RealVisXL)": "SG161222/RealVisXL_V4.0"
            }
            device_map = {
                "Auto (Recommended)": "auto",
                "GPU (CUDA)": "cuda", 
                "CPU (Slower)": "cpu"
            }
            model_id = model_map.get(model_choice, "runwayml/stable-diffusion-v1-5")
            device = device_map.get(device_choice, "auto")
            
            self.generator = StableDiffusionGenerator(model_id=model_id, device=device)
            memory_info = self.generator.get_memory_usage()
            memory_text = f"Memory Usage: {memory_info}" if memory_info else "Ready!"
            return f"Model loaded successfully!\n{memory_text}"
        except Exception as e:
            return f"Initialization failed: {str(e)}"

# CELL 9: UI Class - Part 2 (Image Generation Handler)
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        scheduler: str,
        seed: int,
        save_image: bool
    ) -> Tuple[Optional[Image.Image], str, str]:
        if self.generator is None:
            return None, "Please initialize the model first!", ""
        if not prompt.strip():
            return None, "Please enter a prompt!", ""
            
        try:
            seed = None if seed == -1 else int(seed)
            image, metadata = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                scheduler=scheduler,
                seed=seed
            )
            
            info_text = self._format_generation_info(metadata)
            saved_path = ""
            if save_image:
                saved_path = self.generator.save_image(image, metadata)
            
            self.generation_history.append(metadata)
            self.gallery_images.append(image)
            
            if len(self.gallery_images) > 10:
                self.gallery_images = self.gallery_images[-10:]
                self.generation_history = self.generation_history[-10:]
            
            return image, info_text, saved_path
        except Exception as e:
            return None, f"Generation failed: {str(e)}", ""

# CELL 10: UI Class - Part 3 (Helper Methods)
    def get_available_loras(self):
        lora_dir = "lora_output"
        if not os.path.exists(lora_dir):
            return ["None"]
        loras = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
        return ["None"] + loras

    def train_lora(self, zip_file, trigger_word, domain, steps, lr):
        if not zip_file:
            yield "Please upload a ZIP file.", None
            return
        import zipfile, shutil, os
        from fine_tuner import LoRAFineTuner
        from PIL import Image as PILImage

        extract_dir = "temp_dataset"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir)
        try:
            with zipfile.ZipFile(zip_file.name, "r") as z:
                z.extractall(extract_dir)
            yield f"Starting LoRA training ({domain} domain)... check console.", None
            tuner = LoRAFineTuner(output_dir="lora_output")
            save_path = tuner.train(
                dataset_dir=extract_dir,
                trigger_word=trigger_word,
                num_steps=int(steps),
                learning_rate=float(lr),
                domain=domain,
            )
            loss_img = None
            loss_path = f"outputs/experiments/lora_loss_{trigger_word}.png"
            if os.path.exists(loss_path):
                loss_img = PILImage.open(loss_path)
            yield f"Training complete!\nSaved: {save_path}\nDomain: {domain}", loss_img
        except Exception as e:
            yield f"Error: {str(e)}", None
        finally:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

    def update_lora_dropdown(self):
        return gr.Dropdown(choices=self.get_available_loras())

    def apply_lora(self, lora_name):
        if self.generator:
            return self.generator.load_lora_weights(lora_name)
        return "Initialize model first"

    def _format_generation_info(self, metadata: dict) -> str:
        return f"""
Generation Complete!

Parameters Used:
- Prompt: {metadata['prompt'][:100]}{'...' if len(metadata['prompt']) > 100 else ''}
- Size: {metadata['width']} x {metadata['height']} pixels
- Steps: {metadata['steps']} (more steps = higher quality, slower)
- Guidance Scale: {metadata['guidance_scale']} (higher = follows prompt more closely)
- Scheduler: {metadata['scheduler']} 
- Seed: {metadata['seed']} (for reproducible results)

Performance:
- Generation Time: {metadata['generation_time']}s
- Device: {metadata['device']}
- Precision: {metadata['dtype']}
"""
    
    def get_example_prompts(self) -> list:
        return [
            ["a serene mountain landscape at sunrise, photorealistic, highly detailed", "blurry, low quality"],
            ["portrait of a wise old wizard, fantasy art, digital painting", "ugly, deformed"],
            ["cyberpunk cityscape at night, neon lights, futuristic", "daytime, bright"],
            ["cute cartoon cat wearing a hat, kawaii style", "realistic, scary"],
            ["abstract geometric patterns, colorful, modern art", "representational, dull colors"]
        ]
    
    def show_scheduler_info(self, scheduler: str) -> str:
        scheduler_info = {
            "euler_a": "Euler Ancestral: Fast and creative, adds slight randomness for variety",
            "euler": "Euler: Deterministic and consistent, same seed = same result", 
            "ddim": "DDIM: Classic scheduler, high quality but slower",
            "dpm_solver": "DPM Solver: Efficient high-quality generation",
            "lms": "LMS: Linear multistep, very stable results"
        }
        return scheduler_info.get(scheduler, "Scheduler information not available")
    
    def get_memory_info(self) -> str:
        if self.generator is None:
            return "Model not loaded"
        try:
            memory_info = self.generator.get_memory_usage()
            if 'allocated_gb' in memory_info:
                return f"""
GPU Memory Usage:
- Allocated: {memory_info['allocated_gb']:.2f}GB
- Reserved: {memory_info['reserved_gb']:.2f}GB  
- Total Available: {memory_info['total_gb']:.2f}GB
- Usage: {(memory_info['allocated_gb']/memory_info['total_gb']*100):.1f}%
                """
            else:
                return "CPU mode - memory tracking not available"
        except:
            return "Memory info unavailable"

    def get_cgan_model(self):
        if self.cgan_model is None:
            from cgan import CGANModel
            self.cgan_model = CGANModel()
            self.cgan_model.load_weights()
        return self.cgan_model

    def train_cgan(self, epochs):
        model = self.get_cgan_model()
        yield "Generating dataset and starting training... Check console for progress.", None, None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import os, io
            from PIL import Image as PILImage

            history = model.train(num_epochs=int(epochs))

            # Plot loss curves
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(history["epochs"], history["g_losses"], label="Generator Loss", color="#4C9BE8", linewidth=2)
            ax.plot(history["epochs"], history["d_losses"], label="Discriminator Loss", color="#E84C4C", linewidth=2)
            ax.set_title("CGAN Training Loss Curves", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            os.makedirs("outputs/experiments", exist_ok=True)
            loss_path = "outputs/experiments/cgan_loss_curves.png"
            fig.savefig(loss_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            loss_img = PILImage.open(loss_path)

            # Generate grid
            grid = model.generate_grid(n_per_class=5, upscale=4)
            grid_path = "outputs/experiments/cgan_shape_grid.png"
            grid.save(grid_path)

            summary = (f"? Training complete! {int(epochs)} epochs\n"
                       f"Final G-Loss: {history['g_losses'][-1]:.4f}  "
                       f"D-Loss: {history['d_losses'][-1]:.4f}\n"
                       f"Saved: cgan_generator.pth | cgan_training_metrics.json")
            yield summary, loss_img, grid
        except Exception as e:
            yield f"Training failed: {e}", None, None

    def generate_cgan_grid(self):
        model = self.get_cgan_model()
        if not model.is_trained:
            return None, "Model not trained yet!"
        try:
            grid = model.generate_grid(n_per_class=5, upscale=4)
            return grid, "? 5-class shape grid generated"
        except Exception as e:
            return None, f"Error: {e}"

    def generate_cgan_shape(self, shape_name):
        model = self.get_cgan_model()
        if not model.is_trained:
            return None, "Model is not trained yet! Please train it first."
        
        shapes = ["Square", "Circle", "Triangle", "Rectangle", "Ellipse"]
        if shape_name not in shapes:
            return None, "Invalid shape"
            
        try:
            idx = shapes.index(shape_name)
            img = model.generate(idx)
            return img, f"Generated a {shape_name}"
        except Exception as e:
            return None, f"Error: {e}"

# CELL 11: UI Class - Part 4 (Interface Creation)

    def run_exp1(self, epochs):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        from PIL import Image as PILImage
        os.makedirs("outputs/experiments", exist_ok=True)
        yield "Training CGAN...", None, None
        try:
            model = self.get_cgan_model()
            history = model.train(num_epochs=int(epochs))
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(history["epochs"], history["g_losses"], label="G Loss", color="#4C9BE8", linewidth=2)
            ax.plot(history["epochs"], history["d_losses"], label="D Loss", color="#E84C4C", linewidth=2)
            ax.set_title("CGAN Loss Curves"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            lp = "outputs/experiments/cgan_loss_curves.png"
            fig.savefig(lp, dpi=120, bbox_inches="tight"); plt.close(fig)
            grid = model.generate_grid(n_per_class=5, upscale=4)
            gp = "outputs/experiments/cgan_shape_grid.png"; grid.save(gp)
            yield (f"? Done! G:{history['g_losses'][-1]:.4f} D:{history['d_losses'][-1]:.4f}\n"
                   f"Saved: {lp}, {gp}"), PILImage.open(lp), grid
        except Exception as e:
            yield f"Error: {e}", None, None

    def run_exp2(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt, time, os
        from PIL import Image as PILImage
        os.makedirs("outputs/experiments", exist_ok=True)
        if self.generator is None:
            yield "Initialize SD model first!", None, None; return
        SCHEDULERS = ["euler_a","euler","ddim","dpm_solver","lms"]
        LABELS = {"euler_a":"Euler-A","euler":"Euler","ddim":"DDIM","dpm_solver":"DPM Solver","lms":"LMS"}
        PROMPT = "a futuristic city skyline at dusk, neon lights, highly detailed, photorealistic"
        results = {}
        for s in SCHEDULERS:
            yield f"Testing {s}...", None, None
            try:
                t0 = time.time()
                img, _ = self.generator.generate_image(PROMPT,"blurry,low quality",512,512,20,7.5,42,s)
                results[s] = {"time":time.time()-t0,"image":img}
            except Exception as e:
                results[s] = {"time":None,"image":None}
        valid = {k:v for k,v in results.items() if v["time"]}
        times = [v["time"] for v in valid.values()]
        labels = [LABELS.get(k,k) for k in valid]
        colors = ["#4C9BE8","#5EBA7D","#E8A44C","#E84C4C","#9B4CE8"][:len(valid)]
        fig, ax = plt.subplots(figsize=(9,4))
        bars = ax.bar(labels, times, color=colors, width=0.6)
        ax.bar_label(bars, labels=[f"{t:.1f}s" for t in times], padding=4)
        ax.set_title("Scheduler Inference Time Comparison"); ax.set_ylabel("Seconds")
        ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        bp = "outputs/experiments/scheduler_benchmark_times.png"
        fig.savefig(bp, dpi=120, bbox_inches="tight"); plt.close(fig)
        n = len(valid)
        fig2, axes = plt.subplots(1,n,figsize=(4*n,4))
        if n==1: axes=[axes]
        for ax2,(s,r) in zip(axes,valid.items()):
            ax2.imshow(r["image"]); ax2.set_title(f"{LABELS.get(s,s)}\n{r['time']:.1f}s"); ax2.axis("off")
        fig2.tight_layout(); gp = "outputs/experiments/scheduler_image_grid.png"
        fig2.savefig(gp, dpi=100, bbox_inches="tight"); plt.close(fig2)
        fastest = min(valid, key=lambda k:valid[k]["time"])
        yield f"? Done! Fastest: {fastest} ({valid[fastest]['time']:.1f}s)", PILImage.open(bp), PILImage.open(gp)

    def run_exp3(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt, os
        from PIL import Image as PILImage
        os.makedirs("outputs/experiments", exist_ok=True)
        if self.generator is None:
            yield "Initialize SD model first!", None; return
        CFG_VALUES = [1,3,5,7,10,15,20]
        PROMPT = "a majestic lion in a golden savannah, dramatic lighting, ultra detailed, 4k"
        NEG = "blurry, low quality, cartoon, ugly"
        imgs = {}
        for cfg in CFG_VALUES:
            yield f"Generating CFG={cfg}...", None
            try:
                img, _ = self.generator.generate_image(PROMPT,NEG,512,512,25,cfg,1337,"euler_a")
                imgs[cfg] = img
            except Exception as e:
                pass
        n = len(imgs)
        ql = {1:"Noisy",3:"Creative",5:"Balanced",7:"Recommended",10:"Prompt-heavy",15:"Over-sat.",20:"Artefacts"}
        fig, axes = plt.subplots(1,n,figsize=(3.5*n,4.5))
        if n==1: axes=[axes]
        for ax,(v,img) in zip(axes,imgs.items()):
            ax.imshow(img); ax.set_title(f"CFG={v}\n{ql.get(v,'')}",fontsize=9); ax.axis("off")
        fig.suptitle(f'CFG Ablation | Seed=1337 | Steps=25',fontsize=11,fontweight="bold")
        fig.tight_layout(); gp = "outputs/experiments/cfg_ablation_grid.png"
        fig.savefig(gp, dpi=110, bbox_inches="tight"); plt.close(fig)
        yield "? CFG ablation complete!", PILImage.open(gp)


    def _get_prompt_encoder(self):
        if not hasattr(self, "_prompt_encoder") or self._prompt_encoder is None:
            from prompt_encoder import PromptEncoder
            self._prompt_encoder = PromptEncoder()
        return self._prompt_encoder

    def pe_encode_ui(self, text):
        if not text.strip():
            return "Please enter a prompt.", "", ""
        try:
            enc = self._get_prompt_encoder()
            r = enc.encode(text)
            rows = []
            for i in range(r["n_tokens"] + 1):
                rows.append(f"| {i:>2} | {r['decoded_tokens'][i]:<20} | {r['token_ids'][i]:>6} | {r['token_norms'][i]:>8.4f} |")
            table = "| # | Token | ID | Norm |\n|---|-------|-----|------|\n" + "\n".join(rows)
            s = r["embedding_stats"]
            stats = (f"Tokens: {r['n_tokens']} / {r['max_length']}  ({'TRUNCATED' if r['truncated'] else 'OK'})\n"
                     f"Mean norm : {s['mean_norm']:.4f} +/- {s['std_norm']:.4f}\n"
                     f"Value range: [{s['global_min']:.4f}, {s['global_max']:.4f}]\n"
                     f"Global mean/std: {s['global_mean']:.4f} / {s['global_std']:.4f}")
            pooled = str(r["pooled_output"][:8].round(4)) + "  ... (768-dim)"
            return table, stats, pooled
        except Exception as e:
            return f"Error: {e}", "", ""

    def pe_compare_ui(self, text_a, text_b):
        if not text_a.strip() or not text_b.strip():
            return "Please enter both prompts."
        try:
            enc = self._get_prompt_encoder()
            res = enc.compare(text_a, text_b)
            sim = res["cosine_similarity"]
            label = "very similar" if sim > 0.9 else "similar" if sim > 0.7 else "different"
            return (f"Cosine Similarity: {sim:.4f}  ({label})\n\n"
                    f"Prompt A  tokens: {res['result_a']['n_tokens']}\n"
                    f"Prompt B  tokens: {res['result_b']['n_tokens']}")
        except Exception as e:
            return f"Error: {e}"

    def pe_export_ui(self, text, out_dir):
        if not text.strip():
            return "Please enter a prompt."
        try:
            enc = self._get_prompt_encoder()
            r   = enc.encode(text)
            paths = enc.export(r, out_dir or "outputs/embeddings")
            return f"Saved:\n  {paths['npy']}\n  {paths['json']}"
        except Exception as e:
            return f"Error: {e}"


    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Educational Stable Diffusion Generator",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("""
            # Educational Stable Diffusion Text-to-Image Generator
            **Learn Generative AI concepts while creating images!**
            """)
            
            with gr.Tab("Setup & Generation"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Setup")
                        model_choice = gr.Dropdown(
                            choices=[
                                "Stable Diffusion 1.5 (Recommended)",
                                "Stable Diffusion 2.1", 
                                "Realistic Vision (RealVisXL)"
                            ],
                            value="Stable Diffusion 1.5 (Recommended)",
                            label="Model Selection"
                        )
                        device_choice = gr.Dropdown(
                            choices=[
                                "Auto (Recommended)",
                                "GPU (CUDA)",
                                "CPU (Slower)"
                            ],
                            value="Auto (Recommended)", 
                            label="Device Selection"
                        )
                        init_btn = gr.Button("Initialize Model", variant="primary")
                        init_status = gr.Textbox(
                            label="Initialization Status",
                            placeholder="Click Initialize Model to start",
                            lines=3
                        )
                    with gr.Column():
                        gr.Markdown("### System Info")
                        memory_btn = gr.Button("Check Memory Usage")
                        memory_info = gr.Textbox(
                            label="Memory Information",
                            placeholder="Click to check memory usage",
                            lines=6
                        )
                
                gr.Markdown("### Image Generation")
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Prompt (Describe what you want)",
                            placeholder="a beautiful landscape painting, oil on canvas, detailed",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (What to avoid)",
                            placeholder="blurry, low quality, bad anatomy",
                            lines=2
                        )
                        generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
                    with gr.Column():
                        with gr.Accordion("Advanced Settings", open=True):
                            with gr.Row():
                                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                                height = gr.Slider(256, 1024, 512, step=64, label="Height")
                            with gr.Row():
                                steps = gr.Slider(10, 100, 20, step=1, label="Inference Steps")
                                guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
                            scheduler = gr.Dropdown(
                                choices=["euler_a", "euler", "ddim", "dpm_solver", "lms"],
                                value="euler_a",
                                label="Scheduler"
                            )
                            scheduler_info = gr.Textbox(
                                label="Scheduler Information",
                                interactive=False,
                                lines=2
                            )
                            with gr.Row():
                                seed = gr.Number(-1, label="Seed")
                                save_image = gr.Checkbox(True, label="Save Generated Images")
                            with gr.Row():
                                lora_dropdown = gr.Dropdown(choices=self.get_available_loras(), value="None", label="Apply LoRA (Custom Style)")
                                refresh_lora_btn = gr.Button("🔄 Refresh")
                                apply_lora_btn = gr.Button("Apply LoRA")
                                lora_status = gr.Textbox(label="LoRA Status", interactive=False)
                
                with gr.Row():
                    output_image = gr.Image(label="Generated Image", type="pil")
                with gr.Row():
                    generation_info = gr.Textbox(
                        label="Generation Information",
                        lines=10,
                        interactive=False
                    )
                    saved_path = gr.Textbox(
                        label="Saved File Path",
                        interactive=False
                    )
            
            with gr.Tab("Fine-Tune Model (LoRA)"):
                gr.Markdown("### Train a Domain-Specific Custom Model with LoRA")
                gr.Markdown("Upload a `.zip` of images. Optionally add `.txt` caption files with the same name as each image for domain-specific prompts.")
                with gr.Row():
                    with gr.Column():
                        dataset_zip = gr.File(label="Upload Dataset (.zip format)", file_types=[".zip"])
                        trigger_word = gr.Textbox(label="Trigger Word", placeholder="e.g., sks, myphoto", info="Unique token injected into captions.")
                        domain_choice = gr.Dropdown(
                            choices=["custom", "artwork", "medical", "product", "portrait"],
                            value="custom",
                            label="Domain",
                            info="Sets caption template. Use 'medical' for MRI/X-ray, 'artwork' for paintings, etc."
                        )
                    with gr.Column():
                        train_steps = gr.Slider(100, 2000, 500, step=100, label="Training Steps", info="More steps = better learning but takes longer.")
                        learning_rate = gr.Number(1e-4, label="Learning Rate")
                        train_btn = gr.Button("Start Fine-Tuning", variant="primary")
                        train_status = gr.Textbox(label="Training Status", lines=3)
                with gr.Row():
                    lora_loss_chart = gr.Image(label="Training Loss Curve", type="pil")
            
            with gr.Tab("CGAN Basic Shapes"):
                gr.Markdown("### Conditional GAN ? Generate Basic Shapes from Text Labels")
                gr.Markdown("Train the CGAN from scratch. Then generate individual shapes or a full 5-class grid with loss curves.")
                with gr.Row():
                    with gr.Column():
                        cgan_epochs = gr.Slider(5, 100, 50, step=5, label="Training Epochs")
                        cgan_train_btn = gr.Button("Train CGAN Model", variant="primary")
                        cgan_train_status = gr.Textbox(label="Training Status", lines=3)
                    with gr.Column():
                        cgan_shape_choice = gr.Dropdown(choices=["Square", "Circle", "Triangle", "Rectangle", "Ellipse"], value="Square", label="Shape Label")
                        cgan_gen_btn = gr.Button("Generate Single Shape")
                        cgan_grid_btn = gr.Button("Generate 5-Class Grid", variant="secondary")
                        cgan_gen_status = gr.Textbox(label="Generation Status")
                with gr.Row():
                    cgan_output_image = gr.Image(label="Single Shape Output", type="pil", image_mode="RGB")
                    cgan_grid_image = gr.Image(label="5-Class Shape Grid", type="pil")
                with gr.Row():
                    cgan_loss_image = gr.Image(label="Training Loss Curves", type="pil")
            
            with gr.Tab("Learning Resources"):
                gr.Markdown("""
                ## Understanding Stable Diffusion
                ### What is Diffusion?
                Diffusion models learn to gradually remove noise from random data.
                ### Key Components:
                **CLIP (Text Encoder)**
                **U-Net (Denoising Network)** 
                **VAE (Variational Autoencoder)**
                **Schedulers**
                ### Parameter Guide:
                **Steps (10-100)**: More steps = higher quality but slower generation
                **Guidance Scale (1-20)**: Higher values make the AI follow your prompt more strictly
                **Seed**: Controls randomness - same seed + settings = same image
                **Resolution**: Higher resolution = more detail but needs more GPU memory
                """)
            
            with gr.Tab("Examples & Gallery"):
                gr.Markdown("### Example Prompts to Try")
                examples = gr.Examples(
                    examples=self.get_example_prompts(),
                    inputs=[prompt, negative_prompt],
                    label="Click any example to load it"
                )
                gr.Markdown("### Recent Generations")
                gallery = gr.Gallery(
                    value=[],
                    label="Your Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
            

            with gr.Tab("? Experiments"):
                gr.Markdown("### Reproducible Internship Experiments")
                gr.Markdown("Run each experiment directly from the UI. Outputs saved to `outputs/experiments/`.")
                with gr.Tab("Exp 1 ? CGAN Analytics"):
                    exp1_epochs = gr.Slider(10, 100, 50, step=10, label="Epochs")
                    exp1_btn = gr.Button("Run CGAN Experiment", variant="primary")
                    exp1_status = gr.Textbox(label="Status", lines=3)
                    with gr.Row():
                        exp1_loss = gr.Image(label="Loss Curves")
                        exp1_grid = gr.Image(label="Shape Grid")
                with gr.Tab("Exp 2 ? Scheduler Benchmark"):
                    gr.Markdown("Requires SD model initialized in Setup tab.")
                    exp2_btn = gr.Button("Run Scheduler Benchmark", variant="primary")
                    exp2_status = gr.Textbox(label="Status", lines=3)
                    with gr.Row():
                        exp2_chart = gr.Image(label="Time Bar Chart")
                        exp2_grid = gr.Image(label="Image Grid")
                with gr.Tab("Exp 3 ? CFG Ablation"):
                    gr.Markdown("Requires SD model initialized in Setup tab.")
                    exp3_btn = gr.Button("Run CFG Ablation", variant="primary")
                    exp3_status = gr.Textbox(label="Status", lines=3)
                    exp3_grid = gr.Image(label="CFG Grid")


            with gr.Tab("Prompt Encoder"):
                gr.Markdown("### Text Prompt Encoder Inspector")
                gr.Markdown(
                    "Inspect how HuggingFace CLIP tokenizes and encodes your prompts into "
                    "768-dim embeddings used by Stable Diffusion. Supports token-level "
                    "analysis, prompt comparison, and embedding export."
                )
                with gr.Tab("Tokenize & Embed"):
                    pe_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="a red circle on a white background",
                        lines=2
                    )
                    pe_btn = gr.Button("Encode Prompt", variant="primary")
                    pe_table   = gr.Markdown(label="Token Table")
                    pe_stats   = gr.Textbox(label="Embedding Statistics", lines=6, interactive=False)
                    pe_pooled  = gr.Textbox(label="Pooled Embedding Preview (768-dim)", interactive=False)
                with gr.Tab("Compare Prompts"):
                    pe_a  = gr.Textbox(label="Prompt A", placeholder="a square", lines=2)
                    pe_b  = gr.Textbox(label="Prompt B", placeholder="a circle", lines=2)
                    pe_cb = gr.Button("Compare Similarity", variant="primary")
                    pe_cr = gr.Textbox(label="Cosine Similarity Result", lines=4, interactive=False)
                with gr.Tab("Export Embedding"):
                    pe_ep  = gr.Textbox(label="Prompt to Export", lines=2)
                    pe_ed  = gr.Textbox(label="Output Directory", value="outputs/embeddings")
                    pe_eb  = gr.Button("Export .npy + .json", variant="primary")
                    pe_er  = gr.Textbox(label="Saved Paths", interactive=False)

            # Event handlers
            init_btn.click(
                fn=self.initialize_generator,
                inputs=[model_choice, device_choice],
                outputs=init_status
            )
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, negative_prompt, width, height, steps, guidance, scheduler, seed, save_image],
                outputs=[output_image, generation_info, saved_path]
            ).then(
                fn=lambda: self.gallery_images,
                outputs=gallery
            )
            scheduler.change(
                fn=self.show_scheduler_info,
                inputs=scheduler,
                outputs=scheduler_info
            )
            memory_btn.click(
                fn=self.get_memory_info,
                outputs=memory_info
            )
            refresh_lora_btn.click(
                fn=self.update_lora_dropdown,
                outputs=lora_dropdown
            )
            apply_lora_btn.click(
                fn=self.apply_lora,
                inputs=lora_dropdown,
                outputs=lora_status
            )
            train_btn.click(
                fn=self.train_lora,
                inputs=[dataset_zip, trigger_word, domain_choice, train_steps, learning_rate],
                outputs=[train_status, lora_loss_chart]
            )
            cgan_train_btn.click(
                fn=self.train_cgan,
                inputs=cgan_epochs,
                outputs=[cgan_train_status, cgan_loss_image, cgan_grid_image]
            )
            cgan_gen_btn.click(
                fn=self.generate_cgan_shape,
                inputs=cgan_shape_choice,
                outputs=[cgan_output_image, cgan_gen_status]
            )
            cgan_grid_btn.click(
                fn=self.generate_cgan_grid,
                outputs=[cgan_grid_image, cgan_gen_status]
            )
        

            exp1_btn.click(fn=self.run_exp1, inputs=exp1_epochs,
                outputs=[exp1_status, exp1_loss, exp1_grid])
            exp2_btn.click(fn=self.run_exp2, outputs=[exp2_status, exp2_chart, exp2_grid])
            exp3_btn.click(fn=self.run_exp3, outputs=[exp3_status, exp3_grid])
        
            pe_btn.click(fn=self.pe_encode_ui, inputs=pe_input,
                outputs=[pe_table, pe_stats, pe_pooled])
            pe_cb.click(fn=self.pe_compare_ui, inputs=[pe_a, pe_b], outputs=pe_cr)
            pe_eb.click(fn=self.pe_export_ui, inputs=[pe_ep, pe_ed], outputs=pe_er)

        return interface

# CELL 12: Launch the Application
ui = StableDiffusionUI()
interface = ui.create_interface()
interface.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    debug=True,
    show_error=True
)

