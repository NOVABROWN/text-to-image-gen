"""
fine_tuner.py  — LoRA Fine-Tuning for Stable Diffusion
=======================================================
Upgrades over original:
  1. Caption file support  — reads per-image .txt captions for domain-specific
     prompts (e.g. "an MRI scan of a brain with tumour").
     Falls back to "a photo of <trigger_word>" when no caption exists.
  2. Domain preset prompts — artwork / medical / product / custom modes.
  3. Training loss curve   — per-step loss saved as PNG + JSON for README.
"""

import os
import json
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# ── Domain caption templates ────────────────────────────────────────────────
DOMAIN_TEMPLATES = {
    "artwork":   "a painting of {trigger}, detailed artwork, oil on canvas",
    "medical":   "a medical image of {trigger}, clinical scan, highly detailed",
    "product":   "a product photo of {trigger}, studio lighting, white background",
    "portrait":  "a portrait of {trigger}, professional photography, sharp focus",
    "custom":    "a photo of {trigger}",
}


class CustomImageDataset(Dataset):
    """
    Dataset that supports:
      - Per-image caption .txt files (same name as image, .txt extension)
      - Domain template strings with {trigger} placeholder
      - Fallback to generic 'a photo of <trigger_word>' prompt
    """

    def __init__(
        self,
        image_dir: str,
        tokenizer,
        size: int = 512,
        trigger_word: str = "sks",
        domain: str = "custom",
    ):
        self.image_dir = image_dir
        self.trigger_word = trigger_word
        self.domain = domain
        self.tokenizer = tokenizer
        self.size = size

        self.image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        template = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["custom"])
        self.default_caption = template.replace("{trigger}", trigger_word)
        print(f"Dataset: {len(self.image_paths)} images | domain='{domain}' | "
              f"default caption='{self.default_caption}'")

    def _get_caption(self, img_path: str) -> str:
        """Read per-image caption file if it exists, else return default."""
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_path):
            cap = open(caption_path, encoding="utf-8").read().strip()
            if cap:
                return cap
        return self.default_caption

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = self._get_caption(img_path)
        inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": image, "input_ids": inputs.input_ids[0], "caption": caption}


class LoRAFineTuner:
    """
    LoRA fine-tuner for Stable Diffusion.

    New features:
      - domain parameter selects caption template
      - caption .txt files per image are respected
      - per-step loss history saved as PNG + JSON
    """

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "lora_output",
    ):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(
        self,
        dataset_dir: str,
        trigger_word: str,
        num_steps: int = 500,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        domain: str = "custom",
        progress_callback=None,
    ) -> str:
        """
        Train LoRA adapters on the custom dataset.

        Parameters
        ----------
        dataset_dir   : Directory with images (and optional .txt captions)
        trigger_word  : Unique token injected into captions (e.g. 'sks')
        num_steps     : Total gradient update steps
        learning_rate : Adam learning rate
        batch_size    : Images per batch (1 recommended for <8 GB VRAM)
        domain        : One of artwork | medical | product | portrait | custom
        progress_callback : fn(step, total, loss)

        Returns
        -------
        str — path to saved LoRA weights
        """
        if self.device == "cpu":
            raise RuntimeError("CUDA GPU is required for LoRA fine-tuning.")

        print(f"Loading SD pipeline from '{self.base_model_id}' ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        unet          = pipe.unet
        text_encoder  = pipe.text_encoder
        vae           = pipe.vae
        tokenizer     = pipe.tokenizer
        noise_sched   = DDPMScheduler.from_pretrained(self.base_model_id, subfolder="scheduler")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            lora_dropout=0.0,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        unet.to(self.device, dtype=torch.float32)
        vae.to(self.device, dtype=torch.float16)
        text_encoder.to(self.device, dtype=torch.float16)

        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)
            print("Optimizer: 8-bit AdamW")
        except ImportError:
            optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
            print("Optimizer: standard AdamW")

        dataset    = CustomImageDataset(dataset_dir, tokenizer, trigger_word=trigger_word, domain=domain)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"Training: {num_steps} steps | lr={learning_rate} | domain={domain}")
        unet.train()
        global_step = 0
        loss_history = []          # (step, loss)

        while global_step < num_steps:
            for batch in dataloader:
                if global_step >= num_steps:
                    break

                pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
                input_ids    = batch["input_ids"].to(self.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise      = torch.randn_like(latents)
                bsz        = latents.shape[0]
                timesteps  = torch.randint(
                    0, noise_sched.config.num_train_timesteps, (bsz,), device=self.device
                ).long()
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                loss_val = loss.item()
                loss_history.append((global_step, loss_val))

                if global_step % 10 == 0:
                    print(f"Step {global_step}/{num_steps}  loss={loss_val:.4f}")
                    if progress_callback:
                        progress_callback(global_step, num_steps, loss_val)

        print("Training complete — saving LoRA weights ...")
        save_path = os.path.join(self.output_dir, f"lora_{trigger_word}.safetensors")
        unet.save_pretrained(save_path)
        print(f"Saved: {save_path}")

        # ── Save loss curve PNG + JSON ──────────────────────────────────────
        self._save_loss_curve(loss_history, trigger_word, domain)

        # Cleanup
        del unet, vae, text_encoder, pipe
        torch.cuda.empty_cache()
        gc.collect()

        return save_path

    # ── Loss curve helper ──────────────────────────────────────────────────
    def _save_loss_curve(self, loss_history, trigger_word, domain):
        """Save training loss as PNG chart + JSON log."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps  = [s for s, _ in loss_history]
            losses = [l for _, l in loss_history]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, losses, color="#4C9BE8", linewidth=1.5, label="MSE Loss")
            # Smoothed trend
            if len(losses) > 20:
                import numpy as np
                window = max(5, len(losses) // 20)
                smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
                ax.plot(steps[window-1:], smooth, color="#E84C4C", linewidth=2,
                        linestyle="--", label=f"Smoothed (w={window})")
            ax.set_title(
                f"LoRA Fine-Tuning Loss  |  trigger='{trigger_word}'  |  domain='{domain}'",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()

            os.makedirs("outputs/experiments", exist_ok=True)
            png_path  = f"outputs/experiments/lora_loss_{trigger_word}.png"
            json_path = f"outputs/experiments/lora_loss_{trigger_word}.json"
            fig.savefig(png_path, dpi=130, bbox_inches="tight")
            plt.close(fig)

            with open(json_path, "w") as f:
                json.dump({"trigger": trigger_word, "domain": domain,
                           "steps": steps, "losses": losses}, f, indent=2)

            print(f"Loss curve saved: {png_path}")
            print(f"Loss data saved : {json_path}")
        except Exception as e:
            print(f"Could not save loss curve: {e}")
