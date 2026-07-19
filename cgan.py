"""
cgan.py — Conditional GAN for Shape Generation
================================================
Generates 5 basic shapes (Square, Circle, Triangle, Rectangle, Ellipse)
conditioned on a text label.

Architecture:
  - Generator : Embedding(label) + Linear → ConvTranspose2d blocks → 64×64 RGB
  - Discriminator: Conv2d blocks with label embedding concatenated spatially
  - Loss: Binary Cross-Entropy (standard GAN)

Internship Extensions (added on top of base implementation):
  - train() now returns per-epoch loss history dict
  - Saves cgan_training_metrics.json alongside model weights
  - generate_grid(n_per_class) produces a PIL image grid for all 5 classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import gc
import json
from datetime import datetime

# ─── Constants ────────────────────────────────────────────────────────────────
SHAPES = ["Square", "Circle", "Triangle", "Rectangle", "Ellipse"]
NUM_CLASSES = len(SHAPES)
IMG_SIZE = 64
LATENT_DIM = 100


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ShapeDataset(Dataset):
    """
    Synthetically generated dataset of coloured geometric shapes.
    Each sample is a 64×64 RGB image paired with a class label (0-4).
    """

    def __init__(self, num_samples: int = 5000, img_size: int = 64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        print(f"Generating synthetic shape dataset ({self.num_samples} samples)...")
        
        # Base colors for the 5 classes: Red, Green, Blue, Yellow, Cyan
        base_colors = [
            (255, 50, 50),   # Red for Square
            (50, 255, 50),   # Green for Circle
            (50, 50, 255),   # Blue for Triangle
            (255, 255, 50),  # Yellow for Rectangle
            (50, 255, 255)   # Cyan for Ellipse
        ]

        for _ in range(self.num_samples):
            label = np.random.randint(0, NUM_CLASSES)
            img = Image.new("RGB", (self.img_size, self.img_size), color="black")
            draw = ImageDraw.Draw(img)

            # Base size with minimal jitter
            size = 28 + np.random.randint(-2, 3)
            
            # Small random positional offsets
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            
            # Color with small variance
            base_color = base_colors[label]
            color = tuple(
                int(np.clip(c + np.random.randint(-15, 16), 0, 255))
                for c in base_color
            )

            if label == 0:  # Square
                x = (self.img_size - size) // 2 + offset_x
                y = (self.img_size - size) // 2 + offset_y
                draw.rectangle([x, y, x + size, y + size], fill=color)
            elif label == 1:  # Circle
                x = (self.img_size - size) // 2 + offset_x
                y = (self.img_size - size) // 2 + offset_y
                draw.ellipse([x, y, x + size, y + size], fill=color)
            elif label == 2:  # Triangle
                x = (self.img_size - size) // 2 + offset_x
                y = (self.img_size - size) // 2 + offset_y
                draw.polygon(
                    [(x + size // 2, y), (x, y + size), (x + size, y + size)],
                    fill=color,
                )
            elif label == 3:  # Rectangle (wider than tall)
                w = size + 14
                h = size - 6
                x = (self.img_size - w) // 2 + offset_x
                y = (self.img_size - h) // 2 + offset_y
                draw.rectangle([x, y, x + w, y + h], fill=color)
            elif label == 4:  # Ellipse (wider than tall)
                w = size + 14
                h = size - 6
                x = (self.img_size - w) // 2 + offset_x
                y = (self.img_size - h) // 2 + offset_y
                draw.ellipse([x, y, x + w, y + h], fill=color)

            # Normalise pixel values to [-1, 1] for Tanh compatibility
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # C, H, W

            self.data.append(img_tensor)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Conditional Generator.
    Input : noise (LATENT_DIM) + label embedding (50-dim)
    Output: 64×64 RGB image (Tanh activation → range [-1, 1])
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, 50)

        self.init_size = IMG_SIZE // 4  # 16×16 feature map to start
        self.l1 = nn.Sequential(
            nn.Linear(LATENT_DIM + 50, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 16 → 32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 32 → 64
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding with noise vector
        gen_input = torch.cat((self.label_emb(labels), noise), dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


# ─── Discriminator ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Conditional Discriminator.
    Input : 64×64 RGB image + spatially-tiled label embedding
    Output: scalar validity score (Sigmoid → [0, 1])
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, NUM_CLASSES)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3 + NUM_CLASSES, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = IMG_SIZE // 2 ** 4  # 64 / 16 = 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Tile label embedding across spatial dimensions and concatenate
        label_emb = self.label_embedding(labels).view(labels.size(0), NUM_CLASSES, 1, 1)
        label_emb = label_emb.repeat(1, 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, label_emb), dim=1)

        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)


# ─── CGAN Model Wrapper ───────────────────────────────────────────────────────
class CGANModel:
    """
    High-level wrapper for Conditional GAN training and inference.

    Methods
    -------
    train(num_epochs, ...)  → returns loss history dict, saves metrics JSON
    load_weights(path)      → loads pre-trained generator weights
    generate(label_idx)     → generates a single PIL Image
    generate_grid(n)        → generates a PIL grid (5 classes × n samples)
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.is_trained = False
        self.training_history = {"g_losses": [], "d_losses": [], "epochs": []}

    # ── Training ──────────────────────────────────────────────────────────────
    def train(
        self,
        num_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.0002,
        progress_callback=None,
    ) -> dict:
        """
        Train the CGAN.

        Returns
        -------
        dict
            {"g_losses": [...], "d_losses": [...], "epochs": [...]}
            Per-epoch average losses for plotting.
        """
        dataset = ShapeDataset(num_samples=4000)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        adversarial_loss = nn.BCELoss()
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        history = {"g_losses": [], "d_losses": [], "epochs": []}

        for epoch in range(num_epochs):
            g_losses_batch = []
            d_losses_batch = []

            for imgs, labels in dataloader:
                batch_sz = imgs.shape[0]
                # Labels for loss: real=1, fake=0
                valid = torch.ones(batch_sz, 1, device=self.device, requires_grad=False)
                fake  = torch.zeros(batch_sz, 1, device=self.device, requires_grad=False)

                real_imgs = imgs.to(self.device)
                labels    = labels.to(self.device)

                # ── Train Discriminator ───────────────────────────────────────
                optimizer_D.zero_grad()
                
                # Real loss
                real_loss = adversarial_loss(self.discriminator(real_imgs, labels), valid)
                
                # Fake loss
                z = torch.randn(batch_sz, LATENT_DIM, device=self.device)
                gen_imgs = self.generator(z, labels)
                fake_loss = adversarial_loss(
                    self.discriminator(gen_imgs.detach(), labels), fake
                )
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                d_losses_batch.append(d_loss.item())

                # ── Train Generator ──────────────────────────────────────────
                optimizer_G.zero_grad()
                g_loss = adversarial_loss(self.discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                optimizer_G.step()
                g_losses_batch.append(g_loss.item())

            avg_g = sum(g_losses_batch) / len(g_losses_batch)
            avg_d = sum(d_losses_batch) / len(d_losses_batch)

            history["g_losses"].append(avg_g)
            history["d_losses"].append(avg_d)
            history["epochs"].append(epoch + 1)

            print(
                f"[Epoch {epoch+1:03d}/{num_epochs}] "
                f"[D loss: {avg_d:.4f}] [G loss: {avg_g:.4f}]"
            )
            if progress_callback:
                progress_callback(epoch, num_epochs, avg_d, avg_g)

        self.is_trained = True
        self.training_history = history

        # Save model weights
        torch.save(self.generator.state_dict(), "cgan_generator.pth")
        print("✓ Generator weights saved → cgan_generator.pth")

        # Save training metrics as JSON (for reproducibility & README)
        metrics_payload = {
            "trained_at": datetime.now().isoformat(),
            "hyperparameters": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "latent_dim": LATENT_DIM,
                "img_size": IMG_SIZE,
                "num_classes": NUM_CLASSES,
                "shapes": SHAPES,
                "dataset_samples": 4000,
                "optimizer": "Adam (betas=0.5,0.999)",
                "loss_fn": "BCELoss",
            },
            "final_metrics": {
                "final_g_loss": round(history["g_losses"][-1], 6),
                "final_d_loss": round(history["d_losses"][-1], 6),
                "min_g_loss":   round(min(history["g_losses"]), 6),
                "min_d_loss":   round(min(history["d_losses"]), 6),
            },
            "per_epoch": {
                "epochs":   history["epochs"],
                "g_losses": [round(v, 6) for v in history["g_losses"]],
                "d_losses": [round(v, 6) for v in history["d_losses"]],
            },
        }
        with open("cgan_training_metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)
        print("✓ Training metrics saved → cgan_training_metrics.json")

        # Cleanup
        del dataset, dataloader
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return history

    # ── Weight Loading ────────────────────────────────────────────────────────
    def load_weights(self, path: str = "cgan_generator.pth") -> bool:
        if os.path.exists(path):
            self.generator.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.is_trained = True
            print(f"✓ Loaded weights from {path}")
            return True
        return False

    # ── Single Image Generation ───────────────────────────────────────────────
    def generate(self, label_idx: int) -> Image.Image:
        """Generate a single 64×64 PIL image for the given class index."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet! Call train() or load_weights().")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=self.device)
            label = torch.tensor([label_idx], dtype=torch.long, device=self.device)
            gen_img = self.generator(z, label)

        # Rescale from [-1,1] → [0,255]
        img_array = (gen_img.squeeze().cpu().numpy() + 1.0) * 127.5
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_array = np.transpose(img_array, (1, 2, 0))  # H, W, C
        return Image.fromarray(img_array)

    # ── Multi-Class Grid Generation ───────────────────────────────────────────
    def generate_grid(self, n_per_class: int = 5, upscale: int = 4) -> Image.Image:
        """
        Generate a PIL image grid showing n samples for each of the 5 shape classes.

        Parameters
        ----------
        n_per_class : int
            Number of sample images to generate per class (columns).
        upscale : int
            Upscale factor applied to each 64×64 tile for better visibility.

        Returns
        -------
        PIL.Image
            Grid image: rows = classes, columns = samples.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")

        tile_size = IMG_SIZE * upscale  # e.g. 256
        label_height = 28              # pixels reserved for text label row
        padding = 4

        grid_w = n_per_class * (tile_size + padding) + padding
        grid_h = NUM_CLASSES * (tile_size + padding + label_height) + padding

        grid_img = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
        draw = ImageDraw.Draw(grid_img)

        self.generator.eval()
        with torch.no_grad():
            for row, shape_name in enumerate(SHAPES):
                label_idx = row
                # Draw class label on the left
                y_top = padding + row * (tile_size + padding + label_height)
                draw.text(
                    (padding + 2, y_top),
                    shape_name,
                    fill=(220, 220, 220),
                )

                for col in range(n_per_class):
                    z = torch.randn(1, LATENT_DIM, device=self.device)
                    label = torch.tensor(
                        [label_idx], dtype=torch.long, device=self.device
                    )
                    gen_img_tensor = self.generator(z, label)

                    img_array = (gen_img_tensor.squeeze().cpu().numpy() + 1.0) * 127.5
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    img_array = np.transpose(img_array, (1, 2, 0))
                    tile = Image.fromarray(img_array).resize(
                        (tile_size, tile_size), resample=Image.NEAREST
                    )

                    x = padding + col * (tile_size + padding)
                    y = y_top + label_height
                    grid_img.paste(tile, (x, y))

        return grid_img
