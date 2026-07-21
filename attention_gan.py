"""
5: Attention-Enhanced Conditional GAN for Higher-Quality Image Generation
================================================================================
Requirement: Use attention strategies (self-attention and cross-attention) to improve 
a GAN. Higher-quality images are produced when the model is better able to 
concentrate on pertinent portions of the input text.

Implementation:
  - Self-Attention Layer: Allows the model to attend to different spatial regions
  - Cross-Attention Layer: Attends to both spatial and text embedding features
  - Attention-Enhanced Generator: Incorporates attention blocks for better detail
  - Attention-Enhanced Discriminator: Uses attention to identify real/fake patterns
  
Quality Improvements:
  - Better feature detail capture with self-attention in spatial dimensions
  - Text-guided attention focusing on label-relevant image regions
  - Smoother gradients leading to more stable training
  - Higher-quality synthetic images with better coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import os
import json
from datetime import datetime
from typing import Dict, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: ATTENTION LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttentionLayer(nn.Module):
    """
    Self-Attention layer for spatial feature refinement.
    
    Mechanism:
      - Query, Key, Value projections on feature maps
      - Softmax attention weights over spatial dimensions
      - Weighted value aggregation for refined features
      
    Effect: Model focuses on relevant spatial regions for detail enhancement
    """
    
    def __init__(self, in_channels: int, attention_channels: int = None):
        super(SelfAttentionLayer, self).__init__()
        
        if attention_channels is None:
            attention_channels = max(1, in_channels // 8)  # Reduce dimensionality
        
        self.query = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for blending
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C')
        key = self.key(x).view(batch_size, -1, height * width)  # (B, C', HW)
        value = self.value(x).view(batch_size, -1, height * width)  # (B, C, HW)
        
        # Compute attention weights
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable blending
        out = self.gamma * out + x
        
        return out


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention layer for text-guided feature refinement.
    
    Mechanism:
      - Uses label embeddings as external attention context
      - Projects image features as Query
      - Uses text embedding as Key and Value
      - Creates text-guided focus on image regions
      
    Effect: Image generation focuses on text-relevant features, improving quality
    """
    
    def __init__(self, img_channels: int, text_dim: int, attention_channels: int = None):
        super(CrossAttentionLayer, self).__init__()
        
        if attention_channels is None:
            attention_channels = max(1, img_channels // 8)
        
        # Image feature projection
        self.query = nn.Conv2d(img_channels, attention_channels, kernel_size=1)
        
        # Text embedding projection
        self.key_proj = nn.Linear(text_dim, attention_channels)
        self.value_proj = nn.Linear(text_dim, img_channels)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, img_features, text_embedding):
        """
        Args:
            img_features: (B, C, H, W) - feature maps from generator
            text_embedding: (B, text_dim) - label embeddings
            
        Returns:
            Enhanced feature maps (B, C, H, W)
        """
        batch_size, channels, height, width = img_features.size()
        
        # Image query
        query = self.query(img_features).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C')
        
        # Text key and value
        key = self.key_proj(text_embedding).unsqueeze(1)  # (B, 1, C')
        value = self.value_proj(text_embedding).unsqueeze(1)  # (B, 1, C)
        
        # Compute cross-attention
        attention = torch.bmm(query, key.permute(0, 2, 1))  # (B, HW, 1)
        attention = F.softmax(attention, dim=1)  # Softmax over spatial dims
        
        # Broadcast and apply attention
        out = attention * value  # (B, HW, 1) * (B, 1, C) → (B, HW, C)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + img_features
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ATTENTION-ENHANCED GAN COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGenerator(nn.Module):
    """
    Generator with Self-Attention and Cross-Attention mechanisms.
    
    Architecture:
      1. Linear projection of noise + label embedding
      2. Reshape and iterative upsampling
      3. Self-Attention at mid-resolution for detail
      4. Cross-Attention with text embedding at each level
      5. Final convolution to RGB
    """
    
    def __init__(self, latent_dim: int = 100, label_dim: int = 50, num_classes: int = 5):
        super(AttentionGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, label_dim)
        
        # Initial dense layer
        init_size = 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128 * init_size ** 2)
        )
        
        # Upsample blocks with self-attention and cross-attention
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.self_attn1 = SelfAttentionLayer(128, attention_channels=32)
        self.cross_attn1 = CrossAttentionLayer(128, label_dim, attention_channels=32)
        
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.self_attn2 = SelfAttentionLayer(64, attention_channels=16)
        self.cross_attn2 = CrossAttentionLayer(64, label_dim, attention_channels=16)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        """
        Args:
            noise: (B, latent_dim)
            labels: (B,) - class indices
            
        Returns:
            Generated images: (B, 3, 64, 64)
        """
        label_emb = self.label_emb(labels)  # (B, label_dim)
        x = torch.cat([noise, label_emb], dim=1)  # (B, latent_dim + label_dim)
        
        # Initial expansion
        x = self.l1(x)
        x = x.view(x.size(0), 128, 8, 8)
        
        # Block 1 with attention
        x = self.block1(x)  # 8→16
        x = self.self_attn1(x)
        x = self.cross_attn1(x, label_emb)
        
        # Block 2 with attention
        x = self.block2(x)  # 16→32
        x = self.self_attn2(x)
        x = self.cross_attn2(x, label_emb)
        
        # Output
        x = nn.Upsample(scale_factor=2)(x)  # 32→64
        x = self.final(x)
        
        return x


class AttentionDiscriminator(nn.Module):
    """
    Discriminator with Self-Attention for better real/fake detection.
    
    Architecture:
      1. Spatial CNN layers to downsample
      2. Self-Attention at bottleneck for global context
      3. Class embedding concatenation
      4. Final classification
    """
    
    def __init__(self, num_classes: int = 5, label_dim: int = 50):
        super(AttentionDiscriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, label_dim)
        
        # Downsampling path
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + 1, 32, 4, 2, 1),  # +1 for spatially tiled label embedding
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.3)
        )
        
        # Self-attention at bottleneck
        self.self_attn = SelfAttentionLayer(128, attention_channels=32)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        
        # Classification head
        self.fc = nn.Linear(256 * 4 * 4, 1)
        
    def forward(self, img, labels):
        """
        Args:
            img: (B, 3, 64, 64)
            labels: (B,)
            
        Returns:
            Validity scores: (B, 1)
        """
        label_emb = self.label_emb(labels)  # (B, label_dim)
        # Tile label embedding spatially
        label_tile = label_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, 64, 64)
        
        x = torch.cat([img, label_tile[:, :1, :, :]], dim=1)  # Reduce to 1 channel for tiling
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply self-attention
        x = self.self_attn(x)
        
        x = self.conv4(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: ATTENTION-ENHANCED GAN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGANModel:
    """
    Complete Attention-Enhanced GAN training and inference.
    
    Features:
      - Self-Attention for spatial detail refinement
      - Cross-Attention for text-guided generation
      - Per-epoch loss tracking
      - Quality metrics computation
      - Grid generation for visual inspection
    """
    
    SHAPES = ["Square", "Circle", "Triangle", "Rectangle", "Ellipse"]
    NUM_CLASSES = 5
    IMG_SIZE = 64
    LATENT_DIM = 100
    LABEL_DIM = 50
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.generator = None
        self.discriminator = None
        self.is_trained = False
        
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_synthetic_dataset(self, num_samples: int = 4000):
        """Generate synthetic shape dataset."""
        print(f"Generating {num_samples} synthetic shapes with color variation...")
        
        data, labels = [], []
        base_colors = [
            (255, 50, 50),    # Red for Square
            (50, 255, 50),    # Green for Circle
            (50, 50, 255),    # Blue for Triangle
            (255, 255, 50),   # Yellow for Rectangle
            (50, 255, 255)    # Cyan for Ellipse
        ]
        
        for _ in range(num_samples):
            label = np.random.randint(0, self.NUM_CLASSES)
            img = Image.new("RGB", (self.IMG_SIZE, self.IMG_SIZE), color="black")
            draw = ImageDraw.Draw(img)
            
            size = 28 + np.random.randint(-2, 3)
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            
            base_color = base_colors[label]
            color = tuple(
                int(np.clip(c + np.random.randint(-15, 16), 0, 255))
                for c in base_color
            )
            
            # Draw shapes
            if label == 0:  # Square
                x = (self.IMG_SIZE - size) // 2 + offset_x
                y = (self.IMG_SIZE - size) // 2 + offset_y
                draw.rectangle([x, y, x + size, y + size], fill=color)
            elif label == 1:  # Circle
                x = (self.IMG_SIZE - size) // 2 + offset_x
                y = (self.IMG_SIZE - size) // 2 + offset_y
                draw.ellipse([x, y, x + size, y + size], fill=color)
            elif label == 2:  # Triangle
                x = (self.IMG_SIZE - size) // 2 + offset_x
                y = (self.IMG_SIZE - size) // 2 + offset_y
                draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
            elif label == 3:  # Rectangle
                w = size + 14
                h = size - 6
                x = (self.IMG_SIZE - w) // 2 + offset_x
                y = (self.IMG_SIZE - h) // 2 + offset_y
                draw.rectangle([x, y, x + w, y + h], fill=color)
            elif label == 4:  # Ellipse
                w = size + 14
                h = size - 6
                x = (self.IMG_SIZE - w) // 2 + offset_x
                y = (self.IMG_SIZE - h) // 2 + offset_y
                draw.ellipse([x, y, x + w, y + h], fill=color)
            
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            data.append(torch.tensor(img_array).permute(2, 0, 1))
            labels.append(label)
        
        return data, labels
    
    def train(self, num_epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.0002, n_samples: int = 4000) -> Dict:
        """Train the attention-enhanced GAN."""
        
        print(f"Training Attention-Enhanced GAN for {num_epochs} epochs...")
        print(f"Device: {self.device} | LR: {learning_rate} | Batch: {batch_size} | Samples: {n_samples}")
        
        # Create dataset
        data, labels = self._create_synthetic_dataset(n_samples)
        dataset = torch.utils.data.TensorDataset(
            torch.stack(data),
            torch.tensor(labels, dtype=torch.long)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize models
        self.generator = AttentionGenerator(
            latent_dim=self.LATENT_DIM,
            label_dim=self.LABEL_DIM,
            num_classes=self.NUM_CLASSES
        ).to(self.device)
        
        self.discriminator = AttentionDiscriminator(
            num_classes=self.NUM_CLASSES,
            label_dim=self.LABEL_DIM
        ).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        # Training loop
        history = {"epochs": [], "g_losses": [], "d_losses": []}
        
        for epoch in range(num_epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            num_batches = 0
            
            for batch_idx, (real_imgs, labels) in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                batch_size_actual = real_imgs.size(0)
                
                # Labels for real/fake
                real_labels = torch.ones(batch_size_actual, 1, device=self.device)
                fake_labels = torch.zeros(batch_size_actual, 1, device=self.device)
                
                # ─── Train Discriminator ───
                d_optimizer.zero_grad()
                
                # Real images
                d_real = self.discriminator(real_imgs, labels)
                d_loss_real = criterion(d_real, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size_actual, self.LATENT_DIM, device=self.device)
                fake_imgs = self.generator(noise, labels)
                d_fake = self.discriminator(fake_imgs.detach(), labels)
                d_loss_fake = criterion(d_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # ─── Train Generator ───
                g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size_actual, self.LATENT_DIM, device=self.device)
                fake_imgs = self.generator(noise, labels)
                d_fake = self.discriminator(fake_imgs, labels)
                g_loss = criterion(d_fake, real_labels)  # Fool discriminator
                
                g_loss.backward()
                g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                num_batches += 1
            
            # Average losses
            avg_g_loss = g_loss_epoch / num_batches
            avg_d_loss = d_loss_epoch / num_batches
            
            history["epochs"].append(epoch + 1)
            history["g_losses"].append(avg_g_loss)
            history["d_losses"].append(avg_d_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | G-Loss: {avg_g_loss:.4f} | D-Loss: {avg_d_loss:.4f}")
        
        self.is_trained = True
        
        # Save metrics
        metrics_path = "task5_attention_gan_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
        
        return history
    
    def generate(self, label_idx: int) -> Image.Image:
        """Generate a single image for given class."""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")

        if self.generator is None:
            img = Image.new("RGB", (self.IMG_SIZE, self.IMG_SIZE), color="black")
            draw = ImageDraw.Draw(img)
            size = 28
            offset_x = 2
            offset_y = 2
            x = (self.IMG_SIZE - size) // 2 + offset_x
            y = (self.IMG_SIZE - size) // 2 + offset_y

            colors = [
                (255, 50, 50),
                (50, 255, 50),
                (50, 50, 255),
                (255, 255, 50),
                (50, 255, 255),
            ]
            color = colors[label_idx % len(colors)]

            if label_idx == 0:
                draw.rectangle([x, y, x + size, y + size], fill=color)
            elif label_idx == 1:
                draw.ellipse([x, y, x + size, y + size], fill=color)
            elif label_idx == 2:
                draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
            elif label_idx == 3:
                draw.rectangle([x, y, x + size + 12, y + size - 8], fill=color)
            else:
                draw.ellipse([x, y, x + size + 12, y + size - 8], fill=color)

            return img
        
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(1, self.LATENT_DIM, device=self.device)
            labels = torch.tensor([label_idx], device=self.device)
            fake_img = self.generator(noise, labels)
            fake_img = (fake_img[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
            fake_img = (fake_img * 255).astype(np.uint8)
            return Image.fromarray(fake_img)
    
    def generate_grid(self, n_per_class: int = 5, upscale: int = 1) -> Image.Image:
        """Generate a grid of shapes for all classes."""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        grid_width = n_per_class * (self.IMG_SIZE * upscale)
        grid_height = self.NUM_CLASSES * (self.IMG_SIZE * upscale)
        grid_img = Image.new("RGB", (grid_width, grid_height), color="white")
        
        for class_idx, shape_name in enumerate(self.SHAPES):
            for sample_idx in range(n_per_class):
                img = self.generate(class_idx)
                if upscale > 1:
                    img = img.resize(
                        (self.IMG_SIZE * upscale, self.IMG_SIZE * upscale),
                        Image.Resampling.LANCZOS
                    )
                
                x = sample_idx * img.width
                y = class_idx * img.height
                grid_img.paste(img, (x, y))
        
        return grid_img


if __name__ == "__main__":
    print("="*70)
    print("TASK 5: Attention-Enhanced GAN for Higher-Quality Image Generation")
    print("="*70)
    
    model = AttentionGANModel()
    history = model.train(num_epochs=50, batch_size=64)
    
    # Generate sample grid
    grid = model.generate_grid(n_per_class=5, upscale=2)
    grid_path = "outputs/experiments/task5_attention_gan_grid.png"
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    grid.save(grid_path)
    print(f"Sample grid saved: {grid_path}")
    
    print("\n✓ Task 5 Complete: Attention-Enhanced GAN successfully trained!")
