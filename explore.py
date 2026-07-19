"""
4: Dataset Exploration
Datasets:
  - Oxford-102 Flowers (torchvision) for image analysis
  - SBU Captions (HuggingFace) for text description analysis
  - Synthetic fallback captions if HF datasets are unavailable
Outputs:
  task4_flowers_class_dist.png  - class distribution bar chart
  task4_flowers_gallery.png     - image gallery with text descriptions
  task4_flickr_stats.png        - caption length statistics
  task4_dashboard.png           - combined summary dashboard
"""

import sys
import os
import random
import textwrap
import numpy as np

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import datasets

# ============================================================
#  PART A - Oxford-102 Flowers (Image Analysis)
# ============================================================
print("=" * 60)
print("  PART A - Oxford-102 Flowers (torchvision)")
print("=" * 60)

FLOWERS_ROOT = "./data/flowers102"
os.makedirs(FLOWERS_ROOT, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Official Oxford-102 class names (first 20 shown in gallery)
FLOWER_LABELS = {
    0: "pink primrose",           1: "hard-leaved pocket orchid",
    2: "Canterbury bells",        3: "sweet pea",
    4: "wild geranium",           5: "tiger lily",
    6: "moon orchid",             7: "bird of paradise",
    8: "monkshood",               9: "globe thistle",
    10: "snapdragon",             11: "colts foot",
    12: "king protea",            13: "spear thistle",
    14: "yellow iris",            15: "globe-flower",
    16: "purple coneflower",      17: "peruvian lily",
    18: "balloon flower",         19: "giant white arum lily",
}

NUM_SAMPLES_FL = 0
sample_images  = {}
class_counts   = Counter()

try:
    flowers = dsets.Flowers102(
        root=FLOWERS_ROOT, split="train", download=True, transform=transform
    )
    NUM_SAMPLES_FL = len(flowers)

    print(f"  [OK] Dataset loaded successfully")
    print(f"  Total training images : {NUM_SAMPLES_FL}")
    print(f"  Number of classes     : 102")

    # Walk first 500 images for statistics
    MAX_INSPECT = 500
    for i in range(min(MAX_INSPECT, NUM_SAMPLES_FL)):
        img_tensor, label = flowers[i]
        class_counts[label] += 1
        if label not in sample_images:
            sample_images[label] = transforms.ToPILImage()(img_tensor)

    avg_per_class = np.mean(list(class_counts.values()))
    print(f"  Avg images/class (first {MAX_INSPECT}): {avg_per_class:.1f}")
    print(f"  Tensor shape (C,H,W)  : {list(flowers[0][0].shape)}")
    print(f"  Resized resolution    : 128 x 128 px (uniform)")

    # --- Figure 1: Class Distribution Bar Chart ---
    fig1, ax1 = plt.subplots(figsize=(13, 4))
    sorted_cls = sorted(class_counts.items())
    ax1.bar(
        [str(c) for c, _ in sorted_cls],
        [cnt for _, cnt in sorted_cls],
        color="#7C5CBF", edgecolor="white", linewidth=0.3
    )
    ax1.axhline(avg_per_class, color="#FFD700", linestyle="--",
                linewidth=1.5, label=f"Mean = {avg_per_class:.1f}")
    ax1.set_title("Oxford-102 Flowers - Class Distribution (first 500 samples)",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Class ID")
    ax1.set_ylabel("Sample Count")
    ax1.tick_params(axis='x', labelsize=6)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig("task4_flowers_class_dist.png", dpi=120)
    print("  Saved: task4_flowers_class_dist.png")
    plt.close(fig1)

    # --- Figure 2: Image Gallery with Text Descriptions ---
    GALLERY_N = 12
    collected_labels = sorted(sample_images.keys())[:GALLERY_N]

    fig2, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig2.patch.set_facecolor("#1a1a2e")
    fig2.suptitle(
        "Oxford-102 Flowers - Image Gallery with Text Descriptions",
        fontsize=14, fontweight="bold", color="white"
    )

    for idx, ax in enumerate(axes.flatten()):
        if idx < len(collected_labels):
            lbl  = collected_labels[idx]
            img  = sample_images[lbl]
            name = FLOWER_LABELS.get(lbl, f"Flower Class {lbl}")
            desc = f"Class {lbl}: {name}\n128x128 px | Flowers102 dataset"
            ax.imshow(img)
            ax.set_title(textwrap.fill(desc, 28), fontsize=8, color="white",
                         pad=4, backgroundcolor="#1a1a2e")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    fig2.savefig("task4_flowers_gallery.png", dpi=130, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    print("  Saved: task4_flowers_gallery.png")
    plt.close(fig2)

except Exception as ex:
    print(f"  [ERR] Error with Flowers102: {ex}")

# ============================================================
#  PART B - Text Description Analysis
# ============================================================
print()
print("=" * 60)
print("  PART B - Text Caption Analysis")
print("=" * 60)

cap_lengths  = []
captions_raw = []

# Try SBU Captions on HuggingFace
try:
    print("  Trying: sbu_captions ...")
    ds = datasets.load_dataset("sbu_captions", split="train", streaming=True)
    row_count = 0
    for row in ds:
        if row_count >= 200:
            break
        cap = row.get("caption", "")
        if cap:
            cap_lengths.append(len(str(cap).split()))
            captions_raw.append(str(cap))
        row_count += 1
    if captions_raw:
        print(f"  [OK] Loaded {len(captions_raw)} captions from sbu_captions")
except Exception as e:
    print(f"  [SKIP] sbu_captions: {e}")

# Fallback: synthetic captions from Oxford-102 flower names
if not captions_raw:
    print("  [FALLBACK] Generating synthetic flower text descriptions ...")
    random.seed(42)
    templates = [
        "A close-up photograph of a {name}, showing its vivid {color} petals in full bloom.",
        "An image of a {name} flower in a garden with soft natural lighting.",
        "This photo captures the delicate structure of a {name} flower.",
        "A macro shot of a {name} with its distinctive shape and rich {color} color.",
        "Botanical photograph of a {name} taken during spring bloom season.",
        "A beautiful {color} {name} growing in a meadow on a sunny afternoon.",
        "High-resolution image of a {name}, showcasing intricate petal details.",
        "The {name} is known for its {color} blossoms and elegant appearance.",
    ]
    colors = ["red", "purple", "yellow", "pink", "white", "orange", "blue", "violet"]
    all_labels = list(FLOWER_LABELS.items()) * 12
    for lbl, name in all_labels[:200]:
        color = random.choice(colors)
        tmpl  = random.choice(templates)
        cap   = tmpl.format(name=name, color=color)
        captions_raw.append(cap)
        cap_lengths.append(len(cap.split()))
    print(f"  Generated {len(captions_raw)} synthetic captions.")

print(f"  Total captions analyzed  : {len(captions_raw)}")
print(f"  Avg caption length       : {np.mean(cap_lengths):.2f} words")
print(f"  Min / Max caption length : {min(cap_lengths)} / {max(cap_lengths)} words")
print(f"  Std deviation            : {np.std(cap_lengths):.2f}")

# --- Figure 3: Caption Length Statistics ---
try:
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Text Description Analysis - Caption Length Statistics",
                  fontsize=14, fontweight="bold")

    ax3a.hist(cap_lengths, bins=25, color="#E84393", edgecolor="white", linewidth=0.4)
    ax3a.axvline(np.mean(cap_lengths), color="#FFD700", linestyle="--", linewidth=1.8,
                 label=f"Mean = {np.mean(cap_lengths):.1f} words")
    ax3a.set_xlabel("Words per Caption")
    ax3a.set_ylabel("Frequency")
    ax3a.set_title("Caption Length Distribution")
    ax3a.legend()

    ax3b.boxplot(cap_lengths, vert=True, patch_artist=True,
                 boxprops=dict(facecolor="#7C5CBF", color="white"),
                 medianprops=dict(color="#FFD700", linewidth=2),
                 whiskerprops=dict(color="white"),
                 capprops=dict(color="white"),
                 flierprops=dict(marker="o", color="#E84393", markersize=4))
    ax3b.set_ylabel("Words per Caption")
    ax3b.set_title("Caption Length Box Plot")
    ax3b.set_xticks([])

    plt.tight_layout()
    fig3.savefig("task4_flickr_stats.png", dpi=120)
    print("  Saved: task4_flickr_stats.png")
    plt.close(fig3)
except Exception as ex:
    print(f"  [ERR] Figure 3 error: {ex}")

# ============================================================
#  PART C - Combined Summary Dashboard
# ============================================================
print()
print("=" * 60)
print("  PART C - Combined Statistics Dashboard")
print("=" * 60)

try:
    fig4 = plt.figure(figsize=(15, 8))
    fig4.patch.set_facecolor("#0f0f1e")

    # --- Panel 1: Oxford-102 stats (horizontal bar) ---
    ax_stats = fig4.add_subplot(2, 3, 1)
    ax_stats.set_facecolor("#1a1a2e")
    s_labels = ["Num Classes", "Train Samples", "Img Width", "Img Height"]
    s_values = [102, NUM_SAMPLES_FL if NUM_SAMPLES_FL else 1020, 128, 128]
    s_colors = ["#7C5CBF", "#5BC0DE", "#E84393", "#FFD700"]
    bars = ax_stats.barh(s_labels, s_values, color=s_colors)
    for bar, val in zip(bars, s_values):
        ax_stats.text(
            bar.get_width() * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"  {val}", va="center", color="white", fontsize=9, fontweight="bold"
        )
    ax_stats.set_title("Oxford-102 Flowers\nDataset Statistics",
                       color="white", fontsize=10, fontweight="bold")
    ax_stats.tick_params(colors="white", labelsize=9)
    for spine in ax_stats.spines.values():
        spine.set_edgecolor("#333")

    # --- Panel 2: Sample flower image ---
    ax_img = fig4.add_subplot(2, 3, 2)
    ax_img.set_facecolor("#1a1a2e")
    if sample_images:
        first_label = sorted(sample_images.keys())[0]
        ax_img.imshow(sample_images[first_label])
        name = FLOWER_LABELS.get(first_label, f"Class {first_label}")
        ax_img.set_title(f"Sample Image\nClass {first_label}: {name}",
                         color="white", fontsize=9)
    ax_img.axis("off")

    # --- Panel 3: Resolution pie ---
    ax_pie = fig4.add_subplot(2, 3, 3)
    ax_pie.set_facecolor("#1a1a2e")
    ax_pie.pie([100], labels=["128 x 128 px"], colors=["#7C5CBF"],
               autopct="%1.0f%%",
               textprops={"color": "white", "fontsize": 10},
               wedgeprops={"edgecolor": "white"})
    ax_pie.set_title("Image Resolution\n(Uniform after resize)",
                     color="white", fontsize=10)

    # --- Panel 4: Caption histogram ---
    ax_hist = fig4.add_subplot(2, 3, 4)
    ax_hist.set_facecolor("#1a1a2e")
    if cap_lengths:
        ax_hist.hist(cap_lengths, bins=20, color="#E84393", edgecolor="#0f0f1e")
        ax_hist.axvline(np.mean(cap_lengths), color="#FFD700", linestyle="--",
                        linewidth=1.8, label=f"Mean={np.mean(cap_lengths):.1f}w")
        ax_hist.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax_hist.set_title("Caption Length Distribution", color="white", fontsize=10)
    ax_hist.set_xlabel("Words per Caption", color="white")
    ax_hist.set_ylabel("Frequency", color="white")
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#333")

    # --- Panel 5-6: Sample captions text ---
    ax_text = fig4.add_subplot(2, 3, (5, 6))
    ax_text.set_facecolor("#1a1a2e")
    ax_text.axis("off")
    ax_text.set_title("Sample Text Descriptions",
                      color="#FFD700", fontsize=10, fontweight="bold", loc="left")

    if captions_raw:
        lines = []
        for j in range(min(7, len(captions_raw))):
            wrapped = textwrap.fill(captions_raw[j], 70)
            lines.append(f"[{j+1}] {wrapped}")
        sample_cap_text = "\n\n".join(lines)
        ax_text.text(0.01, 0.90, sample_cap_text,
                     transform=ax_text.transAxes,
                     fontsize=8, color="white", va="top",
                     fontfamily="monospace", linespacing=1.5)

    fig4.suptitle("Task 4 - Dataset Exploration Summary Dashboard",
                  fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    fig4.savefig("task4_dashboard.png", dpi=130, bbox_inches="tight",
                 facecolor=fig4.get_facecolor())
    print("  Saved: task4_dashboard.png")
    plt.close("all")

except Exception as ex:
    print(f"  [ERR] Dashboard error: {ex}")

# ============================================================
print()
print("=" * 60)
print("  DONE - All visualizations saved!")
print("=" * 60)
print("  task4_flowers_class_dist.png  - Class distribution bar chart")
print("  task4_flowers_gallery.png     - Image gallery with descriptions")
print("  task4_flickr_stats.png        - Caption length statistics")
print("  task4_dashboard.png           - Combined summary dashboard")
print("=" * 60)
