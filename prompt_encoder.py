"""
prompt_encoder.py  — Standalone Text Prompt Encoder Tool
=========================================================
Uses HuggingFace Transformers to
preprocess text descriptions into tokenized representations and CLIP embedding vectors.
The text-to-image model will use these embeddings as inputs.

Features
--------
- Tokenize any text prompt with CLIP tokenizer
- Display token IDs, decoded tokens, and attention mask
- Compute 768-dim CLIP text embeddings
- Show per-token embedding statistics (norm, mean, std)
- Compare two prompts side-by-side (cosine similarity)
- Export embeddings to .npy / .json
- Works fully standalone (no SD pipeline needed)
"""

import os
import json
import re
import hashlib
import numpy as np
import torch


# ── PromptEncoder class ───────────────────────────────────────────────────────

class PromptEncoder:
    """
    Wraps HuggingFace CLIP tokenizer + text encoder to provide
    detailed inspection of how text prompts become embedding vectors.

    Usage
    -----
    enc = PromptEncoder()
    result = enc.encode("a red circle on a white background")
    enc.display(result)
    """

    MODEL_ID = "openai/clip-vit-large-patch14"   # same as SD 1.5 text encoder

    def __init__(self, model_id: str = None, device: str = "auto", load_model: bool = False):
        self.model_id = model_id or self.MODEL_ID
        self.device   = self._pick_device(device)
        self.load_model = load_model
        self.tokenizer = None
        self.text_model = None

        if not self.load_model:
            print("PromptEncoder using lightweight fallback mode (no CLIP download required).")
            return

        try:
            from transformers import CLIPTokenizer, CLIPTextModel

            print(f"Loading CLIP tokenizer + text encoder from '{self.model_id}' ...")
            print(f"Device: {self.device}")

            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
            self.text_model = CLIPTextModel.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(self.device)
            self.text_model.eval()
            print("PromptEncoder ready.")
        except Exception as exc:
            print(f"Warning: CLIP model could not be loaded ({exc}). Falling back to lightweight mode.")
            self.tokenizer = None
            self.text_model = None

    # ── Device setup ──────────────────────────────────────────────────────────
    def _pick_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _fallback_tokens(self, text: str) -> list:
        return re.findall(r"\b[\w']+\b|[^\w\s]", text.lower())

    def _fallback_embedding(self, text: str, length: int = 768) -> np.ndarray:
        tokens = self._fallback_tokens(text)
        embedding = np.zeros(length, dtype=np.float32)
        if not tokens:
            return embedding

        for idx, token in enumerate(tokens[:64]):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            position = int(digest[:8], 16) % length
            embedding[position] += 1.0 / (idx + 1)

        embedding[0] = len(text)
        embedding[1] = len(tokens)
        embedding[2] = len(text.split())
        embedding /= max(1.0, np.linalg.norm(embedding))
        return embedding

    # ── Core tokenize ─────────────────────────────────────────────────────────
    def tokenize(self, text: str) -> dict:
        """
        Tokenize a prompt and return full token information.

        Returns
        -------
        dict with keys:
          text, token_ids, decoded_tokens, attention_mask,
          n_tokens, truncated, raw_encoding
        """
        if not self.tokenizer:
            tokens = self._fallback_tokens(text)
            return {
                "text": text,
                "token_ids": [ord(ch) % 1000 for ch in text[:77]],
                "decoded_tokens": tokens[:77],
                "attention_mask": [1] * max(1, len(tokens)),
                "n_tokens": len(tokens),
                "max_length": 77,
                "truncated": False,
                "raw_encoding": None,
                "fallback": True,
            }

        max_len = self.tokenizer.model_max_length   # 77 for CLIP

        # Without truncation first — to detect truncation
        raw = self.tokenizer(text, truncation=False, return_tensors="pt")
        n_raw = raw["input_ids"].shape[1]

        # With truncation + padding (standard SD input)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        ids       = enc["input_ids"][0].tolist()
        attn_mask = enc["attention_mask"][0].tolist()

        decoded_tokens = [
            self.tokenizer.convert_ids_to_tokens([tid])[0]
            for tid in ids
        ]

        return {
            "text":            text,
            "token_ids":       ids,
            "decoded_tokens":  decoded_tokens,
            "attention_mask":  attn_mask,
            "n_tokens":        sum(attn_mask),       # real tokens (excl. padding)
            "max_length":      max_len,
            "truncated":       n_raw > max_len,
            "raw_encoding":    enc,
            "fallback":        False,
        }

    # ── Core encode ───────────────────────────────────────────────────────────
    def encode(self, text: str) -> dict:
        """
        Tokenize AND compute 768-dim embeddings for the prompt.

        Returns
        -------
        dict extending tokenize() output with:
          last_hidden_state  : (77, 768) numpy array (per-token embeddings)
          pooled_output      : (768,) numpy array (sentence-level embedding)
          token_norms        : (77,) norms of per-token embeddings
          embedding_stats    : dict of mean/std/min/max for each token
        """
        tok = self.tokenize(text)
        if not self.text_model:
            tokens = self._fallback_tokens(text)
            token_embeddings = np.stack([self._fallback_embedding(token) for token in tokens], axis=0) if tokens else np.zeros((1, 768), dtype=np.float32)
            pooled = token_embeddings.mean(axis=0) if token_embeddings.size else np.zeros(768, dtype=np.float32)
            stats = {
                "mean_norm": float(np.linalg.norm(token_embeddings, axis=-1).mean()) if token_embeddings.size else 0.0,
                "std_norm": float(np.linalg.norm(token_embeddings, axis=-1).std()) if token_embeddings.size else 0.0,
                "global_mean": float(pooled.mean()),
                "global_std": float(pooled.std()),
                "global_min": float(pooled.min()),
                "global_max": float(pooled.max()),
            }
            tok.update({
                "last_hidden_state": token_embeddings,
                "pooled_output": pooled,
                "token_norms": np.linalg.norm(token_embeddings, axis=-1).tolist() if token_embeddings.size else [],
                "embedding_stats": stats,
                "fallback_used": True,
            })
            return tok

        ids  = tok["raw_encoding"]["input_ids"].to(self.device)
        mask = tok["raw_encoding"]["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.text_model(input_ids=ids, attention_mask=mask)

        last_hidden = out.last_hidden_state[0].cpu().numpy()  # (77, 768)
        pooled      = out.pooler_output[0].cpu().numpy()      # (768,)

        # Per-token L2 norm
        token_norms = np.linalg.norm(last_hidden, axis=-1)   # (77,)

        # Per-token stats (only for real tokens)
        n_real = tok["n_tokens"]
        real_embeddings = last_hidden[:n_real]
        stats = {
            "mean_norm":   float(token_norms[:n_real].mean()),
            "std_norm":    float(token_norms[:n_real].std()),
            "global_mean": float(real_embeddings.mean()),
            "global_std":  float(real_embeddings.std()),
            "global_min":  float(real_embeddings.min()),
            "global_max":  float(real_embeddings.max()),
        }

        tok.update({
            "last_hidden_state": last_hidden,
            "pooled_output":     pooled,
            "token_norms":       token_norms.tolist(),
            "embedding_stats":   stats,
            "fallback_used":     False,
        })
        return tok

    # ── Display ───────────────────────────────────────────────────────────────
    def display(self, result: dict, max_tokens: int = 20):
        """Pretty-print tokenization and embedding info."""
        sep = "=" * 62
        print(sep)
        print(f"PROMPT  : {result['text']}")
        print(f"Tokens  : {result['n_tokens']} / {result['max_length']}  "
              f"({'TRUNCATED' if result['truncated'] else 'OK'})")
        print(sep)
        print(f"{'#':>3}  {'Token':<18} {'ID':>6}  {'Mask':>4}  {'Norm':>8}")
        print("-" * 47)
        for i in range(min(result["max_length"], result["n_tokens"] + 2)):
            tok  = result["decoded_tokens"][i]
            tid  = result["token_ids"][i]
            mask = result["attention_mask"][i]
            norm = result["token_norms"][i] if "token_norms" in result else 0.0
            print(f"{i:>3}  {tok:<18} {tid:>6}  {mask:>4}  {norm:>8.4f}")
        if "embedding_stats" in result:
            s = result["embedding_stats"]
            print(sep)
            print("Embedding Statistics (real tokens only):")
            print(f"  Mean token norm : {s['mean_norm']:.4f}  ± {s['std_norm']:.4f}")
            print(f"  Value range     : [{s['global_min']:.4f}, {s['global_max']:.4f}]")
            print(f"  Global mean/std : {s['global_mean']:.4f} / {s['global_std']:.4f}")
        print(sep)

    # ── Compare two prompts ───────────────────────────────────────────────────
    def compare(self, text_a: str, text_b: str) -> dict:
        """
        Compare two prompts by cosine similarity of their pooled embeddings.

        Returns
        -------
        dict: result_a, result_b, cosine_similarity
        """
        ra = self.encode(text_a)
        rb = self.encode(text_b)

        va = ra["pooled_output"]
        vb = rb["pooled_output"]
        cos_sim = float(
            np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        )

        print(f"\nPrompt A: {text_a}")
        print(f"Prompt B: {text_b}")
        print(f"Cosine Similarity: {cos_sim:.4f}  "
              f"({'very similar' if cos_sim > 0.9 else 'similar' if cos_sim > 0.7 else 'different'})")
        return {"result_a": ra, "result_b": rb, "cosine_similarity": cos_sim}

    # ── Export ────────────────────────────────────────────────────────────────
    def export(self, result: dict, out_dir: str = "outputs/embeddings") -> dict:
        """
        Save embedding to .npy (full array) and metadata to .json.

        Returns dict of saved paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        # sanitise prompt to filename
        slug = "".join(c if c.isalnum() else "_" for c in result["text"])[:40]

        npy_path  = os.path.join(out_dir, f"embed_{slug}.npy")
        json_path = os.path.join(out_dir, f"embed_{slug}_meta.json")

        np.save(npy_path, result["last_hidden_state"])
        meta = {
            "text":           result["text"],
            "token_ids":      result["token_ids"],
            "decoded_tokens": result["decoded_tokens"],
            "n_tokens":       result["n_tokens"],
            "truncated":      result["truncated"],
            "embedding_stats": result.get("embedding_stats", {}),
            "pooled_output":  result["pooled_output"].tolist(),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Embedding saved : {npy_path}  ({result['last_hidden_state'].shape})")
        print(f"Metadata saved  : {json_path}")
        return {"npy": npy_path, "json": json_path}

    # ── Batch encode ─────────────────────────────────────────────────────────
    def batch_encode(self, texts: list) -> list:
        """Encode a list of prompts and return list of result dicts."""
        return [self.encode(t) for t in texts]


# ── Gradio UI for the encoder tool ───────────────────────────────────────────

def build_encoder_ui(encoder: PromptEncoder):
    """
    Build and return a Gradio Blocks UI for the PromptEncoder.
    Can be embedded as a tab or launched standalone.
    """
    import gradio as gr

    def _tokenize_ui(text):
        if not text.strip():
            return "Please enter a prompt.", "", ""
        try:
            r = encoder.encode(text)
            # Table of tokens
            rows = []
            for i in range(r["n_tokens"] + 1):  # +1 for EOS
                rows.append(f"| {i:>2} | {r['decoded_tokens'][i]:<18} | "
                            f"{r['token_ids'][i]:>6} | {r['token_norms'][i]:>8.4f} |")
            table = ("| # | Token | ID | Norm |\n"
                     "|---|-------|-----|------|\n" + "\n".join(rows))

            stats = r["embedding_stats"]
            info  = (f"Tokens used: {r['n_tokens']} / {r['max_length']}  "
                     f"({'TRUNCATED' if r['truncated'] else 'OK'})\n"
                     f"Mean token norm : {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}\n"
                     f"Value range     : [{stats['global_min']:.4f}, {stats['global_max']:.4f}]\n"
                     f"Global mean/std : {stats['global_mean']:.4f} / {stats['global_std']:.4f}")

            pooled_preview = str(r["pooled_output"][:8].round(4)) + "  ... (768-dim)"
            return table, info, pooled_preview
        except Exception as e:
            return f"Error: {e}", "", ""

    def _compare_ui(text_a, text_b):
        if not text_a.strip() or not text_b.strip():
            return "Please enter both prompts."
        try:
            res = encoder.compare(text_a, text_b)
            sim = res["cosine_similarity"]
            label = "very similar" if sim > 0.9 else "similar" if sim > 0.7 else "different"
            return (f"Cosine Similarity: {sim:.4f}  ({label})\n\n"
                    f"Prompt A tokens: {res['result_a']['n_tokens']}\n"
                    f"Prompt B tokens: {res['result_b']['n_tokens']}")
        except Exception as e:
            return f"Error: {e}"

    def _export_ui(text, out_dir):
        if not text.strip():
            return "Please enter a prompt."
        try:
            r = encoder.encode(text)
            paths = encoder.export(r, out_dir or "outputs/embeddings")
            return f"Saved:\n  {paths['npy']}\n  {paths['json']}"
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks() as ui:
        gr.Markdown("## Prompt Encoder Inspector\n"
                    "Inspect how CLIP tokenizes and encodes your text prompt into embeddings "
                    "used by Stable Diffusion.")

        with gr.Tab("Tokenize & Embed"):
            inp  = gr.Textbox(label="Text Prompt", placeholder="a red circle on white background", lines=2)
            btn  = gr.Button("Encode Prompt", variant="primary")
            tok_out   = gr.Markdown(label="Token Table")
            stats_out = gr.Textbox(label="Embedding Statistics", lines=5, interactive=False)
            pool_out  = gr.Textbox(label="Pooled Embedding (preview)", interactive=False)
            btn.click(_tokenize_ui, inputs=inp, outputs=[tok_out, stats_out, pool_out])

        with gr.Tab("Compare Two Prompts"):
            ta  = gr.Textbox(label="Prompt A", placeholder="a square", lines=2)
            tb  = gr.Textbox(label="Prompt B", placeholder="a circle", lines=2)
            cb  = gr.Button("Compare", variant="primary")
            cr  = gr.Textbox(label="Similarity Result", lines=5, interactive=False)
            cb.click(_compare_ui, inputs=[ta, tb], outputs=cr)

        with gr.Tab("Export Embedding"):
            ep  = gr.Textbox(label="Prompt to Export", lines=2)
            ed  = gr.Textbox(label="Output Directory", value="outputs/embeddings")
            eb  = gr.Button("Export .npy + .json", variant="primary")
            er  = gr.Textbox(label="Saved Paths", interactive=False)
            eb.click(_export_ui, inputs=[ep, ed], outputs=er)

    return ui


# ── Standalone launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    enc = PromptEncoder()

    # Demo: encode and display a few prompts
    test_prompts = [
        "a red square on a black background",
        "a futuristic city skyline at dusk, neon lights, highly detailed",
        "an MRI scan of a brain with a tumour, clinical, high resolution",
    ]
    for p in test_prompts:
        r = enc.encode(p)
        enc.display(r)
        print()

    # Compare two prompts
    enc.compare("a square", "a circle")

    # Export one embedding
    r = enc.encode(test_prompts[0])
    enc.export(r)

    # Launch Gradio UI
    try:
        ui = build_encoder_ui(enc)
        ui.launch(server_port=7861, share=False)
    except Exception as e:
        print(f"Gradio launch skipped: {e}")
