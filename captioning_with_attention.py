# captioning_with_attention.py
"""
Image Captioning with Attention (Show, Attend and Tell style)
------------------------------------------------------------
- Encoder: ResNet50 (pretrained), keep spatial feature map (not just a single vector)
- Decoder: LSTM with Bahdanau/Additive attention over spatial features
- Dataset: the same tiny 6-image set from tiny_dataset_setup.py


- Unlike the classic CNN+LSTM (single global image vector), here we keep a grid of features.
- At each word step, the decoder computes an attention distribution over the grid and forms a
  context vector — "looking" at the relevant region before predicting the next word.
"""

import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision import models, transforms
from PIL import Image

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# -------------------------
# Config
# -------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/tiny"
IM_DIR = os.path.join(DATA_DIR, "images")
CAP_PATH = os.path.join(DATA_DIR, "captions.json")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

# -------------------------
# Tokenization & Vocab
# -------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())

def build_vocab(captions: Dict[str, List[str]], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    all_tokens = []
    for refs in captions.values():
        for c in refs:
            all_tokens.extend(tokenize(c))
    freqs = Counter(all_tokens)
    itos = [PAD, BOS, EOS, UNK] + sorted([w for w, c in freqs.items() if c >= min_freq])
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_overlay(image_path, words, alphas, grid_hw, out_path=None, cols=6):
    """
    Overlay attention heatmaps per word on the image.

    Args:
      image_path: path to the original RGB image
      words: list[str] tokens produced (without <bos>/<eos>/<pad>)
      alphas: np.ndarray of shape [T, N] where N = H*W (softmaxed)
      grid_hw: (H, W) spatial size from the encoder (e.g., (7,7))
      out_path: optional path to save the figure; if None -> just show
      cols: number of columns in the subplot grid
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((224, 224))  # match training size
    H, W = grid_hw
    T = len(words)
    rows = int(np.ceil(T / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for t, word in enumerate(words):
        plt.subplot(rows, cols, t + 1)
        plt.axis("off")
        plt.title(word, fontsize=10)

        # alpha_t: [N] -> [H, W] -> upsample to image size
        alpha_t = alphas[t].reshape(H, W)
        alpha_img = Image.fromarray((alpha_t / alpha_t.max() * 255).astype(np.uint8))
        alpha_img = alpha_img.resize(img.size, resample=Image.BILINEAR)

        # show base image then overlay heatmap
        plt.imshow(img)
        plt.imshow(alpha_img, cmap="jet", alpha=0.35)  # overlay
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved attention grid to {out_path}")
    else:
        plt.show()


# -------------------------
# Dataset
# -------------------------
class TinyCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, captions_dict: Dict[str, List[str]], transform):
        self.items = []
        for fname, refs in captions_dict.items():
            for c in refs:
                self.items.append((os.path.join(img_dir, fname), c))
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, caption = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        toks = [stoi.get(t, unk_idx) for t in tokenize(caption)]
        toks = [bos_idx] + toks + [eos_idx]
        return img, torch.tensor(toks, dtype=torch.long)

def collate_fn(batch):
    imgs, seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    return imgs, seqs[:, :-1], seqs[:, 1:]

# -------------------------
# Encoder: keep spatial features
# -------------------------
class CNNEncoderSpatial(nn.Module):
    """
    ResNet50 up to the last conv block → spatial feature map [B, C, H, W]
    We'll project channels to a smaller D for attention.
    """
    def __init__(self, enc_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Keep all layers up to (but not including) avgpool & fc
        self.cnn = nn.Sequential(*list(base.children())[:-2])  # -> [B, 2048, H=7, W=7] for 224x224
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.proj = nn.Conv2d(2048, enc_dim, kernel_size=1)  # [B, enc_dim, H, W]

    def forward(self, images):
        fmap = self.cnn(images)           # [B, 2048, 7, 7]
        fmap = self.proj(fmap)            # [B, 512, 7, 7] (enc_dim)
        B, C, H, W = fmap.shape
        feats = fmap.view(B, C, H * W).transpose(1, 2)  # [B, N=H*W, C]
        return feats, (H, W)  # return flattened features + grid size for viz if needed

# -------------------------
# Additive Attention (Bahdanau)
# -------------------------
class AdditiveAttention(nn.Module):
    """
    score_i = v^T tanh(W_h h_t + W_v v_i)
    where:
      h_t : decoder hidden state at time t, [B, H]
      v_i : i-th spatial feature vector, [B, C]
    """
    def __init__(self, enc_dim, dec_hidden, attn_dim=256):
        super().__init__()
        self.W_h = nn.Linear(dec_hidden, attn_dim)
        self.W_v = nn.Linear(enc_dim, attn_dim)
        self.v   = nn.Linear(attn_dim, 1)

    def forward(self, h_t, feats):
        """
        h_t: [B, H]
        feats: [B, N, C]
        returns:
          context: [B, C]
          alpha:   [B, N] (attention weights over spatial positions)
        """
        B, N, C = feats.shape
        h_exp = self.W_h(h_t).unsqueeze(1).expand(B, N, -1)    # [B,N,A]
        v_proj = self.W_v(feats)                               # [B,N,A]
        e = self.v(torch.tanh(h_exp + v_proj)).squeeze(-1)     # [B,N]
        alpha = torch.softmax(e, dim=-1)                       # [B,N]
        context = (alpha.unsqueeze(-1) * feats).sum(dim=1)     # [B,C]
        return context, alpha

# -------------------------
# Decoder with Attention
# -------------------------
class AttnLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, enc_dim=512, emb_dim=256, hidden_dim=512, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTMCell(emb_dim + enc_dim, hidden_dim)  # input = [word_emb ; context]
        self.attn  = AdditiveAttention(enc_dim=enc_dim, dec_hidden=hidden_dim, attn_dim=256)
        self.init_h = nn.Linear(enc_dim, hidden_dim)
        self.init_c = nn.Linear(enc_dim, hidden_dim)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, feats):
        """
        Initialize h0, c0 from the mean of spatial features.
        feats: [B, N, C]
        """
        mean_enc = feats.mean(dim=1)                   # [B, C]
        h0 = torch.tanh(self.init_h(mean_enc))         # [B, H]
        c0 = torch.tanh(self.init_c(mean_enc))         # [B, H]
        return h0, c0

    def forward(self, feats, seq_in):
        """
        feats:  [B, N, C] spatial features
        seq_in: [B, T]    input tokens (teacher forcing)
        Returns logits: [B, T, V]
        """
        B, N, C = feats.shape
        T = seq_in.size(1)
        h, c = self.init_state(feats)
        outs = []
        for t in range(T):
            # Attention on spatial features using current hidden state
            context, _ = self.attn(h, feats)          # [B, C]
            emb = self.embed(seq_in[:, t])            # [B, E]
            inp = torch.cat([emb, context], dim=-1)   # [B, E+C]
            h, c = self.lstm(inp, (h, c))             # step
            outs.append(self.fc(h).unsqueeze(1))      # [B,1,V]
        return torch.cat(outs, dim=1)                  # [B,T,V]

    @torch.no_grad()
    def greedy_decode(self, feats, bos_idx, eos_idx, max_len=16):
        B, N, C = feats.shape
        h, c = self.init_state(feats)
        inp_tok = torch.full((B,), bos_idx, dtype=torch.long, device=feats.device)
        outputs = []
        for _ in range(max_len):
            context, _ = self.attn(h, feats)            # [B, C]
            emb = self.embed(inp_tok)                   # [B, E]
            step_in = torch.cat([emb, context], dim=-1) # [B, E+C]
            h, c = self.lstm(step_in, (h, c))
            logits = self.fc(h)                         # [B, V]
            next_tok = logits.argmax(dim=-1)            # [B]
            outputs.append(next_tok.unsqueeze(1))
            inp_tok = next_tok
            if (next_tok == eos_idx).all():
                break
        return torch.cat(outputs, dim=1)                # [B, <=max_len]
    @torch.no_grad()
    def greedy_decode_with_attention(self, feats, bos_idx, eos_idx, max_len=16):
        """
        Same as greedy_decode(), but also returns attention weights (alphas) for each step.
        Returns:
          tokens: [B, <=max_len]
          alphas: [B, T, N]  (T = generated length, N = H*W spatial locations)
        """
        B, N, C = feats.shape
        h, c = self.init_state(feats)
        inp_tok = torch.full((B,), bos_idx, dtype=torch.long, device=feats.device)
        outputs = []
        alphas_all = []

        for _ in range(max_len):
            context, alpha = self.attn(h, feats)         # alpha: [B, N]
            emb = self.embed(inp_tok)                    # [B, E]
            step_in = torch.cat([emb, context], dim=-1)  # [B, E+C]
            h, c = self.lstm(step_in, (h, c))
            logits = self.fc(h)                           # [B, V]
            next_tok = logits.argmax(dim=-1)             # [B]
            outputs.append(next_tok.unsqueeze(1))
            alphas_all.append(alpha.unsqueeze(1))        # [B,1,N]
            inp_tok = next_tok
            if (next_tok == eos_idx).all():
                break

        tokens = torch.cat(outputs, dim=1)               # [B, T]
        alphas = torch.cat(alphas_all, dim=1)            # [B, T, N]
        return tokens, alphas

# -------------------------
# Helpers
# -------------------------
def detok(ids: List[int], itos: List[str]) -> str:
    toks = []
    for i in ids:
        w = itos[i]
        if w in (BOS, EOS, PAD): continue
        toks.append(w)
    return " ".join(toks)

def bleu1(pred: str, refs: List[str]) -> float:
    smooth = SmoothingFunction().method3
    pred_tokens = pred.split()
    ref_tokens = [r.split() for r in refs]
    return sentence_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)

# -------------------------
# Main
# -------------------------
def main():
    # Ensure NLTK data is present (quiet no-op if already downloaded)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    if not os.path.exists(CAP_PATH):
        raise FileNotFoundError("Run tiny_dataset_setup.py first to create captions.json")

    with open(CAP_PATH) as f:
        captions_dict = json.load(f)

    global stoi, itos, pad_idx, bos_idx, eos_idx, unk_idx
    stoi, itos = build_vocab(captions_dict, min_freq=1)
    pad_idx = stoi[PAD]; bos_idx = stoi[BOS]; eos_idx = stoi[EOS]; unk_idx = stoi[UNK]
    print(f"Vocab size: {len(itos)}")

    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = TinyCaptionDataset(IM_DIR, captions_dict, img_tfms)
    n = len(ds)
    idxs = list(range(n)); random.shuffle(idxs)
    split = max(1, int(0.8 * n))
    train_ds = torch.utils.data.Subset(ds, idxs[:split])
    val_ds   = torch.utils.data.Subset(ds, idxs[split:])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True,  collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Build encoder/decoder
    enc_dim = 512
    encoder = CNNEncoderSpatial(enc_dim=enc_dim).to(device)
    decoder = AttnLSTMDecoder(vocab_size=len(itos), enc_dim=enc_dim, emb_dim=256, hidden_dim=512, pad_idx=pad_idx).to(device)

    # Train decoder only (encoder is frozen)
    params = list(decoder.parameters()) + list(encoder.proj.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def run_epoch(loader, train=True):
        encoder.train(mode=train)
        decoder.train(mode=train)
        total, count = 0.0, 0
        for images, seq_in, seq_tar in loader:
            images, seq_in, seq_tar = images.to(device), seq_in.to(device), seq_tar.to(device)
            with torch.set_grad_enabled(train):
                feats, _ = encoder(images)               # [B, N, C]
                logits = decoder(feats, seq_in)          # [B, T, V]
                loss = criterion(logits.reshape(-1, len(itos)), seq_tar.reshape(-1))
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); count += 1
        return total / max(1, count)

    EPOCHS = 8
    for e in range(1, EPOCHS+1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"Epoch {e:02d} | train {tr:.3f} | val {va:.3f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "stoi": stoi, "itos": itos},
               "checkpoints/tiny_captioner_attention.pt")
    print("Saved checkpoints/tiny_captioner_attention.pt")

    @torch.no_grad()
    def caption_image(path: str) -> str:
        img = Image.open(path).convert("RGB")
        x = img_tfms(img).unsqueeze(0).to(device)
        feats, _ = encoder(x)
        out = decoder.greedy_decode(feats, bos_idx=bos_idx, eos_idx=eos_idx, max_len=16)[0].tolist()
        return detok(out, itos)

    print("\n=== Inference on tiny images (Attention) ===")
    for fname in sorted(os.listdir(IM_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(IM_DIR, fname)
        pred = caption_image(path)
        refs = captions_dict.get(fname, [])
        score = bleu1(pred, refs) if refs else 0.0
        print(f"{fname:10s} | Pred: {pred:40s} | BLEU-1: {score:.3f}")

if __name__ == "__main__":
    main()
    # --- Attention viz demo on one image (optional) ---
    sample_fname = sorted([f for f in os.listdir(IM_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])[0]
    sample_path = os.path.join(IM_DIR, sample_fname)
    x = img_tfms(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
    feats, grid_hw = encoder(x)  # feats: [1, N, C], grid_hw: (H,W)
    ids, alphas = decoder.greedy_decode_with_attention(feats, bos_idx, eos_idx, max_len=16)
    # convert to clean words and strip specials
    gen_ids = ids[0].tolist()
    words = [w for w in (itos[i] for i in gen_ids) if w not in (BOS, EOS, PAD)]
    visualize_attention_overlay(sample_path, words, alphas[0].cpu().numpy(), grid_hw, out_path="attention_grid.png")

