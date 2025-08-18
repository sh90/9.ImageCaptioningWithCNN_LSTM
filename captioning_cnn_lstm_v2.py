"""

Classic CNN + LSTM

Architecture

Encoder: ResNet50 → compresses image into a single feature vector (fixed size).

Decoder: LSTM → generates caption word by word (sequential, one step at a time).

Training: Needs paired (image, caption) dataset (like COCO/Flickr).

Loss: Cross-entropy between predicted tokens and ground-truth caption tokens.

Inference: Greedy decoding (or beam search) to generate sentences.

Characteristics

Captures only a global summary of the image (no explicit attention unless you add it).

Limited by sequential nature of LSTM (slow, hard to scale).

Needs a lot of labeled data to generalize.

Usually trained from scratch (except CNN backbone).

Teaching Note:
Think of this as the “first generation” of image captioning: encode picture → decode words. Works, but brittle.

"""


# captioning_cnn_lstm.py
"""
A tiny, end-to-end Image Captioning demo:
- CNN Encoder (ResNet50 pretrained on ImageNet) extracts an image vector.
- LSTM Decoder generates a caption token-by-token.
- Trains on a micro dataset (6 images, 2 captions each) for fast demo.

==========================================================
####  Why this project?
Classic captioning shows the **encoder–decoder** idea that underpins machine translation,
speech recognition, and even modern Transformer-based LLMs. We "see" with the encoder,
and we "tell" with the decoder. It's the simplest ‘multimodal’ pipeline you can demo live.
==========================================================
"""

import os, re, json, random
from collections import Counter
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# BLEU metric for a tiny quality check (unigram)
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import certifi, os
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())  # helps HF/requests too


# -------------------------
# Config & reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
#DATA_DIR = "data/tiny"
DATA_DIR = "data/tiny_more_v1"
IM_DIR = os.path.join(DATA_DIR, "images")
CAP_PATH = os.path.join(DATA_DIR, "captions.json")

# Special tokens for text processing
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


# -------------------------
# Tokenization & Vocabulary
# -------------------------
def tokenize(text: str) -> List[str]:
    """
    Lowercase + extract alphabetic tokens (super simple tokenizer).
    """
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(captions: Dict[str, List[str]], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    """
    Build vocabulary (string-to-index and index-to-string).
    Keeps words with frequency >= min_freq.

    ==========================================================
    ####  Vocab & Tokens
    We turn words into integers so the model can learn distributions over them.
    Special tokens:
      <bos>  : "start of sentence"
      <eos>  : "end of sentence"
      <pad>  : padding shorter sequences to a common length
      <unk>  : unknown/rare words (fallback bucket)
    ==========================================================
    """
    all_tokens = []
    for refs in captions.values():
        for c in refs:
            all_tokens.extend(tokenize(c))
    freqs = Counter(all_tokens)

    itos = [PAD, BOS, EOS, UNK] + sorted([w for w, c in freqs.items() if c >= min_freq])
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


# -------------------------
# Dataset
# -------------------------
class TinyCaptionDataset(torch.utils.data.Dataset):
    """
    Returns (image_tensor, caption_tensor). We replicate images with
    multiple reference captions — a common setup in captioning datasets.
    """
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
        toks = [bos_idx] + toks + [eos_idx]  # add start/end tokens
        return img, torch.tensor(toks, dtype=torch.long)


def collate_fn(batch):
    """
    Pads sequences in a batch and prepares (inputs, targets) for teacher forcing.

    ==========================================================
    #### Teacher Forcing
    During training, the decoder receives the **ground truth previous word**
    as input (not its own prediction). This stabilizes learning and speeds it up.
    At inference, we don't have ground truth — we feed back our **predictions**.
    ==========================================================
    """
    imgs, seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    # inputs exclude last token; targets exclude first token → aligned next-word prediction
    return imgs, seqs[:, :-1], seqs[:, 1:]


# -------------------------
# Models
# -------------------------
class CNNEncoder(nn.Module):
    """
    ResNet50 (pretrained) → Global Average Pooling → Linear projection.

    ==========================================================
    ####  Transfer Learning
    We reuse a CNN already trained on ImageNet (1M+ images). We **freeze**
    those weights so, on a tiny dataset, we don't overfit. The CNN becomes
    a generic "visual feature extractor" for our captioning task.
    ==========================================================
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove classifier head; keep convolutional trunk → [B, 2048, 1, 1]
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, images):
        feats = self.backbone(images).squeeze(-1).squeeze(-1)  # [B, 2048]
        return self.proj(feats)  # [B, D]


class LSTMDecoder(nn.Module):
    """
    LSTM generates one token at a time, conditioned on:
      - previous token (embedded)
      - its hidden state (initialized from the image embedding)

    ==========================================================
    ####  Why LSTM here?
    Captions are sequences. LSTMs model sequential dependencies and were the
    standard before Transformers. They still make a great teaching example.
    ==========================================================
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

        # Map the image embedding to initial LSTM states (h0, c0)
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)

    def forward(self, enc, seq_in):
        """
        Training forward pass with teacher forcing.
        enc: [B, D]  image embedding
        seq_in: [B, T]  previous ground-truth tokens
        """
        B = enc.size(0)
        h0 = torch.tanh(self.init_h(enc)).unsqueeze(0)  # [1, B, H]
        c0 = torch.tanh(self.init_c(enc)).unsqueeze(0)
        emb = self.embed(seq_in)                        # [B, T, E]
        out, _ = self.lstm(emb, (h0, c0))               # [B, T, H]
        logits = self.fc(out)                           # [B, T, V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, enc, bos_idx: int, eos_idx: int, max_len: int = 16):
        """
        Inference: start with <bos>, repeatedly choose the argmax token,
        and stop when <eos> appears (or when max_len is reached).

        ==========================================================
        ####  Greedy vs. Beam Search
        Greedy = pick the most likely next word each time (fast, simple).
        Beam  = keep top-k partial sequences and expand them (better quality).
        For a live demo, greedy is perfect — one line, fast, predictable.
        ==========================================================
        """
        B = enc.size(0)
        h = torch.tanh(self.init_h(enc)).unsqueeze(0)
        c = torch.tanh(self.init_c(enc)).unsqueeze(0)
        inp = torch.full((B, 1), bos_idx, dtype=torch.long, device=enc.device)
        outputs = []
        for _ in range(max_len):
            emb = self.embed(inp[:, -1:])
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.fc(out[:, -1])
            next_tok = logits.argmax(-1, keepdim=True)   # greedy step
            outputs.append(next_tok)
            inp = torch.cat([inp, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
        return torch.cat(outputs, dim=1)


# -------------------------
# Helpers
# -------------------------
def detok(ids: List[int], itos: List[str]) -> str:
    """Turn token IDs back into a readable sentence (drop special tokens)."""
    toks = []
    for i in ids:
        w = itos[i]
        if w in (BOS, EOS, PAD):  # hide markup
            continue
        toks.append(w)
    return " ".join(toks)


def bleu1(pred: str, refs: List[str]) -> float:
    """
    BLEU-1 (unigram) — super simple sanity check on tiny data.
    Higher is "better", but don’t over-interpret on 6 images.
    """
    smooth = SmoothingFunction().method3
    pred_tokens = pred.split()
    ref_tokens = [r.split() for r in refs]
    return sentence_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)


# -------------------------
# Main training + inference
# -------------------------
def main():
    # Ensure NLTK punkt exists (quietly)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # 1) Load captions
    if not os.path.exists(CAP_PATH):
        raise FileNotFoundError("Run tiny_dataset_setup.py first to create captions.json")
    with open(CAP_PATH) as f:
        captions_dict = json.load(f)

    # 2) Build vocab from tiny captions
    global stoi, itos, pad_idx, bos_idx, eos_idx, unk_idx
    stoi, itos = build_vocab(captions_dict, min_freq=1)
    pad_idx = stoi[PAD]; bos_idx = stoi[BOS]; eos_idx = stoi[EOS]; unk_idx = stoi[UNK]
    print(f"Vocab size: {len(itos)} | PAD={pad_idx}, BOS={bos_idx}, EOS={eos_idx}, UNK={unk_idx}")

    # 3) Image preprocessing — match ImageNet normalization for ResNet
    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 4) Dataset split (80/20) — tiny but enough to demo train/val
    ds = TinyCaptionDataset(IM_DIR, captions_dict, img_tfms)
    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    split = max(1, int(0.8 * n))
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True,  collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 5) Build models: ResNet encoder (frozen) + trainable LSTM decoder
    encoder = CNNEncoder(embed_dim=256).to(device)
    decoder = LSTMDecoder(vocab_size=len(itos), embed_dim=256, hidden_dim=512, pad_idx=pad_idx).to(device)

    # Train decoder and the small projection layer (not the frozen CNN trunk)
    params = list(decoder.parameters()) + list(encoder.proj.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def run_epoch(loader, train=True):
        """
        One pass over data. If train=True → backprop; else just eval.
        Loss is next-token cross-entropy averaged over tokens.
        """
        encoder.train(mode=train)
        decoder.train(mode=train)
        total, count = 0.0, 0

        for images, seq_in, seq_tar in loader:
            images, seq_in, seq_tar = images.to(device), seq_in.to(device), seq_tar.to(device)

            with torch.set_grad_enabled(train):
                enc = encoder(images)                 # [B, D]
                logits = decoder(enc, seq_in)         # [B, T, V]
                loss = criterion(logits.reshape(-1, len(itos)), seq_tar.reshape(-1))

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total += loss.item()
            count += 1

        return total / max(1, count)

    # 6) Train for a few epochs (fast on CPU)
    EPOCHS = 8
    for e in range(1, EPOCHS + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"Epoch {e:02d} | train {tr:.3f} | val {va:.3f}")

    # 7) Save checkpoint (so you can resume or demo later)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "stoi": stoi, "itos": itos},
               "checkpoints/tiny_captioner.pt")
    print("Saved checkpoints/tiny_captioner.pt")

    # 8) Inference helper (greedy)
    def caption_image(path: str) -> str:
        img = Image.open(path).convert("RGB")
        x = img_tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            enc = encoder(x)
            out = decoder.greedy_decode(enc, bos_idx=bos_idx, eos_idx=eos_idx, max_len=16)[0].tolist()
        return detok(out, itos)

    # 9) Demo on every tiny image (print BLEU-1 against its references)
    print("\n=== Inference on tiny images ===")
    for fname in sorted(os.listdir(IM_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(IM_DIR, fname)
        pred = caption_image(path)
        refs = captions_dict.get(fname, [])
        score = bleu1(pred, refs) if refs else 0.0
        print(f"{fname:10s} | Pred: {pred:40s} | BLEU-1: {score:.3f}")

    # ==========================================================
    # ####  Where does this go next?
    # 1) Add **Attention** (e.g., "Show, Attend and Tell") to focus on image regions.
    # 2) Replace LSTM with a **Transformer decoder** (parallelizable, better long-range).
    # 3) Pretrain on large datasets; or jump to **vision-language models** (BLIP, LLaVA).
    # 4) Use **beam search** or **nucleus sampling** for richer, more fluent captions.
    # 5) Evaluate with **BLEU-4**, **CIDEr**, **METEOR** on real datasets (COCO/Flickr).
    # ==========================================================

if __name__ == "__main__":
    main()
