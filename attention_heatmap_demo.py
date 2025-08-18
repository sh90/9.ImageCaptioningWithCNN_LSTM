# attention_heatmap_demo.py
"""
# Train the attention model once (if you haven’t yet)
python captioning_with_attention.py

# Then visualize attention on any image from your tiny set (or your own image)
python attention_heatmap_demo.py --image data/tiny/images/dog.jpg
# Save the figure instead of just showing it:
python attention_heatmap_demo.py --image data/tiny/images/dog.jpg --save dog_attention.png
“Each mini-panel is a generated word. The colored overlay shows where the model was looking (attending) to say that word. Notice how ‘dog’ lights up on the dog region and ‘grass’ on the background.”

"""
import os
import argparse
import torch
import numpy as np
from PIL import Image

# Import encoder/decoder + helpers from your attention script
from captioning_with_attention import (
    CNNEncoderSpatial,
    AttnLSTMDecoder,
    visualize_attention_overlay,  # added in the patch above
    BOS, EOS, PAD
)
from torchvision import transforms

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def build_models_from_ckpt(ckpt, device):
    # These dims must match what you used during training
    enc_dim = 512
    emb_dim = 256
    hid_dim = 512

    encoder = CNNEncoderSpatial(enc_dim=enc_dim).to(device).eval()
    decoder = AttnLSTMDecoder(vocab_size=len(ckpt["itos"]),
                              enc_dim=enc_dim,
                              emb_dim=emb_dim,
                              hidden_dim=hid_dim,
                              pad_idx=ckpt["stoi"][PAD]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return encoder, decoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="data/tiny/images/dog.jpg")
    ap.add_argument("--ckpt", type=str, default="checkpoints/tiny_captioner_attention.pt")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save figure (png)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}\nRun captioning_with_attention.py first.")

    ckpt = load_checkpoint(args.ckpt)
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder = build_models_from_ckpt(ckpt, device)

    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = img_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats, grid_hw = encoder(x)  # feats: [1, N, C], grid_hw: (H,W)
        tokens, alphas = decoder.greedy_decode_with_attention(
            feats, bos_idx=stoi[BOS], eos_idx=stoi[EOS], max_len=16
        )

    ids = tokens[0].tolist()
    words = [w for w in (itos[i] for i in ids) if w not in (BOS, EOS, PAD)]
    alphas_np = alphas[0].cpu().numpy()  # [T, N]

    visualize_attention_overlay(args.image, words, alphas_np, grid_hw, out_path=args.save)

if __name__ == "__main__":
    main()
