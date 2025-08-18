# blip_transformer_demo.py
"""
BLIP (Bootstrapping Language-Image Pretraining) Demo
----------------------------------------------------
This script shows two BLIP use-cases:
  1) Image Captioning (Salesforce/blip-image-captioning-base)
  2) Visual Question Answering (Salesforce/blip-vqa-base)

Run examples:
  python blip_transformer_demo.py
  python blip_transformer_demo.py --image data/tiny/images/dog.jpg
  python blip_transformer_demo.py --image https://images.unsplash.com/photo-1517849845537-4d257902454a?q=80&w=1200
  python blip_transformer_demo.py --image data/tiny/images/dog.jpg --question "What animal is this?"

Key differences vs. CNN+LSTM:
- Uses a Vision Transformer (ViT) encoder (patch embeddings, attention over regions)
- Uses a Transformer decoder (parallel attention over tokens, strong language modeling)
- Pretrained on *millions* of image-text pairs with multiple objectives
- Works zero-shot or few-shot without task-specific training
"""

import argparse
import io
import sys
from typing import Optional

import torch
from PIL import Image
import requests

# Transformers (Hugging Face) provides BLIP models + processors
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,
)

# ------------------------------
# Device setup
# ------------------------------
def get_device() -> str:
    """
    Pick CUDA if available for speed. Fall back to CPU otherwise.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Image loading helpers
# ------------------------------
def load_image(path_or_url: str) -> Image.Image:
    """
    Accept either a local path or an HTTP(S) URL.
    Converts to RGB for model compatibility.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


# ------------------------------
# BLIP Captioning
# ------------------------------
def load_caption_model(device: str = "cpu"):
    """
    Load the BLIP captioning model & processor.
    Model card: Salesforce/blip-image-captioning-base
    """
    # ==========================================================
    # Teaching Note: Model & Processor
    # - The "processor" bundles the image transforms & tokenization
    # - The "model" is a Vision-Language Transformer that can generate text
    # ==========================================================
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name,
                                                         use_safetensors=True).to(device).eval()
    return processor, model


@torch.no_grad()
def caption_image(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: str = "cpu",
    prompt: Optional[str] = None,
    max_new_tokens: int = 20,
    num_beams: int = 1,
    do_sample: bool = False,
    top_p: float = 0.9,
):
    """
    Generate a caption. You can:
      - Use greedy decoding (num_beams=1, do_sample=False)
      - Use beam search (num_beams>1)
      - Use sampling (do_sample=True, top_p controls nucleus sampling)

    ==========================================================
    Teaching Note: Decoding Strategies
      - Greedy: fast, picks argmax each step (deterministic).
      - Beam search: explores multiple alternatives -> often better.
      - Sampling: introduces randomness (more diverse, less safe).
    ==========================================================
    """
    # Optional "prompt" can steer style, e.g., "a photo of", "a detailed photo of"
    # This is "conditional generation" in BLIP captioning.
    text_inputs = {"text": prompt} if prompt else {}
    inputs = processor(images=image, return_tensors="pt", **text_inputs).to(device)

    # Hugging Face `generate` handles decoding for us
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        top_p=top_p,
    )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


# ------------------------------
# BLIP VQA (Visual Question Answering)
# ------------------------------
def load_vqa_model(device: str = "cpu"):
    """
    Load the BLIP VQA model & processor.
    Model card: Salesforce/blip-vqa-base
    """
    # ==========================================================
    # Teaching Note: Task-Specific Head
    # BLIP has multiple heads/variants:
    #  - Captioning: conditional generation of a sentence
    #  - VQA: answer a question (short text) given an image
    # They share the multimodal backbone but differ in heads/objectives.
    # ==========================================================
    model_name = "Salesforce/blip-vqa-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name,
                                                     use_safetensors=True).to(device).eval()
    return processor, model


@torch.no_grad()
def answer_question(
    image: Image.Image,
    question: str,
    processor: AutoProcessor,
    model: BlipForQuestionAnswering,
    device: str = "cpu",
    max_new_tokens: int = 10,
    num_beams: int = 3,
):
    """
    Ask a natural language question about the image (VQA).
    Example: "What animal is this?" -> "dog"
    """
    # Processor formats both image & question for the model
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
    )
    answer = processor.decode(output_ids[0], skip_special_tokens=True)
    return answer


# ------------------------------
# Pretty printing
# ------------------------------
def print_block(title: str, text: str):
    bar = "—" * 60
    print(f"\n{bar}\n{title}\n{bar}\n{text}\n")


# ------------------------------
# Main demo flow
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="BLIP Captioning & VQA Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="data/tiny/images/dog.jpg",
        help="Path or URL to an image (default: data/tiny/images/dog.jpg)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Optional text prompt for captioning, e.g. "a detailed photo of".',
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1 = greedy).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use nucleus sampling (stochastic decoding).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help='Ask a VQA question, e.g., "What animal is this?".',
    )
    args = parser.parse_args()

    # Load image (local or URL)
    try:
        image = load_image(args.image)
    except Exception as e:
        print(f"Failed to load image from {args.image}: {e}")
        sys.exit(1)

    device = get_device()
    print_block("Device", device)

    # ==========================
    # CAPTIONING
    # ==========================
    cap_processor, cap_model = load_caption_model(device=device)

    # 1) Greedy / Beam / Sampling controlled by CLI flags
    caption = caption_image(
        image=image,
        processor=cap_processor,
        model=cap_model,
        device=device,
        prompt=args.prompt,            # None or a steering phrase like "a photo of"
        max_new_tokens=20,
        num_beams=max(args.beams, 1),  # beams>=1
        do_sample=args.sample,         # True to sample
        top_p=0.9,
    )
    print_block("BLIP Caption", caption)

    # ==========================================================
    # Teaching Note: Prompting in BLIP
    # While BLIP isn't a chatty LLM, a short "prompt" like "a photo of"
    # can bias the generation style slightly (conditional captioning).
    # Try:
    #   --prompt "a photo of"
    #   --prompt "a detailed description of"
    #   --prompt "a caption describing"
    # ==========================================================

    # ==========================
    # VQA (optional if --question given)
    # ==========================
    if args.question:
        vqa_processor, vqa_model = load_vqa_model(device=device)
        answer = answer_question(
            image=image,
            question=args.question,
            processor=vqa_processor,
            model=vqa_model,
            device=device,
            max_new_tokens=10,
            num_beams=3,
        )
        print_block(f'VQA Answer (Q="{args.question}")', answer)

        # ==========================================================
        # Why VQA is powerful
        # Captioning provides a general description.
        # VQA lets users *query specific details* (counting, colors, objects).
        # Modern multimodal LLMs unify captioning/VQA/dialog in one interface.
        # ==========================================================

    # ==========================================================
    # Performance Tips
    # - GPU makes generation snappy; on CPU it's still fine for one-off demos.
    # - Lower max_new_tokens for speed; increase for more verbose captions.
    # - For live talks, pre-download models by running once beforehand.
    # ==========================================================


if __name__ == "__main__":
    main()
