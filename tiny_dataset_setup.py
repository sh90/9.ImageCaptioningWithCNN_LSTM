# tiny_dataset_setup.py
import os
import io
import json
import argparse
import requests
from PIL import Image

# Tip: run with --force if you want to overwrite any existing image files.

def prepare_tiny_dataset():
    os.makedirs("data/tiny/images", exist_ok=True)

    # Updated captions (bike & pizza fixed to match the new images)
    captions = {
        "dog.jpg": [
            "a dog is running in the grass",
            "a brown dog runs on a field"
        ],
        "cat.jpg": [
            "a cat is sitting on a sofa",
            "a small cat sits on the couch"
        ],
        "bike.jpg": [
            "a blue bicycle with shopping bags is parked by a stone wall",
            "a parked bicycle loaded with groceries by the sidewalk"
        ],
        "pizza.jpg": [
            "a pepperoni pizza with olives and mushrooms on a wooden table",
            "a slice of cheesy pepperoni pizza being served"
        ],
        "beach.jpg": [
            "people are on a sandy beach",
            "a sunny day at the beach"
        ],
        "car.jpg": [
            "a red car is parked on the road",
            "a car stands on the street"
        ],
    }

    urls = {
        "dog.jpg":   "https://images.unsplash.com/photo-1517849845537-4d257902454a?q=80&w=1200",
        "cat.jpg":   "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?q=80&w=1200",
        "bike.jpg":  "https://upload.wikimedia.org/wikipedia/commons/4/41/Packed_bicycle.jpg",
        "pizza.jpg": "https://upload.wikimedia.org/wikipedia/commons/8/86/Pizza_%281%29.jpg",
        "beach.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1200",
        "car.jpg":   "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1200",
    }

    headers = {"User-Agent": "image-captioning-demo/1.0 (+https://example.com)"}

    for fname, url in urls.items():
        path = f"data/tiny/images/{fname}"
        if os.path.exists(path) and not force:
            print(f"Skip existing {path} (use --force to overwrite)")
            continue
        try:
            print(f"Downloading {fname} …")
            resp = requests.get(url, timeout=30, headers=headers)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(path, format="JPEG", quality=90)
            print("Saved", path)
        except Exception as e:
            print(f"Failed to download {fname} from {url}: {e}")

    with open("data/tiny/captions.json", "w") as f:
        json.dump(captions, f, indent=2)
    print("Tiny dataset ready with", len(captions), "images at data/tiny/images")

if __name__ == "__main__":
    prepare_tiny_dataset()
