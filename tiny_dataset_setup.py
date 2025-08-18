# tiny_dataset_setup.py
import os
import io
import json
import requests
from PIL import Image

def prepare_tiny_dataset():
    os.makedirs("data/tiny/images", exist_ok=True)

    captions = {
        "dog.jpg":   ["a dog is running in the grass", "a brown dog runs on a field"],
        "cat.jpg":   ["a cat is sitting on a sofa", "a small cat sits on the couch"],
        "bike.jpg":  ["a person rides a bicycle on the street", "a cyclist is riding a bike"],
        "pizza.jpg": ["a pizza is on a plate", "a tasty pizza sits on a table"],
        "beach.jpg": ["people are on a sandy beach", "a sunny day at the beach"],
        "car.jpg":   ["a red car is parked on the road", "a car stands on the street"],
    }

    urls = {
        "dog.jpg":   "https://images.unsplash.com/photo-1517849845537-4d257902454a?q=80&w=1200",
        "cat.jpg":   "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?q=80&w=1200",
        "bike.jpg":  "https://images.unsplash.com/photo-1518655048521-f130df041f66?q=80&w=1200",
        "pizza.jpg": "https://images.unsplash.com/photo-1542281286-9e0a16bb7366?q=80&w=1200",
        "beach.jpg": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1200",
        "car.jpg":   "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=1200",
    }

    for fname, url in urls.items():
        path = f"data/tiny/images/{fname}"
        if not os.path.exists(path):
            try:
                print(f"Downloading {fname}...")
                img = Image.open(io.BytesIO(requests.get(url, timeout=20).content)).convert("RGB")
                img.save(path, format="JPEG", quality=90)
                print("Saved", path)
            except Exception as e:
                print(f"Failed to download {fname} from {url}: {e}")

    with open("data/tiny/captions.json", "w") as f:
        json.dump(captions, f, indent=2)

    print("Tiny dataset ready with", len(captions), "images at data/tiny/images")

if __name__ == "__main__":
    prepare_tiny_dataset()
