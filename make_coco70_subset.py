# make_coco70_subset.py
import os, io, json, random, requests
from PIL import Image
from datasets import load_dataset

# Where your training scripts expect data
OUT_IMG_DIR = "data/tiny_more_v1/images"
OUT_JSON = "data/tiny_more_v1/captions.json"
N = 70   # change to 60–70 as you like
SEED = 42

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# Use certifi so macOS/Python trust is consistent (avoids SSL issues)
try:
    import certifi, os as _os
    _os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

print("Loading COCO 2017 (Hugging Face)…")
# This dataset includes: file_name, coco_url, captions (list of 5 strings)
ds = load_dataset("phiyodr/coco2017", split="validation")  # 5k images
ds = ds.shuffle(seed=SEED).select(range(N))

caps = {}
kept = 0
for ex in ds:
    url = ex["coco_url"]              # e.g., http://images.cocodataset.org/val2017/…
    file_name = os.path.basename(ex["file_name"])  # e.g., COCO_val2017_000000xxxxxx.jpg
    out_path = os.path.join(OUT_IMG_DIR, file_name)

    if os.path.exists(out_path):
        print("Exists:", out_path)
        kept += 1
        caps[file_name] = ex["captions"]
        continue

    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        # keep reasonable size; your pipeline will resize to 224 anyway
        img.save(out_path, "JPEG", quality=90)
        caps[file_name] = ex["captions"]  # list of 5 human captions
        kept += 1
        print(f"Saved {file_name}")
    except Exception as e:
        print("  Skip (download error):", url, "->", e)

# Merge with existing captions.json if present
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        existing = json.load(f)
else:
    existing = {}

existing.update(caps)

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(existing, f, indent=2)

print(f"\nWrote {kept} images to {OUT_IMG_DIR}")
print(f"Updated captions file: {OUT_JSON}")
print("Done.")
