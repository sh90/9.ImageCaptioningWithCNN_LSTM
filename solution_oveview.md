1) Data & Labels
“We start with a dataset of images and captions. Each image has 1–5 human-written sentences that describe it. These sentences are our labels.”

2) Preprocessing
“Each image is resized (e.g., 224×224) and normalized. Each caption is tokenized into words, we build a vocabulary, and we add special tokens like <bos> and <eos> so the model knows where to start and stop.”

3) Task
“Given an image, generate a sentence, one word at a time, that reads naturally and matches the scene.”

1) Classic: CNN ➜ LSTM (Encode once, decode sequentially)


“Think of a photographer and a storyteller. The CNN is our photographer—it looks at the whole image and compresses it into a single feature vector (an ‘image embedding’). Then the LSTM is our storyteller—it takes that vector as context and speaks the caption word by word.”

Step-by-step flow

Image in ➜ CNN

Input: image (224×224).

CNN (ResNet50) extracts features → a 2048-dim vector (we project to 256-dim).

This is the image embedding: a compact summary of the picture.

Image embedding ➜ LSTM

We use the image embedding to initialize the LSTM’s hidden state (its memory).

At time step 1, the LSTM sees <bos> and predicts the first word.

At time step 2, it takes the previous word and predicts the next, and so on until <eos>.

Training vs. Inference

Training: use teacher forcing (feed the true previous word) + cross-entropy loss on the next word.

Inference: feed back the model’s own prediction. Use greedy or beam search.

Output

A simple caption like “dog running on grass”.

What to show
Run captioning_cnn_lstm.py and narrate the printed predictions. Call out CNNEncoder, LSTMDecoder, and the “teacher forcing” logic.

2) With Attention: “Show, Attend and Tell”

“Now our photographer doesn’t just hand over one summary photo. It gives a grid of mini-photos (spatial features). As the storyteller speaks each word, it looks (attends) to the most relevant mini-photo. That’s attention: look while you speak.”

Step-by-step flow

Image in ➜ CNN (spatial)

Keep the 7×7 feature map (49 locations), not just a single vector.

Each location = a small patch representation.

Attention at each word

Before predicting a word, the decoder computes attention weights (α) over the 49 locations based on its current hidden state.

It builds a context vector = weighted sum of those locations.

Input to the LSTM cell at each step = [word embedding ; context].

Training & Inference

Same loss/decoding as before, but with attention making the focus dynamic.

Output

Captions often become more specific (e.g., “a brown dog running across a grassy field”), and you can visualize attention heatmaps per word.

What to show
Run captioning_with_attention.py, then attention_heatmap_demo.py. Show the heatmap image and say:
“Each small panel is a generated word. The colored overlay shows where the model was looking to say that word.”

3) Modern: Transformers & Foundation Models (BLIP)

“Instead of training a small model on a small dataset, modern models like BLIP are pretrained on millions of image–text pairs. They use Transformers for both vision and language with attention everywhere. That’s why they can caption images zero-shot—no extra training needed.”

Step-by-step flow

Image in ➜ Vision Transformer (ViT)

The image is split into patches; each patch becomes an embedding.

Self-attention lets the model relate patches to each other (shape, context).

Cross-attention to Text Transformer

A Transformer decoder generates words while attending to image patches and prior text, all in parallelizable attention blocks.

Decoding via greedy/beam/sampling (like before, but inside a Transformer LM).

Pretraining ➜ Zero-shot

Because BLIP learned from huge web-scale data, it already knows a lot about objects, scenes, and language style, so it can caption your image out-of-the-box.

Output

Typically fluent, detailed captions without task-specific training.

Can also do VQA: answer questions about the image.

What to show
Run blip_transformer_demo.py once with greedy and once with --beams 3. Optionally add --prompt "a detailed photo of" to show style steering.

Tiny “Movie” You Can Tell in One Breath

“We collect images with captions, resize the images, tokenize the captions, and add <bos>/<eos>. In the classic approach, a CNN turns the whole image into a single embedding, which an LSTM uses to speak the caption word by word, trained with teacher forcing. With attention, we keep a grid of features so the decoder can look at different regions for each word—like showing where its eyes go as it talks. In modern systems like BLIP, a Vision Transformer encodes the image patches and a Transformer language model generates the text, both trained on millions of image–text pairs, so it can caption zero-shot, often with richer detail.”

Quick Slide-Ready Summary (3 bullets each)

## CNN+LSTM

Single global image vector ➜ sequential word generation

Needs labeled data; simpler but limited detail

Training uses teacher forcing, loss = cross-entropy

## Attention

Spatial features (e.g., 7×7 grid) ➜ focus on regions per word

Clearer, more specific captions; heatmaps for explainability

Same training, better alignment between words and image parts

## BLIP / Transformers

ViT + Transformer LM, attention everywhere

Pretrained on massive data ➜ zero-shot captioning & VQA

Typically fluent, context-aware, scalable
