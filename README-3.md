# 🖼️ Image Recognition — Streamlit App

Part of the **AI Playground: 4 Real-World AI Projects** series.

Upload a photo and this app identifies the object in it using
**MobileNetV2**, a neural network pretrained by Google on the
ImageNet dataset (1.2M images, 1,000 categories) — no training required
(this is called *transfer learning*).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Note: the first run downloads ~14 MB of pretrained weights.

## Files

- `app.py` — Streamlit UI (upload widget, prediction display)
- `model.py` — loads MobileNetV2 and runs predictions
- `requirements.txt` — dependencies (uses `tensorflow-cpu` to keep the
  deploy lighter — this app doesn't need a GPU)

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (its own repo, same as Project 1 —
   e.g. `image-recognition`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo/branch, set **Main file path** to `app.py`.
4. Deploy. First build is slower than Project 1 since TensorFlow is a
   larger dependency — expect a few minutes.

## Note on free-tier hosting

TensorFlow + MobileNetV2 uses noticeably more memory than Project 1.
If Streamlit Cloud's free tier throws a memory error on deploy, the
usual fixes are: confirm `tensorflow-cpu` (not full `tensorflow`) is in
requirements.txt, or reduce concurrent usage.
