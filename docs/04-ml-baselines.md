# ML baselines

## Start with per-user closed vocabulary

Do not try to solve “general silent speech” first. You will hate your life.

### Baseline A: 1D CNN on raw-ish windows
- Input: [channels, time]
- 3–7 conv blocks (Conv1d → ReLU → MaxPool)
- Global max pool
- FC → softmax

### Baseline B: MFCC-like features + 1D CNN
Closer to the 2018 approach:
- Convert each channel into MFCC-like coefficients per frame
- Concatenate channels (or treat channels as feature maps)
- 1D CNN → softmax

---

## Training hygiene

- Stratified splits (train/val/test)
- Repeated CV if your dataset is small
- Track per-class confusion matrix
- Watch for session leakage (don’t mix segments from the same recording burst across splits)

---

## Metrics

- Accuracy (closed vocabulary)
- Top-k accuracy (useful if you want language-model rescoring later)
- Information Transfer Rate (bits/min) if you want to compare to BCI style reports

---

## Personalization tricks that actually work

- Train a global model, then fine-tune last layer per user.
- Or: metric learning + nearest-prototype classifier.
- Or: domain-adversarial training if you have many users.



---

## Realtime alignment tips

If you plan to run streaming inference (sliding window):

- Train on **fixed-length segments** (use `--segment-ms`).
- Use streaming-compatible preprocessing (use `--preprocess-mode streaming`), so your training data looks like what
  your realtime pipeline will see (causal filters + running normalization).

Example:

```bash
openalterego train --data ./session --fs 250 --preprocess-mode streaming --segment-ms 600
```
