# Data collection protocol

Silent speech is *personalized*. If you want anything approaching “works reliably,”
you need per-user data and per-user calibration.

This doc gives a pragmatic protocol based on what the papers describe,
but simplified so you can run it alone.

---

## Recommended first dataset: closed vocabulary commands

Start with something like:
- digits 0–9
- yes / no
- left / right / select / cancel
- a wake word (e.g., “computer”)

### Repetitions
Aim for:
- ~50–200 examples per class (more if you can tolerate it)
- multiple sessions if possible (session shift is real)

---

## Prompting

Use random prompts to avoid patterns.
Display on screen:
- “Get ready” (1s)
- “Silently say: <token>” (2s)
- “Rest” (1–2s)

---

## Labeling / segmentation

### Option A: Marker button (simple)
- Record continuously.
- When you start a token, press a button (or keyboard key).
- Press again at the end (or record fixed duration).

### Option B: Dedicated marker channel (fancy)
If your hardware can:
- reserve a marker channel, or
- inject a digital marker packet into your stream

…do it. Life gets easier.

---

## File format (suggested)

We store each session as:
- `session.json` (metadata: sample rate, channel names, electrode placement notes)
- `signals.npy` (float32 array: [time, channels])
- `events.csv` (start_sample, end_sample, label)

You can upgrade this later to HDF5 or BIDS-like structures.

---

## Basic quality checks (in the moment)

- Channel amplitude not saturating
- Obvious 60 Hz hum not dominating
- Electrode contact stable over 30–60 seconds
- If you add an IMU: note head motion artifacts

