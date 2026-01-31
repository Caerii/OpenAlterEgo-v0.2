# Simulation

The fastest way to iterate on a silent-speech pipeline is to **simulate everything first**:

- synthetic multichannel EMG-ish signals
- “hardware-ish” ADC quantization
- packet framing (BLE notification payloads)
- link jitter + packet loss
- realtime decoding → websocket output

This repo includes all of that so you can stress-test your stack before you ever touch electrodes.

---

## 1) Synthetic signal stream

Implementation: `software/python/openalterego/sim/stream.py`

The simulator generates:

- baseline multi-channel noise
- low-frequency drift (motion-ish)
- token events: short activation bursts with label-specific spatial patterns

It’s *not* a biophysical muscle model. It’s a controllable generator that’s good enough to:
- validate your buffering and timing
- debug preprocessing
- build XR integrations
- train a toy model end-to-end

---

## 2) Dataset generation

```bash
openalterego sim-dataset --out ./sim_session --minutes 2
```

Output folder:

- `signals.npy` — `(time, channels)` float32
- `events.csv` — segments with `start_sample,end_sample,label`
- `meta.json` — session metadata

---

## 3) Virtual BLE (byte-level transport)

Implementation:
- `openalterego.acquisition.packet` (OpenAlterEgo v1 framing)
- `openalterego.acquisition.virtual` (virtual notifications + parsing)
- `openalterego.sim.transport` (loss/jitter)

Run with:

```bash
openalterego serve --source virtual_ble --model ./sim_session/model.pt --loss 0.05 --jitter-ms 10
```

This is the closest thing to “hardware in the loop” you can do without hardware.

---

## 4) Why packet framing matters

Raw int16 samples *work* for quick hacks. They also break the second you introduce:

- different MTUs
- packet loss
- versioned firmware changes

The OpenAlterEgo v1 packet format includes:
- magic + version
- channel + frame counts
- sample index (so you can detect gaps)

See `openalterego.acquisition.packet`.
