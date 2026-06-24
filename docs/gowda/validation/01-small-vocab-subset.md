# Validation: Small-Vocab Subset (`gowda_sv`)

> **Historical note:** Imported before June 2026 bugfix (misaligned `events.csv`, equal quarter splits). For current work use [02-top30-corrected.md](02-top30-corrected.md).

---

## Session

| Field | Value |
|-------|-------|
| Path | `sessions/gowda_sv/` |
| Trials | 200 (subset) |
| Labels | 12 words |
| Events | 300 |
| fs | 5000 Hz, 31 ch |

## Best result (pre-fix)

| Config | Val accuracy |
|--------|--------------|
| SE-ResNet, wide, 600 ms, event split | **16.4%** (12-way, chance ~8.3%) |

## Reproduce (legacy)

```bash
openalterego dataset import-gowda --download --out ./sessions/gowda_sv \
  --max-segments 200 --top-labels 12 --min-samples-per-label 5
openalterego train --data ./sessions/gowda_sv --fs 5000 --emg-mode wide --arch se_resnet
```

Full narrative was in [`../../16-gowda-validation-results.md`](../../16-gowda-validation-results.md).
