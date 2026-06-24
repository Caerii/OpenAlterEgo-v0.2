# Mechanical design and wearable form factors

Repeatable electrode positioning, cable management, and form-factor trade-offs from literature.

---

## Core principle

> **Repeatability beats cleverness.** ([docs/01-hardware.md](../docs/01-hardware.md))

Silent speech accuracy degrades with electrode displacement ([Gowda 2024](https://arxiv.org/abs/2411.02591) domain shift). Mechanical design should **constrain degrees of freedom** while remaining adjustable for head size and comfort.

---

## Form-factor comparison

| Form factor | Example | Comfort | Discretion | Repeatability | Channels | Ref |
|-------------|---------|---------|------------|---------------|----------|-----|
| **Head band + rigid arms** | AlterEgo 2018 | Moderate | Low | **High** (if arms fixed) | 7 | [IUI 2018](https://dl.acm.org/doi/10.1145/3172944.3172977) |
| **Adhesive patches only** | V0 benchtop | High | Moderate | **Low** (manual) | 8 | Kapur 2020 |
| **Textile neckband** | SilentWear | High | **High** | Moderate (velcro sizing) | 10–14 | [arXiv:2603.02847](https://arxiv.org/abs/2603.02847) |
| **Headphone earmuff** | Tang 2025 | High | **High** | High (fixed shell) | 4 | [arXiv:2504.13921](https://arxiv.org/abs/2504.13921) |
| **Tattoo / epidermal** | Wang 2021 | Very high | Moderate | High (skin-bound) | 4 | [npj Flex](https://doi.org/10.1038/s41528-021-00119-7) |

### OpenAlterEgo recommended path

1. **V0:** Adhesive gel — learn muscle landmarks
2. **V1:** Head band prototype (AlterEgo mechanical lineage from patent + 2018 paper)
3. **V2a:** Evaluate neckband if discretion required (SilentWear pattern)
4. **V2b:** Headphone integration if 4-ch vocabulary sufficient (Tang pattern)

---

## AlterEgo 2018 mechanical notes (from public sources)

From [00-background.md](../docs/00-background.md) / Kapur 2018:

- Band around **back of head**
- **Photopolymer resin** frame + **brass rod** structure
- **Modular brass electrode supports** — adjustable reach to skin
- Gold-plated silver electrodes with Ten20 paste **or** passive dry Ag/AgCl

**Engineering translation:**
- 3D-printed PETG/ABS frame acceptable for V2 prototype
- Electrode arms: brass rod or carbon fiber tube (stiff, light)
- Sliding clamp per arm for length adjustment; lock screw at target depth
- Target **consistent gentle pressure** (~enough to stabilize dry contact, not blanch skin)

---

## SilentWear neckband pattern

From [arXiv:2603.02847](https://arxiv.org/abs/2603.02847):

- Soft fabric band, **velcro** sizing
- **27 snap fasteners** for dry electrodes (Datwyler SoftPulse)
- **15 electrodes** in 3 rows → overlapping differential → 10 effective channels
- **4 shorted electrodes** at back of neck for ground
- Electronics: **26 × 65 × 13 mm** BioGAP-Ultra module at rear

**Takeaways for OpenAlterEgo:**
- Neck-dominant montage reduces facial stigma
- Overlapping diff increases channel count without more skin area
- Rear-mounted PCB balances weight

---

## Tang 2025 headphone pattern

- **4 textile electrodes** in earmuff foam
- **Towel-like micro-fiber protrusions** for skin coupling ([arXiv:2504.13921](https://arxiv.org/abs/2504.13921))
- ESP32-S3 wireless module
- Trade-off: only 4 channels — software must use SE attention weighting

---

## Cable and motion management

Motion reduces SNR by **~33%** (18.9 → 12.7 dB) in Tang 2025 — much is **cable sway** and **electrode slip**.

| Technique | Implementation |
|-----------|----------------|
| **Route along frame** | Zip-tie leads to band arms, not free-hanging |
| **Service loop at ear** | Allow jaw open without pulling earlobe ref |
| **Low-profile connector** | JST at band → short pigtails to electrodes |
| **Strain relief** | Knot + glue at PCB entry; prevents yank on skin |
| **Lightweight leads** | Shielded micro-coax or ribbon; avoid heavy RG174 |

**IMU (optional):** mount on band center; log head motion alongside EMG for artifact regression (future `meta` field).

---

## Adjustability vs repeatability

| DOF | AlterEgo approach | Recommendation |
|-----|-------------------|----------------|
| Head circumference | Elastic band segment | Velcro + measured marks |
| Arm length | Sliding brass supports | Scale marks per user profile |
| Arm angle | Fixed detents at 15° steps | 3D-printed detent washer |
| Electrode pressure | Spring at tip | Silicone bumper + constant-force spring |

**Per-user mechanical profile:** store arm length marks in `UserProfile` metadata (future field) — "alice: CH3 arm = 42 mm".

---

## Materials

| Part | V0 | V2 |
|------|----|----|
| Frame | — | PETG/ABS print or brass rod |
| Skin contact | Hydrogel Ag/AgCl | Dry textile / conductive silicone |
| Adhesion | Medical tape / paste | Band tension only |
| Enclosure | Dev board box | SLA resin or injection pocket |

**Wang 2021:** tattoo-like interfaces need specialized materials (Cu nanowire mesh, silicone) — defer to research partnerships.

---

## Mechanical exit criteria (V2)

- [ ] Re-seat time < 2 min with written landmark guide
- [ ] Inter-session position variance < 5 mm on 3 landmark points (caliper / photo)
- [ ] No electrode pull during normal jaw open/close (video verified)
- [ ] 4 h wear without skin damage (gel) or pressure ulcer risk (dry)
- [ ] Device mass < 150 g on head (comfort threshold, informal)

---

## 3D / CAD workflow (suggested)

1. Landmark placement on manikin head → arm endpoint coordinates
2. CAD band + arms in Fusion360 / FreeCAD
3. Print V1 frame → iterate arm length
4. Document final STL in `hardware/cad/` (future) with BOM references

---

## Related

- Electrode map: [03-electrodes-montage.md](03-electrodes-montage.md)
- Tier milestones: [01-architecture-tiers.md](01-architecture-tiers.md)
- BOM mechanical parts: [BOM.md](BOM.md)
