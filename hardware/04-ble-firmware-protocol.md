# BLE, firmware, and host protocol

MCU selection, BLE transport, OpenAlterEgo v1 packet format, and host integration.

---

## Transport choice: BLE

| Property | BLE | WiFi | USB |
|----------|-----|------|-----|
| Power | Low | High | N/A (tethered) |
| Wearable fit | Excellent | Poor | Debug only |
| Throughput for 8ch@250Hz | Sufficient | Overkill | Lab |
| AlterEgo lineage | **Yes** ([Kapur 2018](https://dl.acm.org/doi/10.1145/3172944.3172977)) | No | Dev kits |

**Bandwidth budget (8 ch, 250 Hz, int16):**

```
250 samples/s × 8 ch × 2 bytes = 4000 bytes/s payload
+ OA v1 header (~20 B) per packet
```

With 12 frames/packet (default `PacketSpec`): ~833 packets/s × ~212 B ≈ **176 kB/s** — well within BLE 5.0 throughput with connection interval 7.5–15 ms.

At **1000 Hz**, scale ×4 → still feasible with larger MTU (247 B) and efficient packing; monitor connection interval and processor load ([Tang 2025](https://arxiv.org/abs/2504.13921) uses 1 kHz).

---

## MCU + BLE stack

| Platform | Used in | Pros | Cons |
|----------|---------|------|------|
| **nRF52840** + SoftDevice / Zephyr | Wearable reference | Low power, mature BLE | Nordic toolchain |
| **nRF5340** | BioGAP-Ultra, SilentWear | Dual-core, headroom | Complexity |
| **ESP32-S3** | Tang 2025 headphone SSI | Fast dev, WiFi option | Higher idle power |

**OpenAlterEgo V1 recommendation:** nRF52840 + Zephyr BLE peripheral (or Adafruit/nRF Connect SDK).

### Firmware responsibilities

1. Initialize ADS1299 (SPI): rate, gain, BIAS, lead-off optional
2. DRDY interrupt → read sample frame
3. Pack `oa_v1` with monotonic `seq` and `sample_index`
4. BLE GATT Notify on data characteristic
5. Optional: battery SOC, lead-off flags in packet `flags` field (reserved in v1)

---

## OpenAlterEgo v1 packet format

**Canonical spec:** `software/python/openalterego/acquisition/packet.py`

### Header (20 bytes, little-endian)

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | magic | 2×char | `b"OA"` |
| 2 | version | u8 | `1` |
| 3 | channels | u8 | e.g. `8` |
| 4 | frames | u16 | samples in this packet |
| 6 | flags | u16 | reserved (lead-off, IMU sync, …) |
| 8 | seq0 | u32 | packet sequence number |
| 12 | sample_index0 | u64 | index of first sample |

### Payload

`frames × channels` int16 values, **interleaved by channel** (C-order: time outer, channel inner).

### Scaling on host

```python
@dataclass(frozen=True)
class AfeSpec:
    adc_bits: int = 16      # on-wire bits
    vref_v: float = 2.4     # match firmware
    gain: float = 24.0      # PGA setting
```

µV per count = `(Vref / gain) / (2^(bits-1) - 1) × 10^6`

**Critical:** Firmware `gain` and `vref` must match `AfeSpec` passed to `parse_oa_v1()`.

---

## BLE GATT profile (template)

Custom service — UUIDs are **project-specific** (generate for your firmware):

| Characteristic | Properties | Content |
|----------------|------------|---------|
| `EMG_DATA` | Notify | OA v1 packets |
| `DEVICE_INFO` | Read | fw version, channel count, fs |
| `CONFIG` | Write | optional: fs, gain (future) |

Host: `BleSpec` in `acquisition/ble_client.py`:

```python
BleSpec(
    device_name="OpenAlterEgo",
    data_char_uuid="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    fs_hz=250,
    channels=8,
    packet_format="oa_v1",
    afe=AfeSpec(adc_bits=16, vref_v=2.4, gain=24.0),
)
```

---

## Loss detection

`ble_client.py` tracks `sample_index` continuity:

- Gap → increment `lost_samples_total`, log warning
- Host pipeline can flag `meta["samples_lost"]` for quality monitor

**Virtual test without radio:**

```bash
openalterego serve --source virtual_ble
```

Simulates loss/jitter via `acquisition/virtual.py` — use before field BLE debug.

---

## Firmware development workflow

```
1. Unit test pack/parse round-trip (Python tests/test_packet.py)
2. Firmware pack identical bytes → logic analyzer / sniffer
3. virtual_ble with recorded byte stream
4. collect ble → session folder → calibrate → train
5. serve --source ble --user-id <id>
```

### Minimum viable firmware features

- [ ] ADS1299 @ 250 Hz, gain 24, 8 ch differential
- [ ] OA v1 notify @ ≥ 20 Hz packet rate
- [ ] Stable `sample_index` (no gaps under 1 m LOS)
- [ ] Battery voltage in device info (optional)

### Stretch goals

- [ ] Runtime fs/gain change via CONFIG characteristic
- [ ] Lead-off bitmask in `flags`
- [ ] OTA firmware update (nRF DFU)

---

## Patent / protocol notes

US10878818B2 (AlterEgo patent) describes:
- Wireless relay from wearable to mobile
- Framed biosignal packets for mobile processing
- Safety-isolated acquisition path

OA v1 is our open, microcontroller-friendly framing — not identical to commercial AlterEgo packets but functionally equivalent for the host stack.

---

## Related

- Analog configuration: [02-analog-front-end.md](02-analog-front-end.md)
- Power / RF layout: [05-power-safety.md](05-power-safety.md)
- Host WebSocket output: `docs/06-protocol-xr.md`
