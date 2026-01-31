# Protocol and XR integration

OpenAlterEgo’s realtime output is a websocket server that emits JSON messages.

Implementation: `software/python/openalterego/api/server.py`

---

## Token message

Server → client:

```json
{
  "type": "token",
  "token": "yes",
  "confidence": 0.92,
  "t": 1730000000.123,
  "seq": 12345,
  "source": "sim",
  "meta": {}
}
```

- `token`: decoded label (closed vocabulary)
- `confidence`: softmax probability for that label
- `t`: best-effort timestamp (seconds)
- `seq`: best-effort sample index
- `source`: where it came from (sim / virtual_ble / ble / demo)

---

## Control messages

Client → server:

```json
{"type":"control","cmd":"ping"}
```

Server replies with:

```json
{"type":"status","status":"pong","t":1730000000.123,"meta":{}}
```

---

## Unity

See `software/unity/OpenAlterEgoReceiver.cs`.

It connects to the websocket and parses token messages.

Recommended next steps:
- add a “HUD text” overlay that shows the most recent token
- drive menu selection (“left/right/select/cancel”)
- test under packet loss using `--source virtual_ble`

---

## Smart glasses stub

Terminal client:

```bash
openalterego glasses --url ws://127.0.0.1:8765
```
