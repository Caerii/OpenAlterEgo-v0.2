"""Smart-glasses display stub (terminal).

This is a placeholder for a real glasses/AR display integration.
For now it just connects to the websocket server and renders the latest token.

Usage:
    python -m openalterego.clients.glasses_stub --url ws://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets


def _render_line(s: str) -> None:
    # single-line overwrite (nice for HUD-ish feel)
    sys.stdout.write("\r" + s + " " * max(0, 60 - len(s)))
    sys.stdout.flush()


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default="ws://127.0.0.1:8765")
    ap.add_argument("--multiline", action="store_true", help="Print each token on a new line")
    args = ap.parse_args()

    async with websockets.connect(args.url) as ws:
        if args.multiline:
            print(f"[OpenAlterEgo glasses stub] connected: {args.url}")
        else:
            _render_line(f"[connected] {args.url}")

        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") != "token":
                continue
            tok = str(msg.get("token", ""))
            conf = float(msg.get("confidence", 0.0))
            t = float(msg.get("t", time.time()))
            line = f"🕶️  {tok}  ({conf:.2f})  t={t:.2f}"
            if args.multiline:
                print(line)
            else:
                _render_line(line)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
