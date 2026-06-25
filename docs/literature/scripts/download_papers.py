#!/usr/bin/env python3
"""Download open-access papers into docs/literature/papers/ from manifest.yaml."""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]  # docs/literature
REPO = Path(__file__).resolve().parents[3]  # repo root
PAPERS = ROOT / "papers"
MANIFEST = PAPERS / "manifest.yaml"
STATUS = PAPERS / "download-status.json"

USER_AGENT = "OpenAlterEgo-literature-mirror/1.0 (+local research archive)"


def _load_manifest() -> List[Dict[str, Any]]:
    text = MANIFEST.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return list(data.get("papers") or [])
    # Minimal YAML fallback for papers: list only
    papers: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    for line in text.splitlines():
        if line.strip().startswith("- id:"):
            if cur:
                papers.append(cur)
            cur = {"id": line.split(":", 1)[1].strip()}
        elif line.strip().startswith("id:") and not cur:
            cur = {"id": line.split(":", 1)[1].strip()}
        elif line.strip().startswith("section:"):
            cur["section"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("file:"):
            cur["file"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("title:"):
            cur["title"] = line.split(":", 1)[1].strip().strip('"')
        elif line.strip().startswith("year:"):
            cur["year"] = int(line.split(":", 1)[1].strip())
        elif line.strip() == "paywalled: true":
            cur["paywalled"] = True
        elif line.strip().startswith("official:"):
            cur["official"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("- https://") or line.strip().startswith("- http://"):
            cur.setdefault("urls", []).append(line.strip()[2:].strip())
    if cur:
        papers.append(cur)
    return papers


def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _download(url: str, timeout: float = 120.0) -> bytes:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except ssl.SSLError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()


def _write_stub(entry: Dict[str, Any], dest_dir: Path, reason: str) -> Path:
    stub = dest_dir / f"{entry['file']}.source.md"
    lines = [
        f"# {entry.get('title', entry['id'])}",
        "",
        f"**Bibliography ID:** `{entry['id']}`",
        f"**Year:** {entry.get('year', '—')}",
        "",
        f"**Status:** {reason}",
        "",
    ]
    if entry.get("official"):
        lines.append(f"**Official link:** {entry['official']}")
        lines.append("")
    if entry.get("urls"):
        lines.append("**Attempted mirrors:**")
        for u in entry["urls"]:
            lines.append(f"- {u}")
        lines.append("")
    lines.append(
        "This repository keeps a stub when the publisher PDF is paywalled or "
        "requires institutional access. Use the official link above."
    )
    stub.write_text("\n".join(lines), encoding="utf-8")
    return stub


def _process_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    section = str(entry.get("section") or "misc")
    dest_dir = PAPERS / section
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = str(entry["file"])
    pdf_path = dest_dir / f"{base}.pdf"

    row: Dict[str, Any] = {
        "id": entry["id"],
        "title": entry.get("title"),
        "section": section,
        "local_pdf": None,
        "local_stub": None,
        "local_extract": None,
        "status": "pending",
        "bytes": 0,
        "source_url": None,
    }

    extract = dest_dir / f"{base}.extract.md"
    if extract.is_file():
        row["local_extract"] = str(extract.relative_to(REPO)).replace("\\", "/")

    if entry.get("paywalled") and not entry.get("urls"):
        stub = _write_stub(entry, dest_dir, "Paywalled — no open PDF mirror configured")
        row["local_stub"] = str(stub.relative_to(REPO)).replace("\\", "/")
        row["status"] = "paywalled_stub"
        return row

    if pdf_path.is_file() and pdf_path.stat().st_size > 10_000:
        row["local_pdf"] = str(pdf_path.relative_to(REPO)).replace("\\", "/")
        row["status"] = "cached"
        row["bytes"] = pdf_path.stat().st_size
        return row

    urls = list(entry.get("urls") or [])
    last_err = ""
    for url in urls:
        try:
            print(f"[download] {entry['id']} <- {url}")
            data = _download(url)
            if not _is_pdf(data):
                # HTML landing page — not a PDF
                last_err = f"non-PDF response ({len(data)} bytes)"
                continue
            pdf_path.write_bytes(data)
            row["local_pdf"] = str(pdf_path.relative_to(REPO)).replace("\\", "/")
            row["status"] = "ok"
            row["bytes"] = len(data)
            row["source_url"] = url
            time.sleep(0.5)
            return row
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = str(e)
            continue

    if entry.get("paywalled"):
        stub = _write_stub(entry, dest_dir, f"Download failed; paywalled. Last error: {last_err}")
        row["local_stub"] = str(stub.relative_to(REPO)).replace("\\", "/")
        row["status"] = "paywalled_stub"
    else:
        stub = _write_stub(entry, dest_dir, f"Download failed: {last_err}")
        row["local_stub"] = str(stub.relative_to(REPO)).replace("\\", "/")
        row["status"] = "failed_stub"
    return row


def _write_readme(results: List[Dict[str, Any]]) -> None:
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_section.setdefault(str(r["section"]), []).append(r)

    lines = [
        "# Local paper library (PDFs)",
        "",
        "Offline copies of papers cited in OpenAlterEgo documentation.",
        "",
        "**Regenerate:** `python docs/literature/scripts/download_papers.py`",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
        "| `ok` / `cached` | PDF on disk |",
        "| `paywalled_stub` | Official link only (publisher paywall) |",
        "| `failed_stub` | Open URL failed; see stub for mirrors |",
        "",
        f"**Summary:** {sum(1 for r in results if r['status'] in ('ok', 'cached'))} PDFs, "
        f"{sum(1 for r in results if 'stub' in str(r['status']))} stubs, "
        f"{len(results)} total entries.",
        "",
    ]
    for section in sorted(by_section.keys()):
        lines.append(f"## `{section}/`")
        lines.append("")
        lines.append("| ID | Title | Local file | Status |")
        lines.append("|----|-------|------------|--------|")
        for r in sorted(by_section[section], key=lambda x: str(x["id"])):
            local = r.get("local_pdf") or r.get("local_stub") or "—"
            if local != "—":
                # README lives in papers/; strip leading docs/literature/papers/
                rel = str(r.get("local_pdf") or r.get("local_stub") or "")
                if rel.startswith("docs/literature/papers/"):
                    rel = rel[len("docs/literature/papers/") :]
                local = f"[{Path(rel).name}]({rel.replace(chr(92), '/')})"
            title = str(r.get("title") or "")[:70]
            lines.append(f"| `{r['id']}` | {title} | {local} | `{r['status']}` |")
        lines.append("")

    (PAPERS / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not MANIFEST.is_file():
        print(f"Missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    entries = _load_manifest()
    results = [_process_entry(e) for e in entries]
    STATUS.write_text(json.dumps({"papers": results}, indent=2), encoding="utf-8")
    _write_readme(results)
    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    stub = sum(1 for r in results if "stub" in str(r["status"]))
    print(f"[done] {ok} PDFs, {stub} stubs, status -> {STATUS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
