"""Fetch every basis set published by the Basis Set Exchange and write each as
JSON under data/basis/. Also emits data/basis/_index.json mapping canonical
basis names to their sanitized filenames so the Zig loader can do
canonical-name lookups without reimplementing the sanitization rule."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import basis_set_exchange as bse


REPO_DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DATA_DIR / "basis"


_FILENAME_SUBS = [
    ("*", "_st_"),
    ("/", "_sl_"),
    ("(", "_"),
    (")", "_"),
    (",", "_"),
    ("+", "_p_"),
    (" ", "_"),
]


def sanitize(name: str) -> str:
    out = name.lower()
    for src, dst in _FILENAME_SUBS:
        out = out.replace(src, dst)
    out = re.sub(r"_+", "_", out).strip("_")
    return out + ".json"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = bse.get_metadata()
    names = sorted(metadata.keys())
    index: dict[str, str] = {}

    total_bytes = 0
    failures: list[tuple[str, str]] = []
    start = time.perf_counter()

    for i, name in enumerate(names, start=1):
        filename = sanitize(name)
        try:
            payload = bse.get_basis(
                name,
                fmt="json",
                header=False,
                optimize_general=False,
            )
        except Exception as exc:
            failures.append((name, repr(exc)))
            continue

        parsed = json.loads(payload)
        text = json.dumps(parsed, indent=2, sort_keys=True) + "\n"
        path = OUTPUT_DIR / filename
        path.write_text(text, encoding="utf-8")

        index[name] = filename
        total_bytes += len(text.encode("utf-8"))

        if i % 50 == 0 or i == len(names):
            print(f"  [{i:4d}/{len(names)}] {name} -> {filename}", flush=True)

    (OUTPUT_DIR / "_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start
    written = len(index)
    print(
        f"Wrote {written} basis sets "
        f"({total_bytes / (1024 * 1024):.2f} MiB) "
        f"in {elapsed:.1f}s to {OUTPUT_DIR}"
    )
    if failures:
        print(f"WARNING: {len(failures)} basis set(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
