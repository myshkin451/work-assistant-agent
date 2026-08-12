#!/usr/bin/env python3
"""Fail on obvious secret/company material in Git-tracked public files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "api_key_shape": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Assemble private product markers so the scanner does not flag its own rules.
    "private_product_term": re.compile(
        "|".join(("企" + "跑线", "天" + "高", "qipao" + "xian")),
        re.IGNORECASE,
    ),
    "internal_ipv4": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
}


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for relative in result.stdout.splitlines():
        path = ROOT / relative
        if path.is_file():
            files.append(path)
    return files


def main() -> int:
    findings: list[str] = []
    for path in candidate_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}:{label}")
    if findings:
        print("public boundary scan failed:")
        for finding in sorted(findings):
            print(f"- {finding}")
        return 1
    print(f"public boundary scan passed ({len(candidate_files())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
