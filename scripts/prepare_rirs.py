"""
@author aud2studio maintainers <maintainers@example.com>
@description Index existing RIR .wav files under a directory into a JSONL catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rir_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    rir_dir = Path(args.rir_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for wav in sorted(rir_dir.rglob("*.wav")):
            f.write(json.dumps({"path": str(wav)}) + "\n")
    print(f"Indexed RIRs to {out}")


if __name__ == "__main__":
    main()
