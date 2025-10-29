"""
@author aud2studio maintainers <maintainers@example.com>
@description Build a JSONL manifest from a single folder of pairs.

Naming rule:
- For each group, files share the same prefix before the last underscore `_`.
- The file with suffix `_0` is the ground-truth reference.
- All other files in the same group are audience inputs.

Examples:
  input0_0.wav (reference)
  input0_1.wav, input0_blah.wav (audience inputs)
  input1_0.mp3 (reference), input1_1.mp3 (audience input)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def split_group_key(stem: str) -> Tuple[str, str]:
    """Return (group_prefix, suffix) by splitting at the last underscore.
    If no underscore is present, the whole stem is the group and suffix is ''."""
    if "_" not in stem:
        return stem, ""
    i = stem.rfind("_")
    return stem[:i], stem[i + 1 :]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder with .wav/.mp3 files")
    p.add_argument("--out", type=str, required=True, help="Output JSONL manifest path")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, Dict[str, List[str] | str]] = {}
    for pth in sorted(list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))):
        stem = pth.stem
        group, suffix = split_group_key(stem)
        record = groups.setdefault(group, {"aud": []})
        if suffix == "0":
            record["ref"] = str(pth.resolve())
        else:
            record["aud"].append(str(pth.resolve()))

    num_written = 0
    with out.open("w") as f:
        for g, rec in sorted(groups.items()):
            ref = rec.get("ref")  # type: ignore[assignment]
            aud = rec.get("aud", [])  # type: ignore[assignment]
            if not ref or not aud:
                continue
            f.write(json.dumps({"aud": aud, "ref": ref}) + "\n")
            num_written += 1

    print(f"Wrote {num_written} pairs to {out}")


if __name__ == "__main__":
    main()


