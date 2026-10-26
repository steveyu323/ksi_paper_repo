#!/usr/bin/env python3
"""Build a manifest of the raw/result files a section depends on.

Walks the symlink trees under data/raw and data/results, resolves each link to
its real path, and records size and mtime. Checksums are opt-in: the raw trees
total well over 100 GB on a network mount, so hashing them is a deliberate
step taken once, just before upload, not something to do on every run.

    ./make_manifest.py <package_root> [--checksum] [--section NAME]

Writes data/manifests/<tree>_<section>.tsv, one row per file, plus a
_summary.tsv giving per-section totals.
"""

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "section",
    "tree",
    "link_path",
    "real_path",
    "size_bytes",
    "size_h",
    "mtime_utc",
    "sha256",
    "missing",
]


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def walk_tree(root, tree, do_checksum):
    """Yield one row per regular file reachable under root.

    Symlinked directories are followed, since that is how whole sequencing
    runs are staged, but each file is reported by its resolved real path so
    the manifest stays valid if the package is moved.
    """
    base = root / "data" / tree
    if not base.is_dir():
        return

    for section_dir in sorted(p for p in base.iterdir() if p.is_dir() or p.is_symlink()):
        section = section_dir.name
        for dirpath, dirnames, filenames in os.walk(section_dir, followlinks=True):
            dirnames[:] = sorted(
                d for d in dirnames if d not in {"__pycache__", ".ipynb_checkpoints"}
            )
            for name in sorted(filenames):
                link_path = Path(dirpath) / name
                try:
                    real = link_path.resolve(strict=True)
                    stat = real.stat()
                except (OSError, RuntimeError):
                    yield {
                        "section": section,
                        "tree": tree,
                        "link_path": str(link_path.relative_to(root)),
                        "real_path": str(os.readlink(link_path))
                        if link_path.is_symlink()
                        else str(link_path),
                        "size_bytes": "",
                        "size_h": "",
                        "mtime_utc": "",
                        "sha256": "",
                        "missing": "yes",
                    }
                    continue

                yield {
                    "section": section,
                    "tree": tree,
                    "link_path": str(link_path.relative_to(root)),
                    "real_path": str(real),
                    "size_bytes": stat.st_size,
                    "size_h": human(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "sha256": sha256(real) if do_checksum else "",
                    "missing": "",
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--checksum", action="store_true", help="hash every file (slow)")
    ap.add_argument("--section", help="limit to one section, e.g. phosphoPCA")
    args = ap.parse_args()

    root = args.root.resolve()
    out_dir = root / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {}
    for tree in ("raw", "results"):
        rows = [
            r
            for r in walk_tree(root, tree, args.checksum)
            if args.section is None or r["section"] == args.section
        ]
        if not rows:
            continue

        by_section = {}
        for r in rows:
            by_section.setdefault(r["section"], []).append(r)

        for section, srows in by_section.items():
            out = out_dir / f"{tree}_{section}.tsv"
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, FIELDS, delimiter="\t")
                w.writeheader()
                w.writerows(srows)

            n_missing = sum(1 for r in srows if r["missing"])
            nbytes = sum(int(r["size_bytes"]) for r in srows if r["size_bytes"])
            totals[(tree, section)] = (len(srows), nbytes, n_missing)
            print(
                f"{out.relative_to(root)}: {len(srows)} files, {human(nbytes)}"
                + (f", {n_missing} MISSING" if n_missing else "")
            )

    summary = out_dir / "_summary.tsv"
    with summary.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["tree", "section", "n_files", "total_bytes", "total_h", "n_missing"])
        for (tree, section), (n, nbytes, miss) in sorted(totals.items()):
            w.writerow([tree, section, n, nbytes, human(nbytes), miss])

    if not totals:
        print("nothing to manifest", file=sys.stderr)
        return 1
    print(f"\n{summary.relative_to(root)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
