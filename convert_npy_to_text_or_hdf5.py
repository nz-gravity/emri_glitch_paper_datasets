#!/usr/bin/env python3
"""
Convert .npy (and .npz) files to human-readable text (CSV/TSV)
and/or a consolidated HDF5 file.

Usage examples:

  - Convert all .npy recursively to CSVs in exports/:
      python convert_npy_to_text_or_hdf5.py --format text

  - Use TSV and higher precision:
      python convert_npy_to_text_or_hdf5.py --format text --delimiter tab --floatfmt %.16g

  - Also create HDF5 at exports/data.h5:
      python convert_npy_to_text_or_hdf5.py --format both

  - Only HDF5:
      python convert_npy_to_text_or_hdf5.py --format hdf5

Requirements:
  - NumPy (required)
  - h5py (optional, only if using --format hdf5 or both)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # h5py is optional
    h5py = None  # type: ignore


def iter_target_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    for pat in patterns:
        # recursive glob
        yield from root.rglob(pat)


def safe_relpath(path: Path, start: Path) -> Path:
    try:
        return path.relative_to(start)
    except Exception:
        return path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_array_as_text(
    arr: np.ndarray,
    out_path: Path,
    delimiter: str = ",",
    floatfmt: str = "%.10g",
) -> None:
    ensure_dir(out_path.parent)

    # Structured arrays: write as CSV with header of field names
    if arr.dtype.names:
        fieldnames = list(arr.dtype.names)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(fieldnames)
            for row in arr:
                writer.writerow([row[name] for name in fieldnames])
        return

    # For numeric/bool dtypes: use np.savetxt. For ndim>2 reshape and keep a header
    if arr.dtype.kind in "iufb":
        header = f"original_shape={arr.shape} dtype={arr.dtype}"
        if arr.ndim == 1:
            data = arr
        elif arr.ndim == 2:
            data = arr
        else:
            # Flatten last dims into columns; keep first dim as rows
            first = arr.shape[0]
            data = arr.reshape(first, -1)
            header += " reshaped_to_rows=arr.shape[0], cols=product(remaining_dims)"
        np.savetxt(
            out_path,
            data,
            delimiter=delimiter,
            fmt=floatfmt if arr.dtype.kind == "f" else "%s",
            header=header,
            comments="# ",
        )
        return

    # Fallback: write repr() per element (best-effort for object/other dtypes)
    with out_path.open("w") as f:
        f.write(f"# original_shape={arr.shape} dtype={arr.dtype} (repr format)\n")
        if arr.ndim <= 1:
            for x in arr:
                f.write(repr(x) + "\n")
        else:
            # flatten to 2D rows
            first = arr.shape[0]
            flat = arr.reshape(first, -1)
            for row in flat:
                f.write(delimiter.join(repr(x) for x in row) + "\n")


def add_array_to_hdf5(
    h5: "h5py.File",
    group_path: str,
    name: str,
    arr: np.ndarray,
) -> None:
    grp = h5.require_group(group_path)
    dset_path = f"{group_path}/{name}" if group_path else name
    if dset_path in h5:
        del h5[dset_path]
    grp.create_dataset(name, data=arr)


def convert_file(
    src_path: Path,
    root: Path,
    outdir: Path,
    write_text: bool,
    write_hdf5: bool,
    delimiter: str,
    floatfmt: str,
    hdf5_file: Optional["h5py.File"],
) -> None:
    rel = safe_relpath(src_path, root)
    base = src_path.stem
    rel_parent = rel.parent

    if src_path.suffix == ".npz":
        with np.load(src_path, allow_pickle=False) as npz:
            for key in npz.files:
                arr = npz[key]
                if write_text:
                    out_path = outdir / rel_parent / f"{base}__{key}.csv"
                    save_array_as_text(arr, out_path, delimiter=delimiter, floatfmt=floatfmt)
                if write_hdf5 and hdf5_file is not None:
                    group_path = str(rel_parent).replace("\\", "/")
                    add_array_to_hdf5(hdf5_file, group_path, f"{base}__{key}", arr)
    else:  # .npy
        arr = np.load(src_path, allow_pickle=False)
        if write_text:
            out_path = outdir / rel_parent / f"{base}.csv"
            save_array_as_text(arr, out_path, delimiter=delimiter, floatfmt=floatfmt)
        if write_hdf5 and hdf5_file is not None:
            group_path = str(rel_parent).replace("\\", "/")
            add_array_to_hdf5(hdf5_file, group_path, base, arr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search (default: CWD)",
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["*.npy", "*.npz"],
        help="Glob patterns to include (recursive). Default: *.npy *.npz",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("exports"),
        help="Directory for exported text files and HDF5 (default: exports/)",
    )
    p.add_argument(
        "--format",
        choices=["text", "hdf5", "both"],
        default="text",
        help="What to write: text, hdf5, or both (default: text)",
    )
    p.add_argument(
        "--delimiter",
        choices=[",", "tab", "space"],
        default=",",
        help="Delimiter for text output (default: ,)",
    )
    p.add_argument(
        "--floatfmt",
        default="%.10g",
        help="Printf-style float format for text output (default: %.10g)",
    )
    p.add_argument(
        "--hdf5-path",
        type=Path,
        default=None,
        help="Path for HDF5 file (default: exports/data.h5)",
    )
    args = p.parse_args()

    delimiter = {",": ",", "tab": "\t", "space": " "}[args.delimiter]
    write_text = args.format in ("text", "both")
    write_hdf5 = args.format in ("hdf5", "both")

    ensure_dir(args.outdir)

    h5: Optional["h5py.File"] = None
    if write_hdf5:
        if h5py is None:
            raise SystemExit("h5py is not installed but --format includes hdf5")
        h5_path = args.hdf5_path or (args.outdir / "data.h5")
        ensure_dir(h5_path.parent)
        h5 = h5py.File(h5_path, "w")

    try:
        files = sorted({Path(p) for p in iter_target_files(args.root, args.patterns)})
        if not files:
            print("No files matched; check --root and --patterns.")
            return
        print(f"Found {len(files)} file(s). Converting...")
        for src in files:
            # Skip writing into exports again
            if args.outdir in src.parents:
                continue
            convert_file(
                src_path=src,
                root=args.root,
                outdir=args.outdir,
                write_text=write_text,
                write_hdf5=write_hdf5,
                delimiter=delimiter,
                floatfmt=args.floatfmt,
                hdf5_file=h5,
            )
            print(f"Converted: {safe_relpath(src, args.root)}")
        print("Done.")
        if write_hdf5 and h5 is not None:
            print(f"HDF5 written to: {h5.filename}")
        if write_text:
            print(f"Text files written under: {args.outdir}")
    finally:
        if h5 is not None:
            h5.close()


if __name__ == "__main__":
    main()

