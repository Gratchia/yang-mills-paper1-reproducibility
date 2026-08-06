"""Build a relative-path SHA256 manifest for the C5X reproduction package."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def local_module_map(calculations: Path) -> dict[str, Path]:
    return {path.stem: path for path in calculations.glob("*.py")}


def imported_local_modules(path: Path, modules: dict[str, Path]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names.intersection(modules)


def import_closure(root: Path, entries: list[str]) -> set[Path]:
    calculations = root / "calculations"
    modules = local_module_map(calculations)
    pending = [root / entry for entry in entries]
    closure: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in closure:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        closure.add(path)
        for module in imported_local_modules(path, modules):
            dependency = modules[module].resolve()
            if dependency not in closure:
                pending.append(dependency)
    return closure


def file_record(root: Path, path: Path) -> dict[str, object]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    payload = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    closure = import_closure(root, args.entry)
    included = {root / path for path in args.include}
    files = sorted(closure.union(path.resolve() for path in included))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    records = [file_record(root, path) for path in files]
    manifest = {
        "schema": "c5x-reproducibility-manifest-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "entry_scripts": sorted(args.entry),
        "local_python_dependency_count": len(closure),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "files": records,
    }

    output_json = Path(args.output_json)
    output_text = Path(args.output_text)
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_text.write_text(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
