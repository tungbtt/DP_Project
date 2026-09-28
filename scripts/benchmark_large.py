"""Opt-in end-to-end benchmark; generated files stay under ignored outputs/."""

import argparse
import csv
import ctypes
import gc
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd

from dataprep.io import MAX_ROWS, load_data
from dataprep.ml import MLConfig, prepare_ml
from dataprep.pipeline import run_pipeline
from dataprep.profile import profile_data
from dataprep.report import export_bundle


def peak_rss_mib():
    """Process high-water RSS; includes interpreter and imported libraries."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1024**2
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--ml", action="store_true", help="Also split, impute, scale and one-hot encode")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_2m"))
    args = parser.parse_args()
    if not 10 <= args.rows <= MAX_ROWS:
        parser.error(f"--rows must be between 10 and {MAX_ROWS}")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "synthetic.csv"
    timings = {}

    def stage(name, function):
        print(f"{name}...", flush=True)
        start = time.perf_counter()
        value = function()
        timings[name] = round(time.perf_counter() - start, 3)
        print(f"  {timings[name]} s; peak RSS {peak_rss_mib():.1f} MiB", flush=True)
        return value

    def generate():
        with source.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["record_id", "value", "quantity", "city", "target"])
            for start in range(0, args.rows, 50_000):
                writer.writerows(
                    (
                        f"{i:09}",
                        "" if i % 1000 == 0 else i % 10000 / 10,
                        i % 20 + 1,
                        [" HN ", "HCM", "DN", "Hue"][i % 4],
                        i % 5000 / 5,
                    )
                    for i in range(start, min(start + 50_000, args.rows))
                )

    stage("generate_csv", generate)
    dataset = stage("load", lambda: load_data(source))
    assert len(dataset.frame) == args.rows
    assert dataset.frame.record_id.iloc[0] == "000000000"
    profile = stage("profile", lambda: profile_data(dataset.frame))
    expected_missing = (args.rows + 999) // 1000
    assert profile["overview"]["missing_cells"] == expected_missing
    assert profile["overview"]["duplicate_rows"] == 0
    config = {
        "version": 1,
        "roles": profile["roles"],
        "steps": [
            {"op": "fill_missing", "columns": ["value"], "strategy": "median"},
            {"op": "normalize_text", "columns": ["city"]},
        ],
    }
    result = stage("clean", lambda: run_pipeline(dataset.frame, config))
    assert len(result.frame) == args.rows and not result.frame.value.isna().any()
    assert dataset.frame.value.isna().sum() == expected_missing
    assert [entry["changed_cells_retained"] for entry in result.log] == [
        expected_missing,
        (args.rows + 3) // 4,
    ]

    def export():
        html, bundle = export_bundle(dataset.frame, result, "2M benchmark", dataset.metadata)
        (args.output / "report.html").write_text(html, encoding="utf-8")
        (args.output / "result.zip").write_bytes(bundle)
        return len(bundle)

    zip_size = stage("html_and_zip", export)

    def verify_zip():
        with (
            zipfile.ZipFile(args.output / "result.zip") as archive,
            archive.open("cleaned_data.csv") as member,
            TextIOWrapper(member, encoding="utf-8-sig", newline="") as text,
        ):
            rows = csv.reader(text)
            assert next(rows) == list(result.frame)
            return sum(1 for _ in rows)

    exported_rows = stage("verify_export", verify_zip)
    assert exported_rows == args.rows
    result = None
    gc.collect()
    ml_summary = None
    if args.ml:
        ml = stage(
            "prepare_ml",
            lambda: prepare_ml(
                dataset.frame,
                MLConfig(
                    target="target", numeric_columns=["value", "quantity"], categorical_columns=["city"]
                ),
            ),
        )
        ml_summary = {split: list(matrix.shape) for split, matrix in ml.matrices.items()}
        assert sum(shape[0] for shape in ml_summary.values()) == args.rows
        assert all(np.isfinite(matrix.data).all() for matrix in ml.matrices.values())

    measurements = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "rows": args.rows,
        "columns": len(dataset.frame.columns),
        "source_bytes": source.stat().st_size,
        "dataframe_bytes": dataset.metadata["memory_bytes"],
        "zip_bytes": zip_size,
        "exported_rows": exported_rows,
        "timings_seconds": timings,
        "peak_process_rss_mib": round(peak_rss_mib(), 1),
        "ml_shapes": ml_summary,
        "scope": "Synthetic narrow CSV; local process, not browser/Cloud/concurrent users or arbitrary wide tables.",
    }
    (args.output / "results.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(json.dumps(measurements, indent=2), flush=True)


if __name__ == "__main__":
    main()
