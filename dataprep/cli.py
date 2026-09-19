import argparse
import sys
from pathlib import Path

from .errors import DataPrepError
from .io import LoadOptions, load_data
from .pipeline import parse_config, run_pipeline
from .report import export_bundle
from .serialization import dumps


def main(argv=None):
    # Windows redirected terminals can default to cp1252 even for Vietnamese paths.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="EDA + cleaning + offline Plotly report")
    parser.add_argument("source", type=Path)
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument(
        "--ml-config", type=Path, help="Prepare ML data from original source with a JSON ML config"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter")
    parser.add_argument("--decimal", default=".", choices=[".", ","])
    parser.add_argument("--json-path", default="")
    parser.add_argument("--table", default="")
    parser.add_argument("--sheet", default=None, help="Excel sheet name; default first sheet")
    parser.add_argument("--missing-token", action="append", default=[])
    parser.add_argument("--no-infer-numeric", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output artifacts")
    args = parser.parse_args(argv)
    try:
        if args.pipeline and args.ml_config:
            raise DataPrepError(
                "Dùng --pipeline cho EDA hoặc --ml-config cho ML; không chuẩn hóa ML trên kết quả fit toàn bộ dữ liệu."
            )
        names = [
            "report.html",
            "cleaned_data.csv",
            "pipeline.json",
            "processing_log.json",
            "schema.json",
            "source_metadata.json",
            "result.zip",
        ]
        if args.ml_config:
            names = ["ml_ready.zip", "ml_manifest.json", "ml_config.json", "source_metadata.json"]
        targets = [args.output / name for name in names]
        if not args.overwrite and any(path.exists() for path in targets):
            raise DataPrepError("Đầu ra đã tồn tại; chọn thư mục mới hoặc dùng --overwrite.")
        if args.source.resolve() in [p.resolve() for p in targets]:
            raise DataPrepError("Không được ghi đè nguồn dữ liệu.")
        dataset = load_data(
            args.source,
            options=LoadOptions(
                encoding=args.encoding,
                delimiter=args.delimiter,
                decimal=args.decimal,
                json_path=args.json_path,
                table=args.table,
                sheet=args.sheet if args.sheet is not None else 0,
                missing_tokens=args.missing_token,
                infer_numeric=not args.no_infer_numeric,
            ),
        )
        if args.ml_config:
            from .ml import MLConfig, export_ml_bundle, prepare_ml

            ml_config = MLConfig.from_json(args.ml_config.read_text(encoding="utf-8-sig"))
            ml_result = prepare_ml(dataset.frame, ml_config)
            bundle = export_ml_bundle(ml_result)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "ml_ready.zip").write_bytes(bundle)
            (args.output / "ml_manifest.json").write_text(dumps(ml_result.manifest), encoding="utf-8")
            (args.output / "ml_config.json").write_text(dumps(ml_result.manifest["config"]), encoding="utf-8")
            (args.output / "source_metadata.json").write_text(dumps(dataset.metadata), encoding="utf-8")
            print(
                f"ML OK: {ml_result.manifest['split_rows']} | {len(ml_result.feature_names)} features | {args.output.resolve()}"
            )
            return 0
        config = (
            parse_config(args.pipeline.read_text(encoding="utf-8-sig"))
            if args.pipeline
            else {"version": 1, "roles": {}, "steps": []}
        )
        result = run_pipeline(dataset.frame, config)
        html, bundle = export_bundle(dataset.frame, result, dataset.name, dataset.metadata)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.html").write_text(html, encoding="utf-8")
        (args.output / "cleaned_data.csv").write_bytes(result.frame.to_csv(index=False).encode("utf-8-sig"))
        (args.output / "pipeline.json").write_text(dumps(config), encoding="utf-8")
        (args.output / "processing_log.json").write_text(dumps(result.log), encoding="utf-8")
        (args.output / "schema.json").write_text(
            dumps({c: str(result.frame[c].dtype) for c in result.frame}), encoding="utf-8"
        )
        (args.output / "source_metadata.json").write_text(dumps(dataset.metadata), encoding="utf-8")
        (args.output / "result.zip").write_bytes(bundle)
        print(f"OK: {len(dataset.frame)} -> {len(result.frame)} rows | {args.output.resolve()}")
        return 0
    except (DataPrepError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
