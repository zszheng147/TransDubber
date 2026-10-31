#!/usr/bin/env python3
"""Compute generated/source duration-ratio metrics from audio-eval records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-audio-dir", type=Path, required=True)
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-sample-out", type=Path, default=None)
    return parser.parse_args()


def latest_successes(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("name", ""))
            if not name:
                raise ValueError(f"record without name at {path}:{line_number}")
            rows[name] = row
    return {
        name: row
        for name, row in rows.items()
        if row.get("status") == "ok"
        and int(row.get("wav_num_frames", 0)) > 0
        and int(row.get("wav_sample_rate", 0)) > 0
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    source_root = args.source_audio_dir.expanduser().resolve()
    records = latest_successes(root / "predictions.jsonl")

    per_sample: list[dict[str, float | str]] = []
    missing_source: list[str] = []
    for name, row in sorted(records.items()):
        source_path = source_root / f"{name}{args.audio_ext}"
        if not source_path.is_file():
            missing_source.append(name)
            continue
        source_duration = float(sf.info(str(source_path)).duration)
        generated_duration = (
            int(row["wav_num_frames"]) / float(row["wav_sample_rate"])
        )
        if not (
            math.isfinite(source_duration)
            and math.isfinite(generated_duration)
            and source_duration > 0
            and generated_duration > 0
        ):
            continue
        ratio = generated_duration / source_duration
        per_sample.append(
            {
                "name": name,
                "source_duration": source_duration,
                "generated_duration": generated_duration,
                "duration_ratio": ratio,
                "relative_error": abs(ratio - 1.0),
            }
        )

    if not per_sample:
        raise RuntimeError("no valid source/generated duration pairs")
    ratios = np.asarray(
        [float(row["duration_ratio"]) for row in per_sample], dtype=np.float64,
    )
    relative_errors = np.abs(ratios - 1.0)
    metrics = {
        "sample_count": len(per_sample),
        "missing_source_count": len(missing_source),
        "duration_ratio_mean": float(np.mean(ratios)),
        "duration_ratio_median": float(np.median(ratios)),
        "duration_ratio_p05": float(np.quantile(ratios, 0.05)),
        "duration_ratio_p95": float(np.quantile(ratios, 0.95)),
        "relative_error_mean": float(np.mean(relative_errors)),
        "within_2_rate": float(np.mean(relative_errors <= 0.02)),
        "within_5_rate": float(np.mean(relative_errors <= 0.05)),
        "within_10_rate": float(np.mean(relative_errors <= 0.10)),
    }

    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.per_sample_out is not None:
        per_sample_out = args.per_sample_out.expanduser().resolve()
        per_sample_out.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "name",
                    "source_duration",
                    "generated_duration",
                    "duration_ratio",
                    "relative_error",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(per_sample)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
