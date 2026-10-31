import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

from silero_vad import get_speech_timestamps, load_silero_vad


def read_audio_16k(wav_path: str, sampling_rate: int = 16000):
    """Load a wav as a 1-D float32 mono torch tensor at ``sampling_rate``.

    Uses soundfile (no ffmpeg/torchcodec dependency), downmixes to mono and
    resamples via polyphase filtering when the file rate differs.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sampling_rate:
        divisor = math.gcd(sr, sampling_rate)
        wav = resample_poly(wav, sampling_rate // divisor, sr // divisor)
    return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))


def coerce_silence_intervals(
    intervals,
    min_silence_s: float,
    duration: Optional[float] = None,
    drop_edge: bool = True,
) -> List[Tuple[float, float]]:
    """Filter raw silence spans like ``PauseOnlyReward._coerce_silence_intervals``.

    Clamps each span to ``[0, duration]``, optionally drops edge-touching
    (head/tail) silences, and keeps only spans strictly longer than
    ``min_silence_s``. This is the exact filter the reward applies to both
    source and generated silences.
    """
    duration_f = float(duration) if duration is not None else None
    if duration_f is not None and (
        not math.isfinite(duration_f) or duration_f <= 0.0
    ):
        duration_f = None

    out: List[Tuple[float, float]] = []
    for seg in intervals or []:
        if isinstance(seg, dict):
            start, end = seg.get("start"), seg.get("end")
        else:
            if len(seg) < 2:
                continue
            start, end = seg[0], seg[1]
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start_f) and math.isfinite(end_f)):
            continue
        if duration_f is not None:
            start_f = max(0.0, min(start_f, duration_f))
            end_f = max(0.0, min(end_f, duration_f))
            if drop_edge and (start_f <= 0.0 or end_f >= duration_f):
                continue
        if end_f > start_f and end_f - start_f > min_silence_s:
            out.append((start_f, end_f))
    out.sort()
    return out


def vad_silences(
    wav_path: str,
    vad_model,
    sampling_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    min_report_silence_ms: int,
    drop_edge: bool = True,
) -> Tuple[List[Tuple[float, float]], float]:
    """Detect internal silences the way training does (``rewards.PauseOnlyReward``).

    Runs Silero VAD to get speech spans, takes their complement over
    ``[0, duration]`` as candidate silences, then coerces them (drop edge
    silences, keep only spans longer than ``min_report_silence_ms``). This
    mirrors ``_generated_silence_intervals`` + ``_coerce_silence_intervals`` so
    the eval scores the same silence definition the reward optimises.
    """
    wav = read_audio_16k(wav_path, sampling_rate=sampling_rate)
    total_samples = wav.numel()
    if total_samples == 0:
        return [], 0.0
    duration = total_samples / float(sampling_rate)

    speech_ts = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if not speech_ts:
        return [], duration

    # Silences are the complement of speech spans: pair up
    # [0, s0, e0, s1, e1, ..., sN, eN, D] into (0,s0),(e0,s1),...,(eN,D).
    edges = (
        [0.0]
        + [t for seg in speech_ts for t in (seg["start"], seg["end"])]
        + [duration]
    )
    silences = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]

    min_report_s = float(min_report_silence_ms) / 1000.0
    silences = coerce_silence_intervals(
        silences, min_report_s, duration=duration, drop_edge=drop_edge,
    )
    return silences, duration


Interval = Tuple[float, float]


def silence_total(intervals: List[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Sort and merge overlapping/touching intervals into a disjoint, ordered set."""
    if not intervals:
        return []
    merged: List[List[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def silence_overlap(a_intervals: List[Interval], b_intervals: List[Interval]) -> float:
    """Total intersection length of two disjoint, sorted interval sets."""
    overlap = 0.0
    a_idx = 0
    b_idx = 0
    while a_idx < len(a_intervals) and b_idx < len(b_intervals):
        a_start, a_end = a_intervals[a_idx]
        b_start, b_end = b_intervals[b_idx]
        overlap += max(0.0, min(a_end, b_end) - max(a_start, b_start))
        if a_end <= b_end:
            a_idx += 1
        else:
            b_idx += 1
    return overlap


def scale_intervals(intervals: List[Interval], scale: float) -> List[Interval]:
    """Map intervals from the target timeline onto the source timeline by duration ratio."""
    return [(start * scale, end * scale) for start, end in intervals]


def shrink_intervals(intervals: List[Interval], tau: float) -> List[Interval]:
    """Erode each interval by ``tau`` on both sides; drop intervals that vanish."""
    out: List[Interval] = []
    for start, end in intervals:
        s, e = start + tau, end - tau
        if e > s:
            out.append((s, e))
    return out


def complement_intervals(
    intervals: List[Interval], lo: float, hi: float
) -> List[Interval]:
    """Return [lo, hi] minus the union of ``intervals`` (intervals dilated upstream)."""
    out: List[Interval] = []
    cur = lo
    for start, end in merge_intervals(intervals):
        start = max(start, lo)
        end = min(end, hi)
        if end <= cur:
            continue
        if start > cur:
            out.append((cur, start))
        cur = max(cur, end)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None,
                        help="eval dir with translated_texts.txt (+ audio/)")
    parser.add_argument("--roots", nargs="+", default=None)
    # Per-root layout: source under <root>/source_audio, target under <root>/audio.
    parser.add_argument("--source-audio-dir", default=None,
                        help="Override source audio dir (default <root>/source_audio)")
    parser.add_argument("--generated-audio-dir", default=None,
                        help="Override target audio dir (default <root>/audio)")
    parser.add_argument("--translated-texts", default=None,
                        help="Override eval list (default <root>/translated_texts.txt)")
    parser.add_argument("--vad-sample-rate", type=int, default=16000)
    parser.add_argument("--vad-threshold", type=float, default=0.8)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--min-report-silence-ms", type=int, default=400)
    # Training (rewards.PauseOnlyReward) drops head/tail silences; keep parity.
    parser.add_argument("--no-drop-edge-silence", dest="drop_edge_silence",
                        action="store_false",
                        help="Keep head/tail (edge) silences (training drops them)")
    parser.set_defaults(drop_edge_silence=True)
    # Collar (tau) used by the collared-silence-compliance metrics, in seconds.
    parser.add_argument("--collar", type=float, default=0.10,
                        help="Collar tau in seconds for eroding/dilating source silences")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Denominator epsilon to avoid div-by-zero (seconds)")
    parser.add_argument("--per-sample-out", default=None,
                        help="Write a per-sample TSV here (default <root>/silence_per_sample.tsv)")
    return parser.parse_args()


def collared_metrics(
    src_silences: List[Interval],
    src_duration: float,
    gen_silences: List[Interval],
    gen_duration: float,
    tau: float,
    eps: float,
) -> dict:
    """Collared silence-compliance metrics on the shared (source) timeline.

    The target timeline is linearly mapped onto the source timeline by the
    duration ratio, then:
      * S_tau  = source silences eroded by tau on each side (the "core" region
                 where target silence is required).
      * P_tau  = source speech region = [0, D] minus source silences dilated by
                 tau (the region where target silence is unexpected).
      * T      = mapped target silences.
    """
    scale = (src_duration / gen_duration) if gen_duration > 0 else 1.0
    target = merge_intervals(scale_intervals(gen_silences, scale))
    source = merge_intervals(src_silences)

    # 1) Core source silence (eroded). Target should be silent here.
    core = shrink_intervals(source, tau)
    core_len = silence_total(core)
    core_hit = silence_overlap(core, target)
    compliance = core_hit / (core_len + eps)
    missed = (core_len - core_hit) / (core_len + eps)

    # 2) Reliable source speech (silences dilated, then complemented).
    speech = complement_intervals([(s - tau, e + tau) for s, e in source], 0.0, src_duration)
    speech_len = silence_total(speech)
    speech_hit = silence_overlap(speech, target)
    extra = speech_hit / (speech_len + eps)

    return {
        "core_len": core_len,
        "core_hit": core_hit,
        "speech_len": speech_len,
        "speech_hit": speech_hit,
        "silence_compliance": compliance,
        "missed_silence_rate": missed,
        "extra_silence_rate": extra,
        "has_core": core_len > eps,
        "has_speech": speech_len > eps,
    }


def evaluate_root(
    args,
    root: str,
    vad_model,
    source_cache: Dict[str, Tuple[List[Interval], float]],
) -> Tuple[List[dict], int]:
    source_audio_dir = args.source_audio_dir or os.path.join(root, "source_audio")
    generated_audio_dir = args.generated_audio_dir or os.path.join(root, "audio")
    translated_texts = args.translated_texts or os.path.join(root, "translated_texts.txt")

    rows = []
    skipped_no_source_silence = 0
    with open(translated_texts) as f:
        for line in f:
            if not line.strip():
                continue
            fname = line.strip().split("\t")[0]
            wav_name = fname + ".wav"
            source_path = os.path.join(source_audio_dir, wav_name)
            generated_path = os.path.join(generated_audio_dir, wav_name)
            if not os.path.exists(source_path) or not os.path.exists(generated_path):
                continue

            if source_path not in source_cache:
                source_cache[source_path] = vad_silences(
                    source_path,
                    vad_model,
                    args.vad_sample_rate,
                    args.vad_threshold,
                    args.min_speech_ms,
                    args.min_silence_ms,
                    args.speech_pad_ms,
                    args.min_report_silence_ms,
                    args.drop_edge_silence,
                )
            src_silences, src_duration = source_cache[source_path]
            if not src_silences:
                skipped_no_source_silence += 1
                continue

            gen_silences, gen_duration = vad_silences(
                generated_path,
                vad_model,
                args.vad_sample_rate,
                args.vad_threshold,
                args.min_speech_ms,
                args.min_silence_ms,
                args.speech_pad_ms,
                args.min_report_silence_ms,
                args.drop_edge_silence,
            )

            row = collared_metrics(
                src_silences,
                src_duration,
                gen_silences,
                gen_duration,
                args.collar,
                args.eps,
            )
            row["name"] = fname
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No paired wavs with qualifying source silence found under {generated_audio_dir}"
        )
    return rows, skipped_no_source_silence


def write_per_sample(path: str, rows: List[dict]) -> None:
    # (output column header, source key in the row dict)
    cols = [
        ("name", "name"),
        ("followed_silence", "silence_compliance"),  # higher = better
        ("added_silence", "extra_silence_rate"),      # lower  = better
    ]
    with open(path, "w") as f:
        f.write("\t".join(header for header, _ in cols) + "\n")
        for r in rows:
            vals = []
            for _, key in cols:
                v = r[key]
                vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            f.write("\t".join(vals) + "\n")
    print(f"per_sample_out\t{path}")


def print_summary(root: str, rows: List[dict], skipped_no_source_silence: int, tau: float) -> None:
    core_rows = [r for r in rows if r["has_core"]]
    speech_rows = [r for r in rows if r["has_speech"]]

    def mean(items, key):
        return sum(r[key] for r in items) / len(items) if items else float("nan")

    mean_compliance = mean(core_rows, "silence_compliance")
    mean_missed = mean(core_rows, "missed_silence_rate")
    mean_extra = mean(speech_rows, "extra_silence_rate")

    total_core = sum(r["core_len"] for r in rows)
    total_core_hit = sum(r["core_hit"] for r in rows)
    total_speech = sum(r["speech_len"] for r in rows)
    total_speech_hit = sum(r["speech_hit"] for r in rows)
    micro_compliance = total_core_hit / total_core if total_core > 0 else float("nan")
    micro_missed = (total_core - total_core_hit) / total_core if total_core > 0 else float("nan")
    micro_extra = total_speech_hit / total_speech if total_speech > 0 else float("nan")

    print(f"root\t{root}")
    print(f"collar_tau_sec\t{tau:.3f}")
    print(f"num_samples\t{len(rows)}")
    print(f"skipped_no_source_silence\t{skipped_no_source_silence}")
    print(f"silence_compliance\t{mean_compliance:.6f}")
    print(f"missed_silence_rate\t{mean_missed:.6f}")
    print(f"extra_silence_rate\t{mean_extra:.6f}")
    # print(f"silence_compliance_micro\t{micro_compliance:.6f}")
    # print(f"missed_silence_rate_micro\t{micro_missed:.6f}")
    # print(f"extra_silence_rate_micro\t{micro_extra:.6f}")


def main():
    args = parse_args()
    roots = args.roots or [args.root]
    vad_model = load_silero_vad()
    source_cache: Dict[str, Tuple[List[Interval], float]] = {}

    for idx, root in enumerate(roots):
        if idx:
            print()
        rows, skipped_no_source_silence = evaluate_root(args, root, vad_model, source_cache)
        print_summary(root, rows, skipped_no_source_silence, args.collar)
        per_sample_out = args.per_sample_out or os.path.join(root, "silence_per_sample.tsv")
        write_per_sample(per_sample_out, rows)


if __name__ == "__main__":
    main()
