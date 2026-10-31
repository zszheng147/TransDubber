"""Isometric-S2ST timeline metrics: pause alignment on the absolute time axis.

Score source and generated silence structure without duration rescaling:

  * tIoU      — temporal IoU of the silence sets on the absolute axis.
                Single headline number: penalises missed pauses, spurious
                pauses, misplaced pauses, wrong pause LENGTH, and (because
                non-overlapping tails fall out of the intersection) duration
                mismatch, all at once.
  * SER       — structure error rate: (miss + false-alarm + |dT|) / T_src,
                where miss = source-silence covered by generated speech,
                fa = generated silence where the source speaks (both within
                the common span), |dT| = length mismatch overhang.
                Additive & interpretable (diarization-style decomposition).
  * F1@tau    — event-level matching of >=min-event pauses by midpoint
                distance (greedy, one-to-one) with tolerance tau; reported
                for tau in {0.2s, 0.4s}, with matched-pair onset/offset
                median absolute errors (boundary precision).
  * duration  — ratio stats and within-10/5/2 rates, plus every timeline
                metric restated on the within-10 subset (pause quality
                *conditional on* meeting the isometric precondition).

Layout matches silence_iou.py: --root with audio/ + translated_texts.txt,
--source-audio-dir for the source wavs.  Source-side VAD results are cached
to a JSON so multi-checkpoint sweeps only VAD the generated side.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from silence_iou import (  # noqa: E402
    merge_intervals,
    silence_overlap,
    silence_total,
    vad_silences,
)

Interval = Tuple[float, float]


# ---------------------------------------------------------------------------
# Interval algebra helpers
# ---------------------------------------------------------------------------
def clip_intervals(intervals: List[Interval], lo: float, hi: float) -> List[Interval]:
    out = []
    for s, e in intervals:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


def subtract_total(a: List[Interval], b: List[Interval]) -> float:
    """Total length of A \\ B for disjoint sorted interval sets."""
    return silence_total(a) - silence_overlap(a, b)


# ---------------------------------------------------------------------------
# Core per-sample scoring
# ---------------------------------------------------------------------------
def timeline_scores(
    src_sil: List[Interval],
    t_src: float,
    gen_sil: List[Interval],
    t_gen: float,
    min_event_s: float,
    taus: List[float],
    eps: float = 1e-6,
) -> Dict[str, float]:
    src_sil = merge_intervals(src_sil)
    gen_sil = merge_intervals(gen_sil)

    # --- timeline metrics on the absolute axis --------------------------
    common = min(t_src, t_gen)
    src_c = clip_intervals(src_sil, 0.0, common)
    gen_c = clip_intervals(gen_sil, 0.0, common)
    inter = silence_overlap(src_c, gen_c)
    # Union over the FULL axes: silence outside the common span can never
    # intersect, so it counts against IoU — duration mismatch hurts.
    union = silence_total(src_sil) + silence_total(gen_sil) - inter
    tiou = inter / union if union > eps else (1.0 if not src_sil and not gen_sil else 0.0)

    miss = subtract_total(src_c, gen_c)          # src silent, gen speaking
    fa = subtract_total(gen_c, src_c)            # gen silent, src speaking
    overhang = abs(t_gen - t_src)                # length mismatch, either side
    ser = (miss + fa + overhang) / max(t_src, eps)

    # --- event-level matching -------------------------------------------
    src_ev = [iv for iv in src_sil if iv[1] - iv[0] >= min_event_s]
    gen_ev = [iv for iv in gen_sil if iv[1] - iv[0] >= min_event_s]
    out: Dict[str, float] = {
        "tiou": tiou,
        "ser": ser,
        "ser_miss": miss / max(t_src, eps),
        "ser_fa": fa / max(t_src, eps),
        "ser_overhang": overhang / max(t_src, eps),
        "n_src_events": float(len(src_ev)),
        "n_gen_events": float(len(gen_ev)),
    }
    for tau in taus:
        # greedy one-to-one midpoint matching
        pairs = []
        used = set()
        for i, (ss, se) in enumerate(src_ev):
            sc = 0.5 * (ss + se)
            best_j, best_d = -1, float("inf")
            for j, (gs, ge) in enumerate(gen_ev):
                if j in used:
                    continue
                d = abs(0.5 * (gs + ge) - sc)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j >= 0 and best_d <= tau:
                used.add(best_j)
                pairs.append((src_ev[i], gen_ev[best_j]))
        n_match = len(pairs)
        prec = n_match / len(gen_ev) if gen_ev else (1.0 if not src_ev else 0.0)
        rec = n_match / len(src_ev) if src_ev else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > eps else 0.0
        key = f"{tau:.1f}".replace(".", "")
        out[f"f1_at_{key}"] = f1
        out[f"prec_at_{key}"] = prec
        out[f"rec_at_{key}"] = rec
        if pairs:
            out[f"onset_mae_at_{key}"] = statistics.median(
                abs(p[0][0] - p[1][0]) for p in pairs
            )
            out[f"offset_mae_at_{key}"] = statistics.median(
                abs(p[0][1] - p[1][1]) for p in pairs
            )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="dir with translated_texts.txt; generated wavs under --generated-subdir")
    p.add_argument("--generated-subdir", default="audio")
    p.add_argument("--translated-texts", default=None)
    p.add_argument("--source-audio-dir", required=True,
                   help="dir with the source <name>.wav files")
    p.add_argument("--source-vad-cache", default=None,
                   help="JSON cache of source-side VAD (default <root>/source_vad_cache.json; "
                        "share one file across checkpoints of the same test set)")
    p.add_argument("--vad-sample-rate", type=int, default=16000)
    p.add_argument("--vad-threshold", type=float, default=0.8)
    p.add_argument("--min-speech-ms", type=int, default=250)
    p.add_argument("--min-silence-ms", type=int, default=100)
    p.add_argument("--speech-pad-ms", type=int, default=30)
    p.add_argument("--min-event-ms", type=int, default=400,
                   help="min pause length for EVENT metrics (timeline metrics use all >=min-silence)")
    p.add_argument("--drop-edge-silence", action="store_true", default=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-tsv", default=None)
    return p.parse_args()


def load_vad_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def main():
    args = parse_args()
    if not args.source_vad_cache:
        args.source_vad_cache = os.path.join(args.root, "source_vad_cache.json")
    texts = args.translated_texts or os.path.join(args.root, "translated_texts.txt")
    names = []
    with open(texts) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                names.append(parts[0])

    model = load_vad_model()
    vad_kwargs = dict(
        sampling_rate=args.vad_sample_rate,
        threshold=args.vad_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        # Timeline metrics use ALL silences >= min-silence; the >=min-event
        # filter for event metrics is applied inside timeline_scores.
        min_report_silence_ms=args.min_silence_ms,
        drop_edge=args.drop_edge_silence,
    )

    # ---- source VAD with cache -----------------------------------------
    cache: Dict[str, Dict] = {}
    if args.source_vad_cache and os.path.exists(args.source_vad_cache):
        with open(args.source_vad_cache) as f:
            cache = json.load(f)
    cache_dirty = False

    rows = []
    taus = [0.2, 0.4]
    skipped = 0
    for name in names:
        src_path = os.path.join(args.source_audio_dir, name + ".wav")
        gen_path = os.path.join(args.root, args.generated_subdir, name + ".wav")
        if not (os.path.exists(src_path) and os.path.exists(gen_path)):
            skipped += 1
            continue
        ck = f"{name}|{args.vad_threshold}|{args.min_silence_ms}|{args.speech_pad_ms}|{int(args.drop_edge_silence)}"
        if ck in cache:
            src_sil = [tuple(x) for x in cache[ck]["sil"]]
            t_src = float(cache[ck]["dur"])
        else:
            src_sil, t_src = vad_silences(src_path, model, **vad_kwargs)
            cache[ck] = {"sil": [list(x) for x in src_sil], "dur": t_src}
            cache_dirty = True
        gen_sil, t_gen = vad_silences(gen_path, model, **vad_kwargs)

        s = timeline_scores(
            src_sil, t_src, gen_sil, t_gen,
            min_event_s=args.min_event_ms / 1000.0, taus=taus,
        )
        ratio = t_gen / max(t_src, 1e-6)
        s["duration_ratio"] = ratio
        s["within_10"] = float(abs(ratio - 1.0) <= 0.10)
        s["within_5"] = float(abs(ratio - 1.0) <= 0.05)
        s["within_2"] = float(abs(ratio - 1.0) <= 0.02)
        s["name"] = name
        s["has_src_pause"] = float(s["n_src_events"] > 0)
        rows.append(s)

    if cache_dirty and args.source_vad_cache:
        os.makedirs(os.path.dirname(args.source_vad_cache), exist_ok=True)
        with open(args.source_vad_cache, "w") as f:
            json.dump(cache, f)

    def agg(sub, key):
        vals = [r[key] for r in sub if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    pauseful = [r for r in rows if r["has_src_pause"] > 0]
    iso = [r for r in pauseful if r["within_10"] > 0]
    summary = {
        "n_samples": len(rows),
        "n_skipped": skipped,
        "n_with_src_pause": len(pauseful),
        "n_isometric10_with_pause": len(iso),
        "duration": {
            "ratio_mean": agg(rows, "duration_ratio"),
            "within_10": agg(rows, "within_10"),
            "within_5": agg(rows, "within_5"),
            "within_2": agg(rows, "within_2"),
        },
        "all_pauseful": {
            k: agg(pauseful, k) for k in (
                "tiou", "ser", "ser_miss", "ser_fa", "ser_overhang",
                "f1_at_02", "f1_at_04", "prec_at_04", "rec_at_04",
                "onset_mae_at_04", "offset_mae_at_04",
            )
        },
        "isometric10_pauseful": {
            k: agg(iso, k) for k in (
                "tiou", "ser", "f1_at_02", "f1_at_04",
                "onset_mae_at_04", "offset_mae_at_04",
            )
        },
    }
    out_json = args.out_json or os.path.join(args.root, "timeline_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    out_tsv = args.out_tsv or os.path.join(args.root, "timeline_per_sample.tsv")
    if rows:
        keys = ["name"] + [k for k in rows[0] if k != "name"]
        with open(out_tsv, "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
