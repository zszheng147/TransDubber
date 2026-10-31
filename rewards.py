"""
Reward functions for GRPO training.
"""
import logging
import math
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


@contextmanager
def _temporary_logger_level(names: List[str], level: int):
    loggers = [logging.getLogger(name) for name in names]
    old_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, old_level in zip(loggers, old_levels):
            logger.setLevel(old_level)


# ---------------------------------------------------------------------------
# Isometric reward (duration matching)
# ---------------------------------------------------------------------------
def compute_isometric_reward(
    source_speech_lengths: torch.Tensor,
    generated_speech_lengths: torch.Tensor,
    source_frame_rate: float = 1.0,
    target_token_rate: float = 25.0,
    eps_free: float = 0.05,
    eps_ref: float = 0.10,
    p_max: float = 4.0,
) -> torch.Tensor:
    """Log-quadratic duration penalty in [-p_max, 0].

    Inside ±eps_free of ratio=1 the reward is 0. Outside the free band a
    log-distance `u` is computed so that |u|≈1 at the ±eps_ref edge; the
    penalty min(u², p_max) then grows quadratically and saturates. This
    continuous shape gives GRPO/GSPO a smooth intra-group ranking signal
    that the prior tier-based bonuses lacked, while still hitting the clip
    at very large mismatches (~±2*eps_ref).
    """
    source_duration = source_speech_lengths.float() / source_frame_rate
    target_duration = generated_speech_lengths.float() / target_token_rate

    ratio = target_duration / torch.clamp(source_duration, min=1e-5)
    ratio = ratio.clamp(min=1e-8)

    log_hi = math.log((1 + eps_ref) / (1 + eps_free))
    log_lo = math.log((1 - eps_free) / (1 - eps_ref))

    u_hi = torch.log(ratio / (1 + eps_free)) / log_hi
    u_lo = torch.log((1 - eps_free) / ratio) / log_lo

    u = torch.where(
        ratio > 1 + eps_free, u_hi,
        torch.where(ratio < 1 - eps_free, u_lo, torch.zeros_like(ratio)),
    )

    return -torch.clamp(u * u, max=p_max)


def compute_duration_within_metrics(
    source_speech_lengths: torch.Tensor,
    generated_speech_lengths: torch.Tensor,
    source_frame_rate: float = 1.0,
    target_token_rate: float = 25.0,
) -> Dict[str, List[float]]:
    """Return inclusive duration-band indicators for logging.

    The duration definition exactly matches ``compute_isometric_reward`` so
    the metrics describe the same codec-length proxy that drives training.
    """
    source_duration = source_speech_lengths.float() / source_frame_rate
    target_duration = generated_speech_lengths.float() / target_token_rate
    ratio = target_duration / torch.clamp(source_duration, min=1e-5)
    relative_error = torch.abs(ratio - 1.0)

    return {
        f"duration_within_{percent}": (
            relative_error <= tolerance + 1e-6
        ).float().cpu().tolist()
        for percent, tolerance in ((2, 0.02), (5, 0.05), (10, 0.10))
    }


def _checked_group_sizes(n_items: int, group_sizes: Optional[List[int]]) -> List[int]:
    if not group_sizes:
        return [n_items]
    total = sum(int(g) for g in group_sizes)
    if total != n_items:
        raise ValueError(
            f"group_sizes sum ({total}) does not match number of scores ({n_items})"
        )
    return [int(g) for g in group_sizes]


def compute_extreme_mask(
    rewards: torch.Tensor,
    group_sizes: List[int],
    top_k: int = 4,
    bottom_k: int = 4,
) -> torch.Tensor:
    """Per-group top-K best + bottom-K worst selection on a higher-is-better score.

    The top set takes the largest `top_k` scores; the bottom set takes the
    smallest `bottom_k` from what remains, so the two sets never overlap
    when `top_k + bottom_k > g`.
    """
    groups = _checked_group_sizes(rewards.numel(), group_sizes)
    keep = torch.zeros(rewards.numel(), dtype=torch.bool, device=rewards.device)
    top_k = max(int(top_k), 0)
    bottom_k = max(int(bottom_k), 0)

    offset = 0
    for g in groups:
        if g <= 0:
            continue
        chunk = rewards[offset: offset + g]
        order = torch.argsort(chunk, descending=True)  # best first
        top_n = min(top_k, g)
        bottom_n = min(bottom_k, max(g - top_n, 0))
        if top_n > 0:
            keep[offset + order[:top_n]] = True
        if bottom_n > 0:
            keep[offset + order[-bottom_n:]] = True
        offset += g
    return keep


# ---------------------------------------------------------------------------
# Pause-only reward (token2wav + internal-silence IoU)
# ---------------------------------------------------------------------------
def silence_total(intervals: List[Tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def silence_overlap(
    a_intervals: List[Tuple[float, float]],
    b_intervals: List[Tuple[float, float]],
) -> float:
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


def silence_iou(
    a_intervals: List[Tuple[float, float]],
    b_intervals: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    overlap = silence_overlap(a_intervals, b_intervals)
    union = silence_total(a_intervals) + silence_total(b_intervals) - overlap
    if union <= 1e-12:
        return 1.0, overlap, 0.0
    return max(0.0, min(1.0, overlap / union)), overlap, union


# Collared silence-compliance metrics (mirror eval/silence_iou.py). The reward
# is `followed_silence - lambda * added_silence`, where followed/added are the
# `silence_compliance` / `extra_silence_rate` of that script.
def merge_intervals(
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
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


def scale_intervals(
    intervals: List[Tuple[float, float]], scale: float,
) -> List[Tuple[float, float]]:
    """Map intervals from the target timeline onto the source timeline by duration ratio."""
    return [(start * scale, end * scale) for start, end in intervals]


def shrink_intervals(
    intervals: List[Tuple[float, float]], tau: float,
) -> List[Tuple[float, float]]:
    """Erode each interval by ``tau`` on both sides; drop intervals that vanish."""
    out: List[Tuple[float, float]] = []
    for start, end in intervals:
        s, e = start + tau, end - tau
        if e > s:
            out.append((s, e))
    return out


def complement_intervals(
    intervals: List[Tuple[float, float]], lo: float, hi: float,
) -> List[Tuple[float, float]]:
    """Return [lo, hi] minus the union of ``intervals`` (intervals dilated upstream)."""
    out: List[Tuple[float, float]] = []
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


def collared_silence_metrics(
    src_silences: List[Tuple[float, float]],
    src_duration: float,
    gen_silences: List[Tuple[float, float]],
    gen_duration: float,
    tau: float,
    eps: float,
    max_scale_deviation: Optional[float] = None,
) -> Dict[str, float]:
    """Collared followed/added-silence metrics on the shared (source) timeline.

    The generated timeline is linearly mapped onto the source timeline by the
    duration ratio, then:
      * core   = source silences eroded by ``tau`` on each side — the region
                 where generated silence is *required*. followed_silence is the
                 fraction of this core that the generated audio is silent in.
      * speech = source speech region (silences dilated by ``tau`` then
                 complemented). added_silence is the fraction of this speech
                 region the generated audio is (wrongly) silent in.
    """
    raw_scale = (src_duration / gen_duration) if gen_duration > 0 else 1.0
    scale = raw_scale
    if max_scale_deviation is not None:
        max_scale_deviation = max(0.0, float(max_scale_deviation))
        scale = min(
            1.0 + max_scale_deviation,
            max(1.0 - max_scale_deviation, scale),
        )
    target = merge_intervals(scale_intervals(gen_silences, scale))
    source = merge_intervals(src_silences)

    core = shrink_intervals(source, tau)
    core_len = silence_total(core)
    core_hit = silence_overlap(core, target)
    followed = core_hit / (core_len + eps)

    speech = complement_intervals(
        [(s - tau, e + tau) for s, e in source], 0.0, max(src_duration, 0.0),
    )
    speech_len = silence_total(speech)
    speech_hit = silence_overlap(speech, target)
    added = speech_hit / (speech_len + eps)

    return {
        "followed": followed,
        "added": added,
        "core_len": core_len,
        "core_hit": core_hit,
        "speech_len": speech_len,
        "speech_hit": speech_hit,
        "timeline_scale": scale,
        "timeline_scale_raw": raw_scale,
    }


# ---------------------------------------------------------------------------
# Pause-alignment reward — shared math + token/audio backends
# ---------------------------------------------------------------------------
# Default zero-based codec IDs for silence/room tone. Configure
# speech_silence_tokens (precision-first) and speech_occupancy_silence_tokens
# (recall-first) for the model in use; this fallback set is not a reliable
# substitute for audio-domain validation.
SPEECH_SILENCE_CODES: Tuple[int, ...] = (
    1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323,
)


def coerce_silence_intervals(
    intervals,
    min_silence_s: float,
    duration: Optional[float] = None,
    drop_edge: bool = True,
) -> List[Tuple[float, float]]:
    """Clamp/filter raw silence intervals exactly like eval/silence_iou.py.

    Accepts (start, end) pairs or {"start","end"} dicts; clamps to
    [0, duration]; drops head/tail silences touching an edge when
    ``drop_edge``; keeps only spans STRICTLY longer than ``min_silence_s``.
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


def codec_silence_intervals(
    speech_codes: List[int],
    silence_codes,
    min_silence_s: float,
    token_rate: float = 25.0,
    drop_edge: bool = True,
    bridge_gap_tokens: int = 0,
) -> List[Tuple[float, float]]:
    """Maximal runs of silence codec tokens, as generated-timeline intervals.

    A run over ordinals [i, j) maps to (i/token_rate, j/token_rate) seconds.
    Only runs STRICTLY longer than ``min_silence_s`` are kept and edge runs
    (touching ordinal 0 or the last token) are dropped when ``drop_edge`` —
    both rules mirror the coercion eval/silence_iou.py applies to VAD
    silences, so token-domain metrics stay definition-compatible with the
    audio-domain ones.

    ``bridge_gap_tokens`` merges runs separated by at most that many
    non-set tokens (the gap is absorbed into the merged run).  Silence in
    this codec is spread across many ids with a long tail, so any practical
    set under-covers pause interiors; a small bridge keeps one acoustic
    pause from fragmenting into sub-threshold pieces over a single missed
    code.  Keep it small (1-2 tokens = 40-80 ms): every bridged token is
    credited as silence without evidence.
    """
    if token_rate <= 0.0:
        raise ValueError("token_rate must be positive")
    if bridge_gap_tokens < 0:
        raise ValueError("bridge_gap_tokens must be non-negative")
    silence_set = {int(code) for code in silence_codes}
    n = len(speech_codes)
    runs: List[List[int]] = []
    run_start: Optional[int] = None
    for idx in range(n + 1):
        is_silence = idx < n and int(speech_codes[idx]) in silence_set
        if is_silence:
            if run_start is None:
                run_start = idx
            continue
        if run_start is None:
            continue
        runs.append([run_start, idx])
        run_start = None

    if bridge_gap_tokens > 0 and runs:
        merged: List[List[int]] = [runs[0]]
        for start, end in runs[1:]:
            if start - merged[-1][1] <= bridge_gap_tokens:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        runs = merged

    intervals: List[Tuple[float, float]] = []
    for start, end in runs:
        touches_edge = start == 0 or end == n
        # Length is derived from the integer run length so the strictly-
        # greater test is position-independent (start/end float subtraction
        # is not: 1.4 - 1.2 != 0.2 exactly in binary).
        run_s = (end - start) / token_rate
        if run_s > min_silence_s and not (drop_edge and touches_edge):
            intervals.append((start / token_rate, end / token_rate))
    return intervals


def compute_pause_alignment(
    source_silences: List[Tuple[float, float]],
    source_duration: float,
    generated_silences: List[Tuple[float, float]],
    generated_duration: float,
    *,
    occupancy_silences: Optional[List[Tuple[float, float]]] = None,
    strict_min_silence: float = 0.4,
    collar: float = 0.10,
    dense_weight: float = 0.25,
    added_silence_lambda: float = 0.3,
    added_silence_cap: float = 2.0,
    added_free_fraction: float = 0.05,
    silence_flood_ratio: float = 3.0,
    no_generated_penalty: float = 0.0,
    target_core_floor: float = 0.4,
    duration_gate_tolerance: float = 0.25,
    duration_gate_floor: float = -1.0,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Sequence-level pause-alignment reward shared by both backends.

    ``generated_silences`` are the DENSE internal pauses (everything above
    the backend's dense threshold, e.g. >200 ms).  The strict subset
    (>``strict_min_silence``, the external eval's 400 ms rule) is re-derived
    here so logged strict metrics stay comparable with eval/silence_iou.py.
    ``occupancy_silences`` is EVERY silence span regardless of length,
    including spans touching the utterance edges; it feeds only the added
    (anti-hack) term and defaults to ``generated_silences`` when a backend
    cannot provide it::

        followed = (1 - w_d) * followed_strict + w_d * followed_dense
        charged  = max(0, occupancy silence seconds inside the source speech
                          region - added_free_fraction * speech seconds)
        added    = charged / max(strict core seconds, target_core_floor)
        reward   = followed - added_silence_lambda * min(added, cap)
        if added > silence_flood_ratio:                    # silence flood
            reward = min(reward, duration_gate_floor)
        if |gen_duration / src_duration - 1| > duration_gate_tolerance:
            reward = min(reward, duration_gate_floor)

    ``added_free_fraction`` is a per-utterance allowance for the natural
    micro-pauses of fluent speech (sub-strict source pauses are not in the
    protected source set, so faithfully reproducing them would otherwise be
    charged).  It scales with the source SPEECH length, so long utterances
    are not flood-gated for ordinary fluency while blanket/fragmented
    silence still exceeds the budget by an order of magnitude.

    Scoring details:
      * A sequence-level scalar is used without per-event standardization
        or token-local credit masks.
      * ``followed_strict`` is the eval metric verbatim; the dense blend only
        adds sub-threshold shaping so a 200–400 ms pause at the right place
        outranks emitting no pause at all.
      * ``added`` counts ALL silence occupancy (sub-threshold fragments and
        edge padding included) so the guard cannot be evaded by chopping
        silence into short runs or parking it at the utterance edges, and it
        is normalized by the target pause core rather than the utterance's
        speech seconds, so scattering silence across a long utterance is not
        cheap.
      * Beyond ``silence_flood_ratio`` the candidate is HARD-floored: a
        capped linear penalty alone would leave blanket silence at
        ``1 - lambda*cap`` — a positive score that beats honest imperfect
        candidates whenever ``lambda*cap < 1``.  The flood threshold sits
        above the penalty cap so measurement noise around the cap changes
        the reward by at most the capped penalty, not by a flip to the
        floor (blanket/fragmented silence lands far above it anyway).
      * Inside ±tolerance, duration contributes zero gradient; outside it,
        the candidate is floored. ``tolerance=0`` floors any deviation.
    """
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be in [0, 1]")
    if min(
        added_silence_lambda, added_silence_cap, target_core_floor,
        duration_gate_tolerance, added_free_fraction,
    ) < 0.0:
        raise ValueError("pause penalty knobs must be non-negative")
    if silence_flood_ratio < added_silence_cap:
        raise ValueError(
            "silence_flood_ratio must be >= added_silence_cap (the flood "
            "gate backs up the capped penalty, it does not replace it)"
        )
    if no_generated_penalty < 0.0:
        raise ValueError("no_generated_penalty must be non-negative")

    strict_generated = [
        (float(start), float(end))
        for start, end in generated_silences
        if float(end) - float(start) > float(strict_min_silence)
    ]
    strict = collared_silence_metrics(
        source_silences, source_duration,
        strict_generated, generated_duration,
        collar, eps,
    )
    dense = collared_silence_metrics(
        source_silences, source_duration,
        generated_silences, generated_duration,
        collar, eps,
    )
    if occupancy_silences is None:
        occupancy_silences = generated_silences
    occupancy = collared_silence_metrics(
        source_silences, source_duration,
        occupancy_silences, generated_duration,
        collar, eps,
    )

    followed = (
        (1.0 - dense_weight) * strict["followed"]
        + dense_weight * dense["followed"]
    )
    charged_seconds = max(
        0.0,
        occupancy["speech_hit"]
        - float(added_free_fraction) * occupancy["speech_len"],
    )
    added_ratio = (
        charged_seconds
        / max(strict["core_len"], target_core_floor, eps)
    )
    added_balanced = min(added_ratio, float(added_silence_cap))
    reward = followed - float(added_silence_lambda) * added_balanced

    # Exploration shaping for hard (real-pause) data: when the source has
    # pauses but the candidate produced NO strict pause at all, subtract a
    # flat penalty.  In groups where nobody pauses at the right place this
    # still creates within-group variance between "tried to pause" and
    # "did not pause", pointing the policy toward emitting pauses; the
    # added/flood guards keep indiscriminate pausing unprofitable.
    no_generated = bool(source_silences) and not strict_generated
    if no_generated:
        reward -= float(no_generated_penalty)

    silence_flood_gated = added_ratio > float(silence_flood_ratio) + 1e-6
    if silence_flood_gated:
        reward = min(reward, float(duration_gate_floor))

    duration_ratio = (
        generated_duration / source_duration
        if source_duration > 0.0 and generated_duration > 0.0
        else 0.0
    )
    # Small numerical slack keeps an exactly-on-the-boundary ratio inside
    # the free band under fp32 arithmetic.
    duration_gated = (
        abs(duration_ratio - 1.0) > float(duration_gate_tolerance) + 1e-6
    )
    if duration_gated:
        reward = min(reward, float(duration_gate_floor))

    return {
        "reward": float(reward),
        "followed": float(followed),
        "followed_strict": float(strict["followed"]),
        "followed_dense": float(dense["followed"]),
        "added_strict": float(strict["added"]),
        "added_dense": float(dense["added"]),
        "added_ratio": float(added_ratio),
        "added_balanced": float(added_balanced),
        "added_charged_seconds": float(charged_seconds),
        "silence_flood_gated": 1.0 if silence_flood_gated else 0.0,
        "no_generated_penalized": 1.0 if (
            no_generated and no_generated_penalty > 0.0
        ) else 0.0,
        "strict_core_len": float(strict["core_len"]),
        "occupancy_speech_hit": float(occupancy["speech_hit"]),
        "duration_ratio": float(duration_ratio),
        "duration_gated": 1.0 if duration_gated else 0.0,
        "timeline_scale": float(strict["timeline_scale"]),
        "strict_generated_silences": strict_generated,
    }


def _pause_metric_record(
    reward: float,
    source_silences: List[Tuple[float, float]],
    source_duration: float,
    generated_duration: float,
    alignment: Optional[Dict[str, Any]] = None,
    dense_generated_silences: Optional[List[Tuple[float, float]]] = None,
    no_speech: bool = False,
    source_vad_detected: bool = False,
    silence_code_fraction: float = 0.0,
) -> Dict[str, float]:
    """One scalar-only metrics row (protocol shared by both backends)."""
    alignment = alignment or {}
    dense_generated_silences = dense_generated_silences or []
    strict_generated = alignment.get("strict_generated_silences", [])
    return {
        "pause_reward": float(reward),
        "pause_followed_silence": float(alignment.get("followed_strict", 0.0)),
        "pause_dense_followed_silence": float(
            alignment.get("followed_dense", 0.0)
        ),
        "pause_followed_shaped": float(alignment.get("followed", 0.0)),
        "pause_added_silence": float(alignment.get("added_strict", 0.0)),
        "pause_dense_added_silence": float(alignment.get("added_dense", 0.0)),
        "pause_added_balanced": float(alignment.get("added_balanced", 0.0)),
        "pause_added_ratio": float(alignment.get("added_ratio", 0.0)),
        "pause_silence_flood_gated": float(
            alignment.get("silence_flood_gated", 0.0)
        ),
        "pause_no_generated_penalized": float(
            alignment.get("no_generated_penalized", 0.0)
        ),
        "pause_source_silence_total": silence_total(source_silences),
        "pause_generated_silence_total": silence_total(strict_generated),
        "pause_dense_generated_silence_total": silence_total(
            dense_generated_silences
        ),
        "pause_source_duration": float(source_duration),
        "pause_generated_duration": float(generated_duration),
        "pause_duration_ratio": float(alignment.get("duration_ratio", 0.0)),
        "pause_duration_gated": float(alignment.get("duration_gated", 0.0)),
        "pause_timeline_scale": float(alignment.get("timeline_scale", 1.0)),
        "num_src_pauses": float(len(source_silences)),
        "num_gen_pauses": float(len(strict_generated)),
        "num_dense_gen_pauses": float(len(dense_generated_silences)),
        "pause_no_source": 1.0 if not source_silences else 0.0,
        "pause_no_generated": 1.0 if not strict_generated else 0.0,
        "pause_no_speech": 1.0 if no_speech else 0.0,
        "pause_source_vad_detected": 1.0 if source_vad_detected else 0.0,
        "pause_silence_code_fraction": float(silence_code_fraction),
    }


class SilenceVAD:
    """Silero-VAD internal-silence extraction (mirrors eval/silence_iou.py).

    Shared by both pause backends: the audio backend runs it on
    token2wav-decoded candidates, and both backends use it to recover a
    single-audio sample's real source pauses (cached per path — source files
    never change during a run, so each unique path is VAD-ed exactly once
    per rank; None results for unreadable/no-speech files are cached too).
    """

    def __init__(
        self,
        device: torch.device | str,
        sample_rate: int = 16000,
        threshold: float = 0.8,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
        drop_edge_silence: bool = True,
    ):
        self.device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.min_silence_ms = int(min_silence_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.drop_edge_silence = bool(drop_edge_silence)
        self._model = None
        self._get_speech_timestamps = None
        self._source_cache: Dict[
            Tuple[str, float],
            Optional[Tuple[List[Tuple[float, float]], float]],
        ] = {}

    def _load(self):
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pause reward requires the `silero-vad` package "
                "(import name `silero_vad`)."
            ) from exc
        self._model = load_silero_vad()
        self._model.eval()
        self._model = self._model.to(self.device)
        self._get_speech_timestamps = get_speech_timestamps
        logging.info(
            "Pause reward Silero VAD loaded on %s at %d Hz.",
            self.device, self.sample_rate,
        )

    def prepare_waveform(
        self, waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        """Mono float32 at the VAD sample rate, on the VAD device."""
        import torchaudio.functional as F

        if waveform.ndim == 2:
            wav = waveform.float().mean(dim=0)
        else:
            wav = waveform.float().view(-1)
        wav = wav.detach()
        if wav.device != self.device:
            wav = wav.to(self.device)
        if wav.numel() == 0:
            return wav
        if int(sample_rate) != self.sample_rate:
            wav = F.resample(
                wav, orig_freq=int(sample_rate), new_freq=self.sample_rate,
            )
        return wav

    def raw_silences(
        self,
        wav: torch.Tensor,
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        """VAD a mono ``sample_rate`` waveform into UN-coerced silences.

        Speech spans from Silero are complemented over [0, duration]; no
        length filtering or edge dropping is applied.  Returns None when the
        waveform is empty or contains no detectable speech.
        """
        self._load()
        if wav.numel() == 0:
            return None
        duration = wav.numel() / float(self.sample_rate)
        speech = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.sample_rate,
            return_seconds=True,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        if not speech:
            return None
        edges = (
            [0.0]
            + [t for seg in speech for t in (seg["start"], seg["end"])]
            + [duration]
        )
        silences = [
            (edges[i], edges[i + 1]) for i in range(0, len(edges), 2)
        ]
        return silences, duration

    def silences(
        self,
        wav: torch.Tensor,
        min_silence_s: float,
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        """Coerced internal silences (clamped, edge-dropped, > threshold)."""
        raw = self.raw_silences(wav)
        if raw is None:
            return None
        silences, duration = raw
        return (
            coerce_silence_intervals(
                silences,
                float(min_silence_s),
                duration=duration,
                drop_edge=self.drop_edge_silence,
            ),
            duration,
        )

    def source_silences(
        self,
        audio_path: str,
        min_silence_s: float,
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        """Cached internal-silence detection on a source audio file."""
        key = (audio_path, float(min_silence_s))
        if key in self._source_cache:
            return self._source_cache[key]
        result: Optional[Tuple[List[Tuple[float, float]], float]] = None
        try:
            import librosa

            wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            result = self.silences(
                torch.from_numpy(wav).to(self.device), float(min_silence_s),
            )
        except Exception:
            logging.warning(
                "pause reward: source VAD failed for %s; treating as no "
                "source pauses.", audio_path, exc_info=True,
            )
            result = None
        self._source_cache[key] = result
        return result


def _stack_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: [float(row.get(key, 0.0)) for row in rows] for key in keys}


class PauseTokenReward:
    """Deterministic pause reward computed directly on generated codec tokens.

    A pause is a maximal run of silence codec IDs at ``token_rate`` tokens/s.
    Direct token scoring avoids per-candidate audio rendering and VAD in the
    training loop. Audio-domain evaluation uses eval/silence_iou.py on
    rendered waveforms.

    Exposes the same ``score()`` / ``last_metrics`` protocol as
    :class:`PauseOnlyReward` so RewardServer can swap backends.
    """

    def __init__(
        self,
        *,
        silence_codes=SPEECH_SILENCE_CODES,
        occupancy_silence_codes=None,
        token_rate: float = 25.0,
        strict_min_silence_ms: int = 400,
        dense_min_silence_ms: int = 200,
        drop_edge_silence: bool = True,
        collar: float = 0.10,
        dense_weight: float = 0.25,
        added_silence_lambda: float = 0.3,
        added_silence_cap: float = 2.0,
        added_free_fraction: float = 0.05,
        silence_flood_ratio: float = 3.0,
        no_generated_penalty: float = 0.0,
        target_core_floor: float = 0.4,
        duration_gate_tolerance: float = 0.25,
        duration_gate_floor: float = -1.0,
        bridge_gap_tokens: int = 2,
        no_speech_reward: float = 0.0,
        source_vad: Optional[SilenceVAD] = None,
        source_vad_from_audio: bool = True,
        eps: float = 1e-6,
    ):
        if strict_min_silence_ms < dense_min_silence_ms:
            raise ValueError(
                "strict pause threshold cannot be below the dense threshold"
            )
        if min(strict_min_silence_ms, dense_min_silence_ms) < 0:
            raise ValueError("pause silence thresholds must be non-negative")
        self.silence_codes = tuple(int(code) for code in silence_codes)
        if not self.silence_codes:
            raise ValueError("token pause backend needs a non-empty silence-code set")
        # The anti-hack occupancy term uses a recall-first superset: counting
        # a borderline code as silence there is conservative, while counting
        # it in the credit (followed) set would be an inflation hack.  The
        # credit set is always included.
        self.occupancy_silence_codes = tuple(sorted(
            {int(code) for code in (occupancy_silence_codes or ())}
            | set(self.silence_codes)
        ))
        self.token_rate = float(token_rate)
        if self.token_rate <= 0.0:
            raise ValueError("token_rate must be positive")
        self.strict_min_silence = float(strict_min_silence_ms) / 1000.0
        self.dense_min_silence = float(dense_min_silence_ms) / 1000.0
        self.drop_edge_silence = bool(drop_edge_silence)
        self.collar = float(collar)
        self.dense_weight = float(dense_weight)
        self.added_silence_lambda = float(added_silence_lambda)
        self.added_silence_cap = float(added_silence_cap)
        self.added_free_fraction = float(added_free_fraction)
        self.silence_flood_ratio = float(silence_flood_ratio)
        self.no_generated_penalty = float(no_generated_penalty)
        self.target_core_floor = float(target_core_floor)
        self.duration_gate_tolerance = float(duration_gate_tolerance)
        self.duration_gate_floor = float(duration_gate_floor)
        self.bridge_gap_tokens = int(bridge_gap_tokens)
        self.no_speech_reward = float(no_speech_reward)
        self.source_vad = source_vad
        self.source_vad_from_audio = bool(source_vad_from_audio)
        if self.source_vad_from_audio and self.source_vad is None:
            raise ValueError(
                "source_vad_from_audio requires a SilenceVAD instance"
            )
        self.eps = float(eps)
        self.last_metrics: Dict[str, List[float]] = {}

    def score(
        self,
        audio_paths: List[str],
        source_durations: List[float],
        source_silences: List[List[Tuple[float, float]]],
        speech_codes: List[List[int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        assert len(audio_paths) == len(source_durations) == len(source_silences) == len(speech_codes), (
            f"pause reward needs aligned audio paths, source durations, source "
            f"silences and speech-code sequences, got {len(audio_paths)}, "
            f"{len(source_durations)}, {len(source_silences)}, {len(speech_codes)}"
        )
        if not audio_paths:
            self.last_metrics = {}
            scores = torch.zeros(0, dtype=torch.float32)
            return scores.to(device) if device is not None else scores

        silence_set = set(self.silence_codes)
        occupancy_set = set(self.occupancy_silence_codes)
        out: List[float] = []
        metric_rows: List[Dict[str, float]] = []
        for audio_path, source_duration, src_silences_raw, codes in zip(
            audio_paths, source_durations, source_silences, speech_codes,
        ):
            source_duration = float(source_duration)
            source_silences_clean = coerce_silence_intervals(
                src_silences_raw,
                self.strict_min_silence,
                duration=source_duration,
                drop_edge=self.drop_edge_silence,
            )

            # Single-audio sample (no inserted-silence ground truth): measure
            # the source's real pauses with the same VAD stack that external
            # eval uses, and trust the decoded duration over the parquet
            # length so collar/scale math shares one timeline.  Keyed on the
            # RAW intervals being absent — multi-audio mixes always ship >=1
            # gap and their audio_path is only the first segment.
            source_vad_detected = False
            if self.source_vad_from_audio and not src_silences_raw and audio_path:
                detected = self.source_vad.source_silences(
                    audio_path, self.strict_min_silence,
                )
                if detected is not None:
                    source_silences_clean = detected[0]
                    source_duration = float(detected[1])
                    source_vad_detected = True

            if not codes or not math.isfinite(source_duration) or source_duration <= 0.0:
                reward = self.no_speech_reward
                out.append(reward)
                metric_rows.append(_pause_metric_record(
                    reward, source_silences_clean, source_duration, 0.0,
                    no_speech=True, source_vad_detected=source_vad_detected,
                ))
                continue

            generated_duration = len(codes) / self.token_rate
            dense_gen_silences = codec_silence_intervals(
                codes,
                silence_set,
                self.dense_min_silence,
                token_rate=self.token_rate,
                drop_edge=self.drop_edge_silence,
                bridge_gap_tokens=self.bridge_gap_tokens,
            )
            # Every silence run, however short and including edge runs:
            # feeds only the anti-hack added term.
            occupancy_silences = codec_silence_intervals(
                codes,
                occupancy_set,
                0.0,
                token_rate=self.token_rate,
                drop_edge=False,
            )
            alignment = compute_pause_alignment(
                source_silences_clean,
                source_duration,
                dense_gen_silences,
                generated_duration,
                occupancy_silences=occupancy_silences,
                strict_min_silence=self.strict_min_silence,
                collar=self.collar,
                dense_weight=self.dense_weight,
                added_silence_lambda=self.added_silence_lambda,
                added_silence_cap=self.added_silence_cap,
                added_free_fraction=self.added_free_fraction,
                silence_flood_ratio=self.silence_flood_ratio,
                no_generated_penalty=self.no_generated_penalty,
                target_core_floor=self.target_core_floor,
                duration_gate_tolerance=self.duration_gate_tolerance,
                duration_gate_floor=self.duration_gate_floor,
                eps=self.eps,
            )
            reward = float(alignment["reward"])
            out.append(reward)
            silence_fraction = (
                sum(1 for code in codes if int(code) in silence_set)
                / len(codes)
            )
            metric_rows.append(_pause_metric_record(
                reward, source_silences_clean, source_duration,
                generated_duration, alignment=alignment,
                dense_generated_silences=dense_gen_silences,
                source_vad_detected=source_vad_detected,
                silence_code_fraction=silence_fraction,
            ))

        scores = torch.tensor(out, dtype=torch.float32)
        self.last_metrics = _stack_metric_rows(metric_rows)
        if device is not None:
            scores = scores.to(device)
        return scores


def _dilate_intervals(
    intervals: List[Tuple[float, float]], tau: float, lo: float, hi: float,
) -> List[Tuple[float, float]]:
    """Expand each interval by ``tau`` on both sides, clip to [lo, hi], merge."""
    return merge_intervals([
        (max(lo, s - tau), min(hi, e + tau)) for s, e in intervals
    ])


class TimelineReward:
    """Unified time-structure reward for isometric S2ST (token domain).

    Duration and pause alignment are scored together on the absolute
    speech/silence timeline, matching eval/timeline_metrics.py:

        r = -(w_miss * miss + w_fa * fa + w_overhang * |dT|) / T_src

      miss — source-silence core (eroded by ``collar``) that the candidate
             covers with speech: deleted or displaced pauses;
      fa   — candidate silence lying on confident source speech (source
             silences dilated by ``collar`` are exempt): spurious pauses;
      |dT| — duration mismatch: the isometric precondition, inside the same
             scalar so shrinking/stretching cannot be financed by silence.

    Deleting a pause raises ``miss`` one-for-one; stuffing a pause with
    speech raises ``miss`` too; padding with silence raises ``fa``; and any
    length change pays ``w_overhang`` — there is no cross-leg arbitrage left.

    Occupancy (recall-first) codes define candidate silence; runs are bridged
    like the pause backend.  Silences down to ``min_silence_ms`` participate
    (structure, not only strict pauses).  ``pause_followed_silence`` /
    ``pause_added_silence`` metrics are emitted as strict-event recall and
    false-alarm share, respectively.
    """

    def __init__(
        self,
        *,
        silence_codes=SPEECH_SILENCE_CODES,
        occupancy_silence_codes=None,
        token_rate: float = 25.0,
        min_silence_ms: int = 120,
        strict_min_silence_ms: int = 400,
        drop_edge_silence: bool = True,
        collar: float = 0.10,
        w_miss: float = 1.5,
        w_fa: float = 1.0,
        w_overhang: float = 2.0,
        w_drift: float = 0.0,
        drift_tol: float = 3.0,
        no_generated_penalty: float = 0.3,
        reward_floor: float = -4.0,
        bridge_gap_tokens: int = 2,
        no_speech_reward: float = 0.0,
        source_vad: Optional[SilenceVAD] = None,
        source_vad_from_audio: bool = True,
        eps: float = 1e-6,
    ):
        self.silence_codes = tuple(int(c) for c in silence_codes)
        self.occupancy_silence_codes = tuple(sorted(
            {int(c) for c in (occupancy_silence_codes or ())}
            | set(self.silence_codes)
        ))
        self.token_rate = float(token_rate)
        self.min_silence = float(min_silence_ms) / 1000.0
        self.strict_min_silence = float(strict_min_silence_ms) / 1000.0
        self.drop_edge_silence = bool(drop_edge_silence)
        self.collar = float(collar)
        self.w_miss = float(w_miss)
        self.w_fa = float(w_fa)
        self.w_overhang = float(w_overhang)
        self.w_drift = float(w_drift)
        self.drift_tol = float(drift_tol)
        self.no_generated_penalty = float(no_generated_penalty)
        self.reward_floor = float(reward_floor)
        self.bridge_gap_tokens = int(bridge_gap_tokens)
        self.no_speech_reward = float(no_speech_reward)
        self.source_vad = source_vad
        self.source_vad_from_audio = bool(source_vad_from_audio)
        if self.source_vad_from_audio and self.source_vad is None:
            raise ValueError("source_vad_from_audio requires a SilenceVAD instance")
        self.eps = float(eps)
        self.last_metrics: Dict[str, List[float]] = {}

    def _load_model(self):  # parity with the pause backends
        return None

    def score(
        self,
        audio_paths: List[str],
        source_durations: List[float],
        source_silences: List[List[Tuple[float, float]]],
        speech_codes: List[List[int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        assert len(audio_paths) == len(source_durations) == len(source_silences) == len(speech_codes)
        if not audio_paths:
            self.last_metrics = {}
            scores = torch.zeros(0, dtype=torch.float32)
            return scores.to(device) if device is not None else scores

        occupancy_set = set(self.occupancy_silence_codes)
        out: List[float] = []
        metric_rows: List[Dict[str, float]] = []
        for audio_path, source_duration, src_silences_raw, codes in zip(
            audio_paths, source_durations, source_silences, speech_codes,
        ):
            source_duration = float(source_duration)
            src_sil = coerce_silence_intervals(
                src_silences_raw, self.min_silence,
                duration=source_duration, drop_edge=self.drop_edge_silence,
            )
            # Real (single-audio) samples ship no inserted-silence intervals:
            # measure the source with the shared VAD stack (same as the pause
            # backends) and trust the decoded duration.
            if self.source_vad_from_audio and not src_silences_raw and audio_path:
                detected = self.source_vad.source_silences(
                    audio_path, self.min_silence,
                )
                if detected is not None:
                    src_sil = detected[0]
                    source_duration = float(detected[1])

            if not math.isfinite(source_duration) or source_duration <= 0.0:
                reward = self.no_speech_reward
                out.append(reward)
                metric_rows.append({
                    "timeline_reward": reward, "timeline_miss": 0.0,
                    "timeline_fa": 0.0, "timeline_overhang": 0.0,
                    "timeline_drift": 0.0,
                    "pause_followed_silence": 0.0, "pause_added_silence": 0.0,
                    "num_strict_gen_pauses": 0.0, "num_strict_src_pauses": 0.0,
                })
                continue

            t_gen = len(codes) / self.token_rate
            gen_sil = codec_silence_intervals(
                codes, occupancy_set,
                min_silence_s=self.min_silence,
                token_rate=self.token_rate,
                drop_edge=self.drop_edge_silence,
                bridge_gap_tokens=self.bridge_gap_tokens,
            )
            src_sil = merge_intervals(src_sil)
            gen_sil = merge_intervals(gen_sil)

            common = min(source_duration, t_gen)
            src_c = [
                (max(0.0, s), min(e, common)) for s, e in src_sil
                if min(e, common) > max(0.0, s)
            ]
            gen_c = [
                (max(0.0, s), min(e, common)) for s, e in gen_sil
                if min(e, common) > max(0.0, s)
            ]

            src_core = shrink_intervals(src_c, self.collar)
            miss = max(
                0.0, silence_total(src_core) - silence_overlap(src_core, gen_c)
            )
            src_dilated = _dilate_intervals(src_c, self.collar, 0.0, common)
            fa = max(
                0.0, silence_total(gen_c) - silence_overlap(gen_c, src_dilated)
            )
            overhang = abs(t_gen - source_duration)

            penalty = (
                self.w_miss * miss + self.w_fa * fa
                + self.w_overhang * overhang
            ) / max(source_duration, self.eps)

            # Monitor strict-event recall and false-alarm share.
            src_ev = [iv for iv in src_sil if iv[1] - iv[0] >= self.strict_min_silence]
            gen_ev = [iv for iv in gen_sil if iv[1] - iv[0] >= self.strict_min_silence]
            matched = 0
            used: set = set()
            tol = self.collar + 0.2
            for ss, se in src_ev:
                sc = 0.5 * (ss + se)
                best_j, best_d = -1, float("inf")
                for j, (gs, ge) in enumerate(gen_ev):
                    if j in used:
                        continue
                    d = abs(0.5 * (gs + ge) - sc)
                    if d < best_d:
                        best_j, best_d = j, d
                if best_j >= 0 and best_d <= tol:
                    used.add(best_j)
                    matched += 1
            # No-source-pause samples score 0, matching the followed_strict
            # convention of the pause backends (keeps dev curves comparable).
            followed = matched / len(src_ev) if src_ev else 0.0

            # Match each source pause to the nearest unused generated pause
            # within drift_tol and penalize its center offset. Unmatched
            # source pauses receive the full tolerance penalty.
            drift_pen = 0.0
            if src_ev and self.w_drift > 0.0:
                used_d: set = set()
                drift_sum = 0.0
                for ss, se in src_ev:
                    sc = 0.5 * (ss + se)
                    best_j, best_d = -1, float("inf")
                    for j, (gs, ge) in enumerate(gen_ev):
                        if j in used_d:
                            continue
                        d = abs(0.5 * (gs + ge) - sc)
                        if d < best_d:
                            best_j, best_d = j, d
                    if best_j >= 0 and best_d <= self.drift_tol:
                        used_d.add(best_j)
                        drift_sum += best_d
                    else:
                        drift_sum += self.drift_tol
                drift_pen = drift_sum / max(source_duration, self.eps)
                penalty += self.w_drift * drift_pen

            # Penalize missing all strict pauses on a pause-bearing source.
            if src_ev and not gen_ev:
                penalty += self.no_generated_penalty
            reward = max(-penalty, self.reward_floor)

            out.append(reward)
            metric_rows.append({
                "timeline_reward": reward,
                "timeline_miss": miss / max(source_duration, self.eps),
                "timeline_fa": fa / max(source_duration, self.eps),
                "timeline_overhang": overhang / max(source_duration, self.eps),
                "timeline_drift": drift_pen,
                "pause_followed_silence": followed,
                "pause_added_silence": fa / max(source_duration, self.eps),
                "num_strict_gen_pauses": float(len(gen_ev)),
                "num_strict_src_pauses": float(len(src_ev)),
            })

        scores = torch.tensor(out, dtype=torch.float32)
        self.last_metrics = _stack_metric_rows(metric_rows)
        if device is not None:
            scores = scores.to(device)
        return scores


class PauseOnlyReward:
    """Audio-domain pause backend: token2wav-decode candidates, VAD, score.

    Selected with ``pause_backend="audio"`` (pause_only or combined) and used
    by audits comparing the token backend against rendered audio.  Scoring
    math is identical to :class:`PauseTokenReward`
    (``compute_pause_alignment``); only the generated-silence extraction
    differs (Silero VAD on the cheap-proxy rendering instead of codec
    silence-token runs).
    """

    def __init__(
        self,
        flow_model: str,
        hift_model: str,
        campplus_model: str,
        speech_tokenizer_model: str,
        device: torch.device | str,
        fp16: bool = False,
        sample_rate: int = 24000,
        vad_sample_rate: int = 16000,
        vad_threshold: float = 0.8,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
        strict_min_silence_ms: int = 400,
        dense_min_silence_ms: int = 200,
        drop_edge_silence: bool = True,
        no_speech_reward: float = 0.0,
        collar: float = 0.10,
        dense_weight: float = 0.25,
        added_silence_lambda: float = 0.3,
        added_silence_cap: float = 2.0,
        added_free_fraction: float = 0.05,
        silence_flood_ratio: float = 3.0,
        no_generated_penalty: float = 0.0,
        target_core_floor: float = 0.4,
        duration_gate_tolerance: float = 0.25,
        duration_gate_floor: float = -1.0,
        token2wav_n_timesteps: int = 2,
        token2wav_cfg: bool = False,
        eps: float = 1e-6,
        source_vad_from_audio: bool = True,
        vad: Optional[SilenceVAD] = None,
    ):
        if strict_min_silence_ms < dense_min_silence_ms:
            raise ValueError(
                "strict pause threshold cannot be below the dense threshold"
            )
        if min(strict_min_silence_ms, dense_min_silence_ms) < 0:
            raise ValueError("pause silence thresholds must be non-negative")
        if token2wav_n_timesteps <= 0:
            raise ValueError("token2wav_n_timesteps must be positive")
        self.flow_model = flow_model
        self.hift_model = hift_model
        self.campplus_model = campplus_model
        self.speech_tokenizer_model = speech_tokenizer_model
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.sample_rate = int(sample_rate)
        self.strict_min_silence = float(strict_min_silence_ms) / 1000.0
        self.dense_min_silence = float(dense_min_silence_ms) / 1000.0
        self.drop_edge_silence = bool(drop_edge_silence)
        self.no_speech_reward = float(no_speech_reward)
        self.collar = float(collar)
        self.dense_weight = float(dense_weight)
        self.added_silence_lambda = float(added_silence_lambda)
        self.added_silence_cap = float(added_silence_cap)
        self.added_free_fraction = float(added_free_fraction)
        self.silence_flood_ratio = float(silence_flood_ratio)
        self.no_generated_penalty = float(no_generated_penalty)
        self.target_core_floor = float(target_core_floor)
        self.duration_gate_tolerance = float(duration_gate_tolerance)
        self.duration_gate_floor = float(duration_gate_floor)
        self.token2wav_n_timesteps = int(token2wav_n_timesteps)
        self.token2wav_cfg = bool(token2wav_cfg)
        self.eps = float(eps)
        self.source_vad_from_audio = bool(source_vad_from_audio)
        self.vad = vad if vad is not None else SilenceVAD(
            device=self.device,
            sample_rate=vad_sample_rate,
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            drop_edge_silence=drop_edge_silence,
        )

        self._model = None
        self._frontend = None
        self._prepare_prompt_conditioning = None
        self._token2wav = None
        self.last_metrics: Dict[str, List[float]] = {}

    def _load_token2wav_model(self):
        if self._model is not None:
            return
        from models import build_model, prepare_prompt_conditioning, token2wav

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._model, self._frontend = build_model(
            flow_model=self.flow_model,
            hift_model=self.hift_model,
            campplus_model=self.campplus_model,
            speech_tokenizer_model=self.speech_tokenizer_model,
            fp16=self.fp16,
        )
        self._model.to(self.device)
        self._prepare_prompt_conditioning = prepare_prompt_conditioning
        self._token2wav = token2wav
        logging.info("PauseOnly token2wav model loaded on %s.", self.device)

    def _load_model(self):
        self._load_token2wav_model()
        self.vad._load()

    def _prompt_cache_entry(
        self,
        audio_path: str,
        prompt_cache: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        entry = prompt_cache.get(audio_path)
        if entry is not None:
            return entry

        speaker_embedding, prompt_speech_token, prompt_mel_features = (
            self._prepare_prompt_conditioning(self._frontend, audio_path)
        )
        # Move to compute device once; flow.inference re-issues `.to(device)`
        # internally but it's a no-op when already on-device.
        speaker_embedding = speaker_embedding.to(self.device)
        prompt_speech_token = prompt_speech_token.to(self.device)
        prompt_mel_features = prompt_mel_features.to(self.device)

        # Precompute the prompt-only flow prefix that doesn't depend on the
        # generated codes (shared across all group candidates for this audio):
        #   - speaker embedding after F.normalize + spk_embed_affine_layer
        #   - input_embedding lookup over prompt tokens
        # autocast must match Token2Wav.token2wav so fp16 weights see fp16
        # activations here, otherwise spk_embed_affine_layer hits a dtype
        # mismatch on the fp32 normalized speaker embedding.
        flow = self._model.flow
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.fp16):
            embedding_proj = flow.spk_embed_affine_layer(
                torch.nn.functional.normalize(speaker_embedding, dim=1)
            )
            prompt_token_emb = flow.input_embedding(
                torch.clamp(prompt_speech_token.to(dtype=torch.long), min=0)
            )

        entry = {
            "speaker_embedding": speaker_embedding,
            "prompt_speech_token": prompt_speech_token,
            "prompt_mel_features": prompt_mel_features,
            "embedding_proj": embedding_proj,
            "prompt_token_emb": prompt_token_emb,
        }
        prompt_cache[audio_path] = entry
        return entry

    def _decode_waveform(
        self,
        audio_path: str,
        speech_codes: List[int],
        prompt_cache: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not speech_codes:
            return torch.zeros(1, 0, dtype=torch.float32, device=self.device)
        self._load_token2wav_model()
        entry = self._prompt_cache_entry(audio_path, prompt_cache)

        token = torch.tensor(speech_codes, dtype=torch.long, device=self.device).unsqueeze(0)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        with torch.inference_mode():
            wav = self._token2wav(
                self._model,
                token,
                prompt_conditioning=(
                    entry["speaker_embedding"],
                    entry["prompt_speech_token"],
                    entry["prompt_mel_features"],
                ),
                n_timesteps=self.token2wav_n_timesteps,
                cfg=self.token2wav_cfg,
                prompt_token_emb=entry["prompt_token_emb"],
                embedding_proj=entry["embedding_proj"],
            )
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav.detach().to(dtype=torch.float32)

    @torch.inference_mode()
    def decode_waveforms(
        self,
        audio_paths: List[str],
        speech_codes: List[List[int]],
    ) -> List[torch.Tensor]:
        assert len(audio_paths) == len(speech_codes), (
            f"pause_only decoder needs aligned audio paths and speech-code "
            f"sequences, got {len(audio_paths)} vs {len(speech_codes)}"
        )
        if not audio_paths:
            return []

        self._load_token2wav_model()
        prompt_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        return [
            self._decode_waveform(audio_path, codes, prompt_cache)
            for audio_path, codes in zip(audio_paths, speech_codes)
        ]

    @torch.inference_mode()
    def score(
        self,
        audio_paths: List[str],
        source_durations: List[float],
        source_silences: List[List[Tuple[float, float]]],
        speech_codes: List[List[int]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        assert len(audio_paths) == len(source_durations) == len(source_silences) == len(speech_codes), (
            f"pause_only needs aligned audio paths, source durations, source "
            f"silences and speech-code sequences, got {len(audio_paths)}, "
            f"{len(source_durations)}, {len(source_silences)}, {len(speech_codes)}"
        )
        if not audio_paths:
            self.last_metrics = {}
            scores = torch.zeros(0, dtype=torch.float32)
            return scores.to(device) if device is not None else scores

        self._load_model()
        prompt_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        out: List[float] = []
        metric_rows: List[Dict[str, float]] = []

        for audio_path, source_duration, src_silences_raw, codes in zip(
            audio_paths, source_durations, source_silences, speech_codes,
        ):
            source_duration = float(source_duration)
            source_silences_clean = coerce_silence_intervals(
                src_silences_raw,
                self.strict_min_silence,
                duration=source_duration,
                drop_edge=self.drop_edge_silence,
            )

            source_vad_detected = False
            if self.source_vad_from_audio and not src_silences_raw and audio_path:
                detected = self.vad.source_silences(
                    audio_path, self.strict_min_silence,
                )
                if detected is not None:
                    source_silences_clean = detected[0]
                    source_duration = float(detected[1])
                    source_vad_detected = True

            if not codes or not math.isfinite(source_duration) or source_duration <= 0.0:
                reward = self.no_speech_reward
                out.append(reward)
                metric_rows.append(_pause_metric_record(
                    reward, source_silences_clean, source_duration, 0.0,
                    no_speech=True, source_vad_detected=source_vad_detected,
                ))
                continue

            gen_wav = self._decode_waveform(audio_path, codes, prompt_cache)
            gen_result = self.vad.raw_silences(
                self.vad.prepare_waveform(gen_wav, self.sample_rate),
            )
            if gen_result is None:
                reward = self.no_speech_reward
                out.append(reward)
                metric_rows.append(_pause_metric_record(
                    reward, source_silences_clean, source_duration, 0.0,
                    no_speech=True, source_vad_detected=source_vad_detected,
                ))
                continue

            raw_gen_silences, gen_duration = gen_result
            dense_gen_silences = coerce_silence_intervals(
                raw_gen_silences,
                self.dense_min_silence,
                duration=gen_duration,
                drop_edge=self.drop_edge_silence,
            )
            # Every silence span, however short and including edge spans:
            # feeds only the anti-hack added term.
            occupancy_silences = coerce_silence_intervals(
                raw_gen_silences,
                0.0,
                duration=gen_duration,
                drop_edge=False,
            )
            alignment = compute_pause_alignment(
                source_silences_clean,
                source_duration,
                dense_gen_silences,
                gen_duration,
                occupancy_silences=occupancy_silences,
                strict_min_silence=self.strict_min_silence,
                collar=self.collar,
                dense_weight=self.dense_weight,
                added_silence_lambda=self.added_silence_lambda,
                added_silence_cap=self.added_silence_cap,
                added_free_fraction=self.added_free_fraction,
                silence_flood_ratio=self.silence_flood_ratio,
                no_generated_penalty=self.no_generated_penalty,
                target_core_floor=self.target_core_floor,
                duration_gate_tolerance=self.duration_gate_tolerance,
                duration_gate_floor=self.duration_gate_floor,
                eps=self.eps,
            )
            reward = float(alignment["reward"])
            out.append(reward)
            metric_rows.append(_pause_metric_record(
                reward, source_silences_clean, source_duration, gen_duration,
                alignment=alignment,
                dense_generated_silences=dense_gen_silences,
                source_vad_detected=source_vad_detected,
            ))

        scores = torch.tensor(out, dtype=torch.float32)
        self.last_metrics = _stack_metric_rows(metric_rows)
        if device is not None:
            scores = scores.to(device)
        return scores


# ---------------------------------------------------------------------------
# Centralized reward server (rank 0 holds the reward models)
# ---------------------------------------------------------------------------
class RewardServer:
    """Centralized reward computation over a torch.distributed world group.

    Rank 0 loads all reward models and idles in `serve_forever()`. Training
    ranks call `request()` once per training step; rank 0 batches every
    rank's payload, runs the reward model(s) once, and scatters scores back.

    All ranks (including rank 0) must call collectives in lockstep, so every
    training rank invokes `request()` exactly once per logical step — even
    when it has no samples (it just sends empty lists).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        reward_type: str,
        speech_token_rate: float = 25.0,
        duration_eps_free: float = 0.05,
        duration_speech_only: bool = False,
        duration_speech_ratio_center: float = 1.0,
        reward_device: str = "cuda:0",
        metricx_model_dir: str = "",
        metricx_repo: str = "",
        metricx_tokenizer: str = "google/mt5-xl",
        metricx_max_input_length: int = 1536,
        metricx_max_score: float = 25.0,
        metricx_batch_size: int = 16,
        token2wav_flow_model: str = "",
        token2wav_hift_model: str = "",
        token2wav_campplus_model: str = "",
        token2wav_speech_tokenizer_model: str = "",
        token2wav_fp16: bool = False,
        token2wav_n_timesteps: int = 2,
        token2wav_cfg: bool = False,
        pause_backend: str = "token",
        pause_silence_codes=SPEECH_SILENCE_CODES,
        pause_occupancy_silence_codes=None,
        timeline_w_miss: float = 1.5,
        timeline_w_fa: float = 1.0,
        timeline_w_overhang: float = 2.0,
        timeline_w_drift: float = 0.0,
        timeline_drift_tol: float = 3.0,
        pause_vad_sample_rate: int = 16000,
        pause_vad_threshold: float = 0.8,
        pause_min_speech_ms: int = 250,
        pause_min_silence_ms: int = 100,
        pause_speech_pad_ms: int = 30,
        pause_strict_min_silence_ms: int = 400,
        pause_dense_min_silence_ms: int = 200,
        pause_drop_edge_silence: bool = True,
        pause_no_speech_reward: float = 0.0,
        pause_collar: float = 0.10,
        pause_dense_weight: float = 0.25,
        pause_added_silence_lambda: float = 0.3,
        pause_added_silence_cap: float = 2.0,
        pause_added_free_fraction: float = 0.05,
        pause_silence_flood_ratio: float = 3.0,
        pause_bridge_gap_tokens: int = 2,
        pause_no_generated_penalty: float = 0.0,
        pause_target_core_floor: float = 0.4,
        pause_duration_gate_tolerance: float = 0.25,
        pause_duration_gate_floor: float = -1.0,
        pause_eps: float = 1e-6,
        pause_source_vad_from_audio: bool = True,
        quality_use_reference: bool = True,
        extreme_filter: bool = False,
        keep_top_k: int = 4,
        keep_bottom_k: int = 4,
    ):
        self.rank = rank
        self.world_size = world_size
        self.reward_type = reward_type
        self.speech_token_rate = speech_token_rate
        self.duration_eps_free = float(duration_eps_free)
        self.duration_speech_only = bool(duration_speech_only)
        self.duration_speech_ratio_center = float(duration_speech_ratio_center)
        self.reward_device = torch.device(reward_device)
        self.metricx_max_score = metricx_max_score
        self.quality_use_reference = bool(quality_use_reference)
        self.extreme_filter = bool(extreme_filter)
        self.keep_top_k = int(keep_top_k)
        self.keep_bottom_k = int(keep_bottom_k)
        self.last_reward_metrics: Dict[str, List[float]] = {}
        if pause_backend not in ("token", "audio", "timeline"):
            raise ValueError(
                f"pause_backend must be 'token', 'audio' or 'timeline', "
                f"got {pause_backend!r}"
            )
        self.pause_backend = pause_backend

        # Single cheap/local rewards train on every rank.  ``combined`` is a
        # hybrid mode: MetricX lives on the dedicated rank 0, while pause and
        # duration are computed on each training rank.
        self.local_mode = reward_type in ("isometric", "pause_only")

        self._metricx: Optional["MetricXReward"] = None
        self._pause_scorer = None

        if rank == 0 and not self.local_mode:
            if self.reward_device.type == "cuda":
                torch.cuda.set_device(self.reward_device)
            if reward_type in ("translation_quality", "combined"):
                assert metricx_model_dir, "metricx_model_dir required"
                assert metricx_repo, "metricx_repo required"
                self._metricx = MetricXReward(
                    model_dir=metricx_model_dir,
                    metricx_repo=metricx_repo,
                    tokenizer_name=metricx_tokenizer,
                    max_input_length=metricx_max_input_length,
                    is_qe=not self.quality_use_reference,
                    batch_size=metricx_batch_size,
                    device=self.reward_device,
                )
                self._metricx._load_model()

        needs_local_pause = (
            reward_type == "pause_only"
            or (reward_type == "combined" and rank > 0)
        )
        if needs_local_pause:
            if self.reward_device.type == "cuda":
                torch.cuda.set_device(self.reward_device)
            source_vad = SilenceVAD(
                device=self.reward_device,
                sample_rate=pause_vad_sample_rate,
                threshold=pause_vad_threshold,
                min_speech_ms=pause_min_speech_ms,
                min_silence_ms=pause_min_silence_ms,
                speech_pad_ms=pause_speech_pad_ms,
                drop_edge_silence=pause_drop_edge_silence,
            )
            if self.pause_backend == "timeline":
                self._pause_scorer = TimelineReward(
                    silence_codes=pause_silence_codes,
                    occupancy_silence_codes=pause_occupancy_silence_codes,
                    token_rate=speech_token_rate,
                    min_silence_ms=pause_dense_min_silence_ms,
                    strict_min_silence_ms=pause_strict_min_silence_ms,
                    drop_edge_silence=pause_drop_edge_silence,
                    collar=pause_collar,
                    w_miss=timeline_w_miss,
                    w_fa=timeline_w_fa,
                    w_overhang=timeline_w_overhang,
                    w_drift=timeline_w_drift,
                    drift_tol=timeline_drift_tol,
                    no_generated_penalty=pause_no_generated_penalty,
                    bridge_gap_tokens=pause_bridge_gap_tokens,
                    no_speech_reward=pause_no_speech_reward,
                    source_vad=source_vad,
                    source_vad_from_audio=pause_source_vad_from_audio,
                    eps=pause_eps,
                )
            elif self.pause_backend == "token":
                self._pause_scorer = PauseTokenReward(
                    silence_codes=pause_silence_codes,
                    occupancy_silence_codes=pause_occupancy_silence_codes,
                    bridge_gap_tokens=pause_bridge_gap_tokens,
                    token_rate=speech_token_rate,
                    strict_min_silence_ms=pause_strict_min_silence_ms,
                    dense_min_silence_ms=pause_dense_min_silence_ms,
                    drop_edge_silence=pause_drop_edge_silence,
                    collar=pause_collar,
                    dense_weight=pause_dense_weight,
                    added_silence_lambda=pause_added_silence_lambda,
                    added_silence_cap=pause_added_silence_cap,
                    added_free_fraction=pause_added_free_fraction,
                    silence_flood_ratio=pause_silence_flood_ratio,
                    no_generated_penalty=pause_no_generated_penalty,
                    target_core_floor=pause_target_core_floor,
                    duration_gate_tolerance=pause_duration_gate_tolerance,
                    duration_gate_floor=pause_duration_gate_floor,
                    no_speech_reward=pause_no_speech_reward,
                    source_vad=source_vad,
                    source_vad_from_audio=pause_source_vad_from_audio,
                    eps=pause_eps,
                )
            else:
                assert token2wav_flow_model, "token2wav_flow_model required"
                assert token2wav_hift_model, "token2wav_hift_model required"
                assert token2wav_campplus_model, "token2wav_campplus_model required"
                assert token2wav_speech_tokenizer_model, (
                    "token2wav_speech_tokenizer_model required"
                )
                self._pause_scorer = PauseOnlyReward(
                    flow_model=token2wav_flow_model,
                    hift_model=token2wav_hift_model,
                    campplus_model=token2wav_campplus_model,
                    speech_tokenizer_model=token2wav_speech_tokenizer_model,
                    fp16=token2wav_fp16,
                    token2wav_n_timesteps=token2wav_n_timesteps,
                    token2wav_cfg=token2wav_cfg,
                    device=self.reward_device,
                    strict_min_silence_ms=pause_strict_min_silence_ms,
                    dense_min_silence_ms=pause_dense_min_silence_ms,
                    drop_edge_silence=pause_drop_edge_silence,
                    no_speech_reward=pause_no_speech_reward,
                    collar=pause_collar,
                    dense_weight=pause_dense_weight,
                    added_silence_lambda=pause_added_silence_lambda,
                    added_silence_cap=pause_added_silence_cap,
                    added_free_fraction=pause_added_free_fraction,
                    silence_flood_ratio=pause_silence_flood_ratio,
                    no_generated_penalty=pause_no_generated_penalty,
                    target_core_floor=pause_target_core_floor,
                    duration_gate_tolerance=pause_duration_gate_tolerance,
                    duration_gate_floor=pause_duration_gate_floor,
                    eps=pause_eps,
                    source_vad_from_audio=pause_source_vad_from_audio,
                    vad=source_vad,
                )
                self._pause_scorer._load_model()

    def _maybe_filter_extremes(
        self,
        rewards: torch.Tensor,
        group_sizes: Optional[List[int]],
    ) -> torch.Tensor:
        """If extreme_filter is on, mark non-extreme candidates as NaN per group.

        The trainer-side finite-mask filter in gspo_train.py drops the NaN
        rows before the policy forward, so only the top/bottom-K survive.
        """
        # A combined run must select rows from all three component scores,
        # not from MetricX alone on rank 0.  The trainer performs that filter.
        if self.reward_type == "combined":
            return rewards
        if not self.extreme_filter or rewards.numel() == 0:
            return rewards
        gs = _checked_group_sizes(rewards.numel(), group_sizes)
        keep = compute_extreme_mask(
            rewards, gs, top_k=self.keep_top_k, bottom_k=self.keep_bottom_k,
        )
        rewards = rewards.clone()
        rewards[~keep] = float("nan")
        return rewards

    # ----- client side (training ranks) -----
    def request(
        self,
        source_texts: List[str],
        trans_texts: List[str],
        src_lens: List[float],
        speech_lens: List[int],
        device: torch.device,
        reference_texts: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        source_silences: Optional[List[List[Tuple[float, float]]]] = None,
        speech_codes: Optional[List[List[int]]] = None,
        group_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Send a scoring request and return rewards for *this rank* on `device`.

        `source_texts` is the source-side reference text (e.g. dataset
        transcript) used by translation-quality reward models — NOT the
        model's own ASR output, so ASR errors do not leak into the reward.
        `trans_texts` is the model-generated translation hypothesis.
        `reference_texts` is the human reference translation in the target
        language; required by reference-based metrics (MetricX) and ignored
        by QE / source-only ones (SeedX).
        """
        self.last_reward_metrics = {}
        combined_pause: Optional[torch.Tensor] = None
        combined_duration: Optional[torch.Tensor] = None
        combined_duration_metrics: Dict[str, List[float]] = {}

        if self.reward_type == "combined":
            # Duration is cheap and local. Pause is also local, but is launched
            # just after gather below so it overlaps rank 0's MetricX forward.
            source_length_tensor = torch.tensor(src_lens, dtype=torch.float32)
            speech_length_tensor = torch.tensor(speech_lens, dtype=torch.float32)
            reward_source_lengths = source_length_tensor
            reward_speech_lengths = speech_length_tensor
            if (
                self.duration_speech_only
                and speech_codes
                and source_silences
                and len(speech_codes) == len(src_lens)
                and len(source_silences) == len(src_lens)
                and self._pause_scorer is not None
            ):
                # Score the non-silent portion of each side. The pause reward
                # controls silence duration; this term controls speech rate.
                occ = set(int(c) for c in self._pause_scorer.occupancy_silence_codes)
                # Center the speech-only ratio to account for systematic
                # offsets between codec occupancy and source-side VAD.
                center = max(self.duration_speech_ratio_center, 1e-3)
                gen_speech_tokens = [
                    float(sum(1 for t in codes if int(t) not in occ)) / center
                    for codes in speech_codes
                ]
                src_speech_secs = []
                for total_s, sils in zip(src_lens, source_silences):
                    sil_secs = sum(
                        max(0.0, float(e) - float(s))
                        for s, e in coerce_silence_intervals(
                            sils, 0.0, drop_edge=False,
                        )
                    )
                    src_speech_secs.append(
                        max(float(total_s) - sil_secs, 0.2)
                    )
                reward_source_lengths = torch.tensor(
                    src_speech_secs, dtype=torch.float32,
                )
                reward_speech_lengths = torch.tensor(
                    gen_speech_tokens, dtype=torch.float32,
                )
            combined_duration = compute_isometric_reward(
                reward_source_lengths,
                reward_speech_lengths,
                source_frame_rate=1.0,
                target_token_rate=self.speech_token_rate,
                eps_free=self.duration_eps_free,
            ).float()
            # Logged within-band metrics keep the WHOLE-duration definition so
            # dev dur10/5/2 stay comparable across runs and match the OOD eval.
            combined_duration_metrics = compute_duration_within_metrics(
                source_length_tensor,
                speech_length_tensor,
                source_frame_rate=1.0,
                target_token_rate=self.speech_token_rate,
            )
        if self.local_mode:
            rt = self.reward_type
            if rt == "isometric":
                if not src_lens:
                    return torch.zeros(0, dtype=torch.float32, device=device)
                source_length_tensor = torch.tensor(src_lens, dtype=torch.float32)
                speech_length_tensor = torch.tensor(speech_lens, dtype=torch.float32)
                rewards = compute_isometric_reward(
                    source_length_tensor,
                    speech_length_tensor,
                    source_frame_rate=1.0,
                    target_token_rate=self.speech_token_rate,
                    eps_free=self.duration_eps_free,
                ).float()
                self.last_reward_metrics = compute_duration_within_metrics(
                    source_length_tensor,
                    speech_length_tensor,
                    source_frame_rate=1.0,
                    target_token_rate=self.speech_token_rate,
                )
            elif rt == "pause_only":
                assert self._pause_scorer is not None, (
                    "local-mode pause_only reward not initialised"
                )
                if not audio_paths:
                    return torch.zeros(0, dtype=torch.float32, device=device)
                rewards = self._pause_scorer.score(
                    list(audio_paths),
                    [float(x) for x in (src_lens or [])],
                    list(source_silences or []),
                    list(speech_codes or []),
                ).float()
                self.last_reward_metrics = {
                    key: list(values)
                    for key, values in self._pause_scorer.last_metrics.items()
                }
            else:
                raise ValueError(
                    f"local_mode does not support reward_type={rt!r}"
                )
            rewards = self._maybe_filter_extremes(rewards, group_sizes)
            return rewards.to(device)
        assert self.rank > 0, "request() is for training ranks only"
        if reference_texts is None:
            reference_texts = [""] * len(source_texts)
        if audio_paths is None:
            audio_paths = []
        if source_silences is None:
            source_silences = []
        if speech_codes is None:
            speech_codes = []

        def _serialise_silences(intervals):
            out = []
            for start, end in coerce_silence_intervals(
                intervals, 0.0, drop_edge=False,
            ):
                out.append([float(start), float(end)])
            return out

        payload = {
            "cmd": "score",
            "source_texts": list(source_texts),
            "trans_texts": list(trans_texts),
            "reference_texts": list(reference_texts),
            "src_lens": [float(x) for x in src_lens],
            "speech_lens": [int(x) for x in speech_lens],
            "audio_paths": list(audio_paths),
            "source_silences": [_serialise_silences(x) for x in source_silences],
            "speech_codes": [[int(t) for t in codes] for codes in speech_codes],
            "group_sizes": [int(g) for g in (group_sizes or [])],
        }
        dist.gather_object(payload, None, dst=0)
        if self.reward_type == "combined":
            # Rank 0 starts MetricX as soon as gather completes.  Training
            # ranks decode token2wav here before entering scatter, overlapping
            # the two expensive reward paths without extra threads/processes.
            if src_lens:
                assert self._pause_scorer is not None, (
                    "combined reward requires a local pause scorer on training ranks"
                )
                combined_pause = self._pause_scorer.score(
                    list(audio_paths),
                    [float(x) for x in src_lens],
                    list(source_silences),
                    list(speech_codes),
                ).float()
                self.last_reward_metrics = {
                    key: list(values)
                    for key, values in self._pause_scorer.last_metrics.items()
                }
            else:
                combined_pause = torch.zeros(0, dtype=torch.float32)
        out: List = [None]
        dist.scatter_object_list(out, None, src=0)
        rewards = out[0]
        if rewards is None or len(rewards) == 0:
            remote_rewards = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            remote_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        if self.reward_type == "combined":
            assert combined_pause is not None and combined_duration is not None
            pause = combined_pause.to(device)
            duration = combined_duration.to(device)
            self.last_reward_metrics["translation_quality_reward"] = (
                remote_rewards.detach().float().cpu().tolist()
            )
            self.last_reward_metrics["duration_reward"] = (
                duration.detach().float().cpu().tolist()
            )
            self.last_reward_metrics.update(combined_duration_metrics)
            return {
                "translation_quality": remote_rewards,
                "pause": pause,
                "duration": duration,
            }
        return remote_rewards

    def get_last_reward_metrics(self) -> Dict[str, List[float]]:
        return {
            key: list(values)
            for key, values in self.last_reward_metrics.items()
        }

    def shutdown(self):
        """Tell rank 0 to break out of serve_forever()."""
        if self.local_mode:
            return
        assert self.rank > 0
        dist.gather_object({"cmd": "stop"}, None, dst=0)
        out: List = [None]
        dist.scatter_object_list(out, None, src=0)

    # ----- server side (rank 0) -----
    def serve_forever(self):
        """Loop until any training rank sends `cmd: stop`."""
        assert self.rank == 0
        while True:
            gathered: List = [None] * self.world_size
            dist.gather_object(None, gathered, dst=0)

            if any(p is not None and p.get("cmd") == "stop" for p in gathered):
                replies: List = [None] * self.world_size
                out: List = [None]
                dist.scatter_object_list(out, replies, src=0)
                break

            counts = [0] * self.world_size
            sources, trans, refs, src, speech = [], [], [], [], []
            audio_paths, source_silences, speech_codes = [], [], []
            all_group_sizes: List[int] = []
            for r in range(1, self.world_size):
                p = gathered[r]
                if not p:
                    continue
                n = max(
                    len(p.get("source_texts", [])),
                    len(p.get("src_lens", [])),
                    len(p.get("speech_lens", [])),
                    len(p.get("audio_paths", [])),
                    len(p.get("source_silences", [])),
                    len(p.get("speech_codes", [])),
                )
                counts[r] = n
                sources.extend(p.get("source_texts") or [""] * n)
                trans.extend(p.get("trans_texts") or [""] * n)
                refs.extend(p.get("reference_texts") or [""] * n)
                src.extend(p.get("src_lens") or [0.0] * n)
                speech.extend(p.get("speech_lens") or [0] * n)
                audio_paths.extend(p.get("audio_paths") or [""] * n)
                source_silences.extend(p.get("source_silences") or [[] for _ in range(n)])
                speech_codes.extend(p.get("speech_codes") or [[] for _ in range(n)])
                all_group_sizes.extend(p.get("group_sizes", []))

            total = sum(counts)
            if total > 0:
                rewards = self._score(
                    sources, trans, refs, src, speech,
                    audio_paths, source_silences, speech_codes, all_group_sizes,
                )
                rewards = self._maybe_filter_extremes(rewards, all_group_sizes)
            else:
                rewards = torch.zeros(0)

            replies = [None] * self.world_size
            offset = 0
            for r in range(1, self.world_size):
                c = counts[r]
                replies[r] = rewards[offset:offset + c].cpu().tolist()
                offset += c

            out = [None]
            dist.scatter_object_list(out, replies, src=0)

    def _score(
        self,
        sources: List[str],
        trans: List[str],
        refs: List[str],
        src_lens: List[float],
        speech_lens: List[int],
        audio_paths: Optional[List[str]] = None,
        source_silences: Optional[List[List[Tuple[float, float]]]] = None,
        speech_codes: Optional[List[List[int]]] = None,
        group_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        rt = self.reward_type
        cpu = torch.device("cpu")

        def translation_quality_reward() -> torch.Tensor:
            """MetricX-only quality reward, normalised to higher = better.

            Raw MetricX is lower-is-better in [0, max_score]. We map to
            `(max_score - raw) / max_score ∈ [0, 1]` and rely on GSPO's
            per-group mean/std to handle in-batch comparison. With
            `quality_use_reference=False` the underlying MetricX is loaded
            in QE mode (no reference text).
            """
            if not sources:
                return torch.zeros(0, dtype=torch.float32)
            raw = self._metricx.score(sources, trans, refs, device=cpu).float()
            return (self.metricx_max_score - raw) / self.metricx_max_score

        if rt in ("translation_quality", "combined"):
            return translation_quality_reward()
        if rt == "pause_only":
            assert self._pause_scorer is not None, "pause_only reward is not loaded"
            return self._pause_scorer.score(
                audio_paths or [],
                src_lens,
                source_silences or [],
                speech_codes or [],
                device=cpu,
            )
        raise ValueError(f"Unknown reward_type: {rt}")


# ---------------------------------------------------------------------------
# COMET reward (translation quality, reference-based when refs are provided)
# ---------------------------------------------------------------------------
class COMETReward:
    """Wrap COMET/COMETKiwi scoring.

    Reference-based COMET models use `{src, mt, ref}`.  QE-style COMETKiwi
    models also accept the same dicts but ignore `ref`; callers may omit
    references for backward compatibility with older DPO code.
    """

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 32,
        gpus: int = 1,
        device: torch.device | str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self._device = torch.device(device) if device is not None else None
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from comet import download_model, load_from_checkpoint

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        model_path = download_model(self.model_name)
        self._model = load_from_checkpoint(model_path)
        self._model.eval()
        logging.info("COMET model loaded: %s on %s", self.model_name, self._device)

    @torch.inference_mode()
    def score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: Optional[List[str]] = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        self._load_model()
        assert len(hypotheses) == len(sources), (
            f"COMET needs one hypothesis per source, got "
            f"{len(hypotheses)} vs {len(sources)}"
        )
        if references is None:
            references = [""] * len(sources)
        assert len(references) == len(sources), (
            f"COMET needs one reference per source, got "
            f"{len(references)} vs {len(sources)}"
        )
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        if not data:
            scores = torch.zeros(0, dtype=torch.float32)
        else:
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            predict_kwargs = {
                "batch_size": self.batch_size,
                "gpus": self.gpus,
                "progress_bar": False,
            }
            if self._device is not None and self._device.type == "cuda" and self.gpus > 0:
                predict_kwargs["devices"] = [self._device.index or 0]
            with _temporary_logger_level(
                [
                    "pytorch_lightning",
                    "pytorch_lightning.utilities.rank_zero",
                    "pytorch_lightning.accelerators.cuda",
                    "lightning",
                    "lightning.pytorch",
                ],
                logging.ERROR,
            ):
                output = self._model.predict(data, **predict_kwargs)
            scores = torch.tensor(output.scores, dtype=torch.float32)
        if device is not None:
            scores = scores.to(device)
        return scores


# ---------------------------------------------------------------------------
# Seed-X-RM reward (translation quality via reward model)
# ---------------------------------------------------------------------------
class SeedXReward:
    """Wraps the Seed-X-RM reward model for translation quality scoring.

    The model is loaded lazily on first call and cached for reuse.
    """

    def __init__(
        self,
        model_dir: str,
        batch_size: int = 32,
        device: torch.device | str | None = None,
    ):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self._device_override = torch.device(device) if device is not None else None
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
            from safetensors.torch import load_file

            config = AutoConfig.from_pretrained(self.model_dir)
            config._attn_implementation = "flash_attention_2"
            self._device = self._device_override or torch.device("cuda")
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            self._model = MistralForCausalLM(config)
            self._model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            except Exception:
                from transformers import PreTrainedTokenizerFast

                tokenizer_file = os.path.join(self.model_dir, "tokenizer.json")
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            state_dict = load_file(f"{self.model_dir}/model.safetensors")
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(dtype=torch.bfloat16)
            self._model.to(device=self._device)
            self._model.eval()
            logging.info("Seed-X-RM model loaded on %s.", self._device)

    @torch.inference_mode()
    def score(
        self,
        prompts: List[str],
        hypotheses: List[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute Seed-X-RM scores.

        Args:
            prompts: source / prompt texts
            hypotheses: generated translations
            device: device to place the returned tensor on

        Returns:
            Tensor of shape [N] with per-sample scores.
        """
        self._load_model()
        tokenizer = self._tokenizer

        # Wrap source texts in the Seed-X-RM prompt template
        prompts = [
            f"Translate the following English sentence into Chinese:\n{p} <zh>"
            for p in prompts
        ]

        all_scores: List[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_hyps = hypotheses[i : i + self.batch_size]

            input_ids_list = [
                tokenizer.encode(p) + tokenizer.encode(h) + [tokenizer.eos_token_id]
                for p, h in zip(batch_prompts, batch_hyps)
            ]
            max_length = max(len(ids) for ids in input_ids_list)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            padded = [
                ids + [pad_id] * (max_length - len(ids)) for ids in input_ids_list
            ]

            input_tensor = torch.tensor(padded, device=self._device)
            logits = self._model(input_tensor).logits

            for j, ids in enumerate(input_ids_list):
                eos_position = ids.index(tokenizer.eos_token_id)
                all_scores.append(logits[j, eos_position, :].squeeze().item())

        scores = torch.tensor(all_scores, dtype=torch.float32)
        if device is not None:
            scores = scores.to(device)
        return scores


# ---------------------------------------------------------------------------
# MetricX-24-Hybrid reward (reference-based or QE translation quality)
# ---------------------------------------------------------------------------
class MetricXReward:
    """Wraps MetricX-24-Hybrid for reference-based translation quality scoring.

    Returns the *raw* MetricX score (lower = better, range [0, 25]); the
    RewardServer is responsible for flipping/normalising the sign so RL
    sees higher = better.

    `metricx_repo` must point at the upstream MetricX checkout that ships
    `metricx24/models.py` — its custom MT5ForRegression is needed to load
    the regression head. `use_cache=False` avoids EncoderDecoderCache shape
    incompatibilities during scoring.
    """

    def __init__(
        self,
        model_dir: str,
        metricx_repo: str,
        tokenizer_name: str = "google/mt5-xl",
        max_input_length: int = 1536,
        is_qe: bool = False,
        batch_size: int = 16,
        device: torch.device | str | None = None,
    ):
        self.model_dir = model_dir
        self.metricx_repo = metricx_repo
        self.tokenizer_name = tokenizer_name
        self.max_input_length = max_input_length
        self.is_qe = is_qe
        self.batch_size = max(1, int(batch_size))
        self._device_override = torch.device(device) if device is not None else None
        self._model = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.metricx_repo and self.metricx_repo not in sys.path:
            sys.path.insert(0, self.metricx_repo)
        from metricx24 import models as _mx_models  # noqa: WPS433
        from transformers import AutoTokenizer

        self._device = self._device_override or torch.device("cuda")
        if self._device.type == "cuda":
            if self._device.index is None:
                self._device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._model = _mx_models.MT5ForRegression.from_pretrained(
            self.model_dir, torch_dtype="auto",
        )
        self._model.config.use_cache = False
        self._model.to(self._device).eval()
        logging.info("MetricX-24 model loaded on %s.", self._device)

    def _build_input(self, source: str, hypothesis: str, reference: str = "") -> str:
        if self.is_qe:
            return f"source: {source} candidate: {hypothesis}"
        return f"source: {source} candidate: {hypothesis} reference: {reference}"

    @torch.inference_mode()
    def score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: Optional[List[str]] = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Batched MetricX scores with padding-aware attention masks.

        `references` is required in reference-based mode (`is_qe=False`) and
        ignored in QE mode.
        """
        self._load_model()
        if not sources:
            scores = torch.zeros(0, dtype=torch.float32)
            if device is not None:
                scores = scores.to(device)
            return scores

        if references is None:
            references = [""] * len(sources)
        if not self.is_qe:
            assert len(references) == len(sources), (
                f"reference-based MetricX needs one reference per source, "
                f"got {len(references)} vs {len(sources)}"
            )

        texts = [
            self._build_input(s, h, r)
            for s, h, r in zip(sources, hypotheses, references)
        ]

        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i: i + self.batch_size]
            enc = self._tokenizer(
                chunk,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)
            # MetricX strips the trailing EOS. With padding the
            # EOS sits at the last attended position of each row; mask it out.
            # If a row hit max_length the tokeniser already dropped the EOS,
            # so we mask the last content token instead — same behaviour as
            # the original `[:, :-1]` slice for a non-padded single example.
            seq_lens = attention_mask.sum(dim=-1)
            last_idx = (seq_lens - 1).clamp(min=0)
            b_idx = torch.arange(input_ids.shape[0], device=self._device)
            attention_mask[b_idx, last_idx] = 0
            r = self._model(input_ids=input_ids, attention_mask=attention_mask)
            preds = r.predictions
            if preds.ndim > 1:
                preds = preds.view(-1)
            out.extend(preds.detach().to(torch.float32).cpu().tolist())

        scores = torch.tensor(out, dtype=torch.float32)
        if device is not None:
            scores = scores.to(device)
        return scores
