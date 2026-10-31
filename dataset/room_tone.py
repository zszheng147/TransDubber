import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 16000
DEFAULT_NOISE_DBFS_MIN = -85.0
DEFAULT_NOISE_DBFS_MAX = -55.0
DEFAULT_NOISE_FLOOR_PERCENTILE = 1.0
DEFAULT_NOISE_FRAME_MS = 20.0
DEFAULT_NOISE_HOP_MS = 10.0
DEFAULT_NOISE_JITTER_DB = 3.0
DEFAULT_BOUNDARY_XFADE_MS = 20.0
DEFAULT_MIN_TILE_LEN_MS = 60.0
DEFAULT_TILE_XFADE_MS = 30.0
DEFAULT_QUIET_MARGIN_DB = 6.0
DEFAULT_TRANSIENT_DB = 12.0

# Built-in pause-ambience spectrum.
# The template defines the relative spectral shape; ambient_dbfs sets its level.
AMBIENT_TEMPLATE_FREQS = np.array(
    [30, 40, 60, 80, 100, 130, 180, 240, 320, 430, 580, 770,
     1040, 1390, 1860, 2490, 3330, 4460, 5980, 8000],
    dtype=np.float64,
)
AMBIENT_TEMPLATE_DB = np.array(
    [-0.0, -0.6, -1.9, -4.8, -7.3, -8.6, -9.5, -11.5, -13.4, -15.4,
     -21.1, -21.6, -23.2, -18.0, -20.5, -20.0, -20.2, -15.8, -24.2, -42.3],
    dtype=np.float64,
)
DEFAULT_AMBIENT_DBFS = -64.0
DEFAULT_AMBIENT_JITTER_DB = 2.0
DEFAULT_AMBIENT_RAMP_MS = 100.0

# Probabilistic mouth events for the ``legacy`` pause style, drawn from banks
# of real samples. The ``clean`` style omits them.
# Point breath_bank_dir at a directory with
# breath/click/exhale wavs to enable them; a missing directory means no events.
DEFAULT_BREATH_DIR = Path(
    os.environ.get(
        "breath_bank_dir",
        Path(__file__).resolve().parent.parent / "assets" / "breath-bank",
    )
)
DEFAULT_BREATH_PROB = 0.5
BREATH_GAIN_ABOVE_AMBIENT_DB = (10.0, 20.0)
BREATH_MAX_DBFS = -34.0
BREATH_END_GAP_MS = (50.0, 200.0)
MIN_PAUSE_FOR_BREATH_S = 0.45
DEFAULT_CLICK_PROB = 0.35
CLICK_GAIN_ABOVE_AMBIENT_DB = (12.0, 22.0)
CLICK_MAX_DBFS = -32.0
DEFAULT_EXHALE_PROB = 0.12
EXHALE_GAIN_ABOVE_AMBIENT_DB = (8.0, 16.0)
EXHALE_MAX_DBFS = -38.0


@dataclass(frozen=True)
class RoomToneConfig:
    frame_ms: float = DEFAULT_NOISE_FRAME_MS
    hop_ms: float = DEFAULT_NOISE_HOP_MS
    floor_percentile: float = DEFAULT_NOISE_FLOOR_PERCENTILE
    quiet_margin_db: float = DEFAULT_QUIET_MARGIN_DB
    # below this p95-floor dynamic range the whole clip is loud content and
    # its "quiet" frames are just soft speech, not room tone
    min_speech_gap_db: float = 15.0
    guard_ms: float = 20.0
    min_tile_len_ms: float = DEFAULT_MIN_TILE_LEN_MS
    # segments at least this long are tiled directly (sparse repetition, real
    # texture); shorter ones drive spectral synthesis instead -- tiling a
    # <200ms segment repeats it so often that its micro-structure shows up as
    # horizontal stripes on a spectrogram
    full_tile_min_ms: float = 200.0
    tile_xfade_ms: float = DEFAULT_TILE_XFADE_MS
    tile_jitter_db: float = 1.0
    transient_db: float = DEFAULT_TRANSIENT_DB
    matched_n_fft: int = 512
    matched_hop: int = 160
    matched_min_quiet_frames: int = 5
    matched_level_jitter_db: float = 2.0
    matched_dbfs_min: float = -85.0
    matched_dbfs_max: float = -40.0
    tpdf_dbfs_min: float = DEFAULT_NOISE_DBFS_MIN
    tpdf_dbfs_max: float = DEFAULT_NOISE_DBFS_MAX
    tpdf_jitter_db: float = DEFAULT_NOISE_JITTER_DB
    highpass_hz: float = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert generated pseudo-silence between two wavs and save the combined audio."
    )
    parser.add_argument("wav1", type=Path, help="First input audio path.")
    parser.add_argument("wav2", type=Path, help="Second input audio path.")
    parser.add_argument("output", type=Path, help="Output combined audio path.")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--min-silence", type=float, default=0.4)
    parser.add_argument("--max-silence", type=float, default=1.0)
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=None,
        help="Use this exact silence duration instead of sampling one.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--noise-type",
        choices=("room_tone", "tpdf", "pink", "zero"),
        default="room_tone",
        help=(
            "room_tone extracts/synthesizes the clip's own noise floor and "
            "crossfades it into the speech; tpdf/pink/zero use "
            "generated-noise paths with fade-to-zero edges."
        ),
    )
    parser.add_argument(
        "--boundary-xfade-ms",
        type=float,
        default=DEFAULT_BOUNDARY_XFADE_MS,
        help="Equal-power speech<->pause crossfade for room_tone mode.",
    )
    parser.add_argument(
        "--min-tile-len-ms",
        type=float,
        default=DEFAULT_MIN_TILE_LEN_MS,
        help="Minimum contiguous quiet segment usable as tiled room tone.",
    )
    parser.add_argument(
        "--tile-xfade-ms",
        type=float,
        default=DEFAULT_TILE_XFADE_MS,
        help="Crossfade between tiled room-tone chunks.",
    )
    parser.add_argument(
        "--quiet-margin-db",
        type=float,
        default=DEFAULT_QUIET_MARGIN_DB,
        help="Frames within this margin above the noise floor count as quiet.",
    )
    parser.add_argument(
        "--transient-db",
        type=float,
        default=DEFAULT_TRANSIENT_DB,
        help="Frames louder than region median + this are rejected as clicks.",
    )
    parser.add_argument(
        "--no-pause-ambient",
        action="store_true",
        help=(
            "Disable the broadband ambience overlaid on room_tone pauses. "
            "Without it, pauses in denoised corpora stay near-black above "
            "1 kHz on a spectrogram."
        ),
    )
    parser.add_argument(
        "--ambient-ref",
        type=Path,
        default=None,
        help=(
            "Reference wav whose quiet frames define the pause-ambience "
            "spectrum and level (e.g. a natural, non-denoised recording). "
            "Default: built-in ambience template."
        ),
    )
    parser.add_argument(
        "--ambient-dbfs",
        type=float,
        default=None,
        help=(
            "Override the pause-ambience RMS level in dBFS "
            "(default: reference/template level with +/- jitter)."
        ),
    )
    parser.add_argument(
        "--ambient-ramp-ms",
        type=float,
        default=DEFAULT_AMBIENT_RAMP_MS,
        help="Ambience fade-in/out length, extending under the speech edges.",
    )
    parser.add_argument(
        "--breath-dir",
        type=Path,
        default=None,
        help=(
            "Directory of breath wavs (see build_breath_bank.py). Default: "
            f"{DEFAULT_BREATH_DIR} if it exists."
        ),
    )
    parser.add_argument(
        "--breath-prob",
        type=float,
        default=DEFAULT_BREATH_PROB,
        help="Probability of overlaying a pre-speech inhale on each pause.",
    )
    parser.add_argument(
        "--click-prob",
        type=float,
        default=DEFAULT_CLICK_PROB,
        help="Probability of a mouth click (before the inhale / speech onset).",
    )
    parser.add_argument(
        "--exhale-prob",
        type=float,
        default=DEFAULT_EXHALE_PROB,
        help="Probability of an exhale right after speech stops.",
    )
    parser.add_argument(
        "--no-breath",
        action="store_true",
        help="Disable all mouth events (inhale/click/exhale).",
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=10.0,
        help="Fade applied to the generated silence edges (tpdf/pink/zero only).",
    )
    parser.add_argument(
        "--noise-dbfs-min",
        type=float,
        default=DEFAULT_NOISE_DBFS_MIN,
        help=(
            "Minimum pseudo-silence RMS level in dBFS. In adaptive mode this "
            "is the lower clamp; in uniform mode this is the sampling minimum."
        ),
    )
    parser.add_argument(
        "--noise-dbfs-max",
        type=float,
        default=DEFAULT_NOISE_DBFS_MAX,
        help=(
            "Maximum pseudo-silence RMS level in dBFS. In adaptive mode this "
            "is the upper clamp; in uniform mode this is the sampling maximum."
        ),
    )
    parser.add_argument(
        "--noise-mode",
        choices=("adaptive", "uniform"),
        default="adaptive",
        help=(
            "How to choose the pseudo-silence RMS (tpdf/pink/zero). Adaptive "
            "estimates a low-energy floor from wav1/wav2; uniform samples directly."
        ),
    )
    parser.add_argument(
        "--noise-floor-percentile",
        type=float,
        default=DEFAULT_NOISE_FLOOR_PERCENTILE,
        help="Low-energy frame percentile used for noise-floor estimation.",
    )
    parser.add_argument(
        "--noise-frame-ms",
        type=float,
        default=DEFAULT_NOISE_FRAME_MS,
        help="Frame size in ms for noise-floor estimation.",
    )
    parser.add_argument(
        "--noise-hop-ms",
        type=float,
        default=DEFAULT_NOISE_HOP_MS,
        help="Hop size in ms for noise-floor estimation.",
    )
    parser.add_argument(
        "--noise-jitter-db",
        type=float,
        default=DEFAULT_NOISE_JITTER_DB,
        help="Uniform +/- jitter in dB around the adaptive noise-floor estimate.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to save duration and output metadata.",
    )
    return parser.parse_args()


def dbfs_to_rms(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def rms_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 20.0 * float(np.log10(eps))
    return 20.0 * float(np.log10(np.sqrt(np.mean(x**2) + eps)))


def normalize_rms(x: np.ndarray, target_dbfs: float, eps: float = 1e-12) -> np.ndarray:
    target_rms = dbfs_to_rms(target_dbfs)
    current_rms = np.sqrt(np.mean(x**2) + eps)
    return x * (target_rms / current_rms)


def apply_fade_edges(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    y = x.copy()
    fade_len = int(sr * fade_ms / 1000.0)
    fade_len = min(fade_len, len(y) // 2)

    if fade_len <= 0:
        return y

    t = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
    fade_in = 0.5 - 0.5 * np.cos(np.pi * t)
    fade_out = fade_in[::-1]
    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y


def estimate_noise_floor_dbfs(
    audio: torch.Tensor,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    percentile: float,
    eps: float = 1e-12,
) -> float:
    if audio.numel() == 0:
        return 20.0 * np.log10(eps)

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    if audio.numel() < frame_len:
        frames = audio.reshape(1, -1)
    else:
        frames = audio.unfold(0, frame_len, hop_len)

    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + eps)
    dbfs = 20.0 * torch.log10(torch.clamp(rms, min=eps))
    finite_dbfs = dbfs[torch.isfinite(dbfs)]
    if finite_dbfs.numel() == 0:
        return 20.0 * np.log10(eps)

    return float(np.percentile(finite_dbfs.cpu().numpy(), percentile))


def choose_noise_dbfs(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    sr: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | str | None]]:
    if args.noise_mode == "uniform":
        noise_dbfs = float(rng.uniform(args.noise_dbfs_min, args.noise_dbfs_max))
        return noise_dbfs, {
            "noise_mode": "uniform",
            "noise_floor_dbfs_wav1": None,
            "noise_floor_dbfs_wav2": None,
            "noise_floor_dbfs_mean": None,
            "noise_dbfs_unclamped": noise_dbfs,
            "noise_jitter_db_applied": None,
        }

    noise_floor_dbfs_wav1 = estimate_noise_floor_dbfs(
        audio1,
        sr=sr,
        frame_ms=args.noise_frame_ms,
        hop_ms=args.noise_hop_ms,
        percentile=args.noise_floor_percentile,
    )
    noise_floor_dbfs_wav2 = estimate_noise_floor_dbfs(
        audio2,
        sr=sr,
        frame_ms=args.noise_frame_ms,
        hop_ms=args.noise_hop_ms,
        percentile=args.noise_floor_percentile,
    )
    noise_floor_dbfs_mean = float(
        (noise_floor_dbfs_wav1 + noise_floor_dbfs_wav2) / 2.0
    )
    jitter_db = float(rng.uniform(-args.noise_jitter_db, args.noise_jitter_db))
    noise_dbfs_unclamped = noise_floor_dbfs_mean + jitter_db
    noise_dbfs = float(
        np.clip(noise_dbfs_unclamped, args.noise_dbfs_min, args.noise_dbfs_max)
    )

    return noise_dbfs, {
        "noise_mode": "adaptive",
        "noise_floor_dbfs_wav1": noise_floor_dbfs_wav1,
        "noise_floor_dbfs_wav2": noise_floor_dbfs_wav2,
        "noise_floor_dbfs_mean": noise_floor_dbfs_mean,
        "noise_dbfs_unclamped": noise_dbfs_unclamped,
        "noise_jitter_db_applied": jitter_db,
    }


def generate_tpdf_pause(
    duration_sec: float,
    sr: int,
    rng: np.random.Generator,
    fade_ms: float,
    noise_dbfs: float,
) -> tuple[torch.Tensor, float | None]:
    num_samples = int(round(duration_sec * sr))
    if num_samples <= 0:
        return torch.empty(0, dtype=torch.float32), None

    u1 = rng.random(num_samples).astype(np.float32) - 0.5
    u2 = rng.random(num_samples).astype(np.float32) - 0.5
    noise = normalize_rms(u1 + u2, noise_dbfs)
    noise = apply_fade_edges(noise.astype(np.float32), sr, fade_ms)
    return torch.from_numpy(noise.astype(np.float32)), noise_dbfs


def generate_pinkish_noise(
    num_samples: int,
    sr: int,
    noise_dbfs: float,
    rng: np.random.Generator,
    low_freq: float = 60.0,
    high_freq: float | None = None,
) -> np.ndarray:
    """Very simple frequency-domain-shaped pink-ish noise (1-D)."""
    if high_freq is None:
        high_freq = min(8000.0, sr * 0.45)

    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sr)
    scale = np.zeros_like(freqs, dtype=np.float32)
    valid = (freqs >= low_freq) & (freqs <= high_freq)
    scale[valid] = 1.0 / np.sqrt(np.maximum(freqs[valid], 1.0))
    scale[0] = 0.0

    white = rng.normal(0.0, 1.0, num_samples).astype(np.float32)
    shaped = np.fft.irfft(np.fft.rfft(white) * scale, n=num_samples)
    return normalize_rms(shaped, noise_dbfs).astype(np.float32)


# ---------------------------------------------------------------------------
# Room-tone core (numpy, 1-D float32 mono)
# ---------------------------------------------------------------------------


def frame_rms_dbfs(
    x: np.ndarray,
    sr: int,
    frame_ms: float = DEFAULT_NOISE_FRAME_MS,
    hop_ms: float = DEFAULT_NOISE_HOP_MS,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-frame RMS level in dBFS of a 1-D waveform (tail samples dropped)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.full(1, 20.0 * np.log10(eps))
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size < frame_len:
        frames = x[None, :]
    else:
        n_frames = 1 + (x.size - frame_len) // hop_len
        idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
        frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + eps)
    return 20.0 * np.log10(np.maximum(rms, eps))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """(first_idx, last_idx) of the longest run of True, or None."""
    best: tuple[int, int] | None = None
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if best is None or j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    return best


def find_quiet_regions(
    x: np.ndarray, sr: int, cfg: RoomToneConfig
) -> list[tuple[int, int]]:
    """Contiguous quiet sample ranges [start, end), longest/quietest first."""
    x = np.asarray(x, dtype=np.float32)
    dbfs = frame_rms_dbfs(x, sr, cfg.frame_ms, cfg.hop_ms)
    floor = float(np.percentile(dbfs, cfg.floor_percentile))
    if float(np.percentile(dbfs, 95.0)) - floor < cfg.min_speech_gap_db:
        return []

    frame_len = max(1, int(round(sr * cfg.frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * cfg.hop_ms / 1000.0)))
    guard = int(round(sr * cfg.guard_ms / 1000.0))
    quiet = dbfs <= floor + cfg.quiet_margin_db

    regions: list[tuple[int, int]] = []
    n = quiet.size
    i = 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and quiet[j + 1]:
            j += 1
        start = i * hop_len
        end = min(j * hop_len + frame_len, x.size)
        # trim sides that border speech frames (breath/coarticulation leakage);
        # clip edges were already VAD-trimmed upstream and need no guard
        if i > 0:
            start += guard
        if j < n - 1:
            end -= guard
        if end > start:
            regions.append((start, end))
        i = j + 1

    def sort_key(region: tuple[int, int]) -> tuple[int, float]:
        s, e = region
        return (-(e - s), rms_dbfs(x[s:e]))

    regions.sort(key=sort_key)
    return regions


def extract_room_tone(
    x: np.ndarray, sr: int, cfg: RoomToneConfig, prefer: str | None = None
) -> tuple[np.ndarray, dict] | None:
    """Best clean quiet segment (DC-removed), or None if nothing usable.

    prefer="end"/"start" picks the usable region closest to that side of the
    clip (for junction continuity when splicing two clips); None keeps the
    longest/quietest-first order.
    """
    x = np.asarray(x, dtype=np.float32)
    min_len = int(round(sr * cfg.min_tile_len_ms / 1000.0))
    frame_len = max(1, int(round(sr * cfg.frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * cfg.hop_ms / 1000.0)))

    candidates = [
        r for r in find_quiet_regions(x, sr, cfg) if r[1] - r[0] >= min_len
    ]
    if prefer == "end":
        candidates.sort(key=lambda r: x.size - r[1])
    elif prefer == "start":
        candidates.sort(key=lambda r: r[0])

    for start, end in candidates:
        seg = x[start:end]
        # reject transient frames (clicks, lip smacks), keep longest clean run
        seg_dbfs = frame_rms_dbfs(seg, sr, cfg.frame_ms, cfg.hop_ms)
        clean = seg_dbfs <= np.median(seg_dbfs) + cfg.transient_db
        run = _longest_true_run(clean)
        if run is None:
            continue
        s2 = run[0] * hop_len
        e2 = min(run[1] * hop_len + frame_len, seg.size)
        seg = seg[s2:e2]
        if seg.size < min_len:
            continue
        level_dbfs = rms_dbfs(seg)
        if level_dbfs < -95.0:
            continue  # digital zero / upstream padding: unusable as tone
        seg = (seg - np.float32(seg.mean())).astype(np.float32)
        info = {
            "start_sec": (start + s2) / sr,
            "end_sec": (start + e2) / sr,
            "dbfs": level_dbfs,
        }
        return seg, info
    return None


def _equal_power_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """sin/cos ramps (fade_in**2 + fade_out**2 == 1): flat RMS when crossfading
    uncorrelated noise, unlike the amplitude-complementary raised cosine."""
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    return np.sin(0.5 * np.pi * t), np.cos(0.5 * np.pi * t)


def tile_room_tone(
    segment: np.ndarray,
    num_samples: int,
    sr: int,
    rng: np.random.Generator,
    cfg: RoomToneConfig,
) -> np.ndarray:
    """Fill num_samples by overlap-adding randomized copies of segment.

    Anti-periodicity: random start offset, random chunk length (80-100%),
    50% time reversal, +/- tile_jitter_db per-chunk gain.
    """
    seg = np.asarray(segment, dtype=np.float32)
    out = np.zeros(num_samples, dtype=np.float32)
    if num_samples <= 0 or seg.size == 0:
        return out

    xfade = int(round(sr * cfg.tile_xfade_ms / 1000.0))
    xfade = max(1, min(xfade, seg.size // 3))
    fade_in, fade_out = _equal_power_ramps(xfade)
    min_take = min(seg.size, 2 * xfade + 1)

    pos = 0
    while pos < num_samples:
        take = int(round(seg.size * (1.0 - 0.2 * rng.random())))
        take = max(min_take, min(take, seg.size))
        offset = int(rng.integers(0, seg.size - take + 1))
        chunk = seg[offset : offset + take]
        if rng.random() < 0.5:
            chunk = chunk[::-1]
        gain = np.float32(
            10.0 ** (rng.uniform(-cfg.tile_jitter_db, cfg.tile_jitter_db) / 20.0)
        )
        chunk = chunk * gain

        if pos == 0:
            n = min(chunk.size, num_samples)
            out[:n] = chunk[:n]
            pos = n
        else:
            start = pos - xfade
            n = min(chunk.size, num_samples - start)
            nf = min(xfade, n)
            out[start : start + nf] *= fade_out[:nf]
            faded = chunk[:n].copy()
            faded[:nf] *= fade_in[:nf]
            out[start : start + n] += faded
            pos = start + n
    return out


def synthesize_matched_noise(
    x: np.ndarray,
    sr: int,
    num_samples: int,
    rng: np.random.Generator,
    cfg: RoomToneConfig,
    target_dbfs: float | None = None,
) -> np.ndarray | None:
    """Random-phase noise shaped by a robust per-bin floor spectrum of x.

    With target_dbfs set (x is a known-clean room-tone segment) all frames
    contribute to the envelope and the level is taken as given; otherwise the
    quietest frames are selected and the level targets the darkest ~50ms
    window of the clip -- its real deepest silence, not the quiet-frame
    median, which sits several dB above it and renders as a visibly brighter
    band on a spectrogram. None if x is shorter than one FFT frame."""
    x = np.asarray(x, dtype=np.float32)
    n_fft, hop = cfg.matched_n_fft, cfg.matched_hop
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if x.size < n_fft:
        return None

    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    frame_dbfs = 20.0 * np.log10(
        np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    )
    if target_dbfs is None:
        floor = float(np.percentile(frame_dbfs, cfg.floor_percentile))
        quiet = np.flatnonzero(frame_dbfs <= floor + cfg.quiet_margin_db)
        if quiet.size < cfg.matched_min_quiet_frames:
            quiet = np.argsort(frame_dbfs)[: cfg.matched_min_quiet_frames]
        win_frames = max(1, int(round(50.0 / (hop / sr * 1000.0))))
        if frame_dbfs.size >= win_frames:
            smoothed = np.convolve(
                frame_dbfs, np.ones(win_frames) / win_frames, mode="valid"
            )
            target_dbfs = float(smoothed.min())
        else:
            target_dbfs = floor
    else:
        quiet = np.arange(n_frames)

    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames[quiet] * window[None, :], axis=1))
    # per-bin low percentile: even "quiet" frames carry speech decay (mostly
    # low frequency); the 25th percentile tracks the true floor per bin
    env = np.percentile(spec, 25.0, axis=0)
    env = np.convolve(env, np.ones(3) / 3.0, mode="same")
    env_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    env[env_freqs < cfg.highpass_hz] = 0.0

    white = rng.standard_normal(num_samples).astype(np.float32)
    full_freqs = np.fft.rfftfreq(num_samples, d=1.0 / sr)
    shaped = np.fft.irfft(
        np.fft.rfft(white) * np.interp(full_freqs, env_freqs, env), n=num_samples
    )

    target_dbfs += float(rng.uniform(-cfg.matched_level_jitter_db, 0.0))
    target_dbfs = float(
        np.clip(target_dbfs, cfg.matched_dbfs_min, cfg.matched_dbfs_max)
    )
    return normalize_rms(shaped, target_dbfs).astype(np.float32)


def _shaped_noise(
    num_samples: int,
    sr: int,
    rng: np.random.Generator,
    env_freqs: np.ndarray,
    env_mag: np.ndarray,
    level_dbfs: float,
) -> np.ndarray:
    """Random-phase noise with the given magnitude envelope, RMS-normalized."""
    white = rng.standard_normal(num_samples).astype(np.float32)
    full_freqs = np.fft.rfftfreq(num_samples, d=1.0 / sr)
    shaped = np.fft.irfft(
        np.fft.rfft(white) * np.interp(full_freqs, env_freqs, env_mag),
        n=num_samples,
    )
    return normalize_rms(shaped, level_dbfs).astype(np.float32)


def extract_ambient_profile(
    x: np.ndarray, sr: int, cfg: RoomToneConfig
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(env_freqs, env_mag, level_dbfs) of a reference recording's ambience,
    from its quietest STFT frames. None if the reference is too short."""
    x = np.asarray(x, dtype=np.float32)
    n_fft, hop = cfg.matched_n_fft, cfg.matched_hop
    if x.size < n_fft:
        return None
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    frame_dbfs = 20.0 * np.log10(
        np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    )
    floor = float(np.percentile(frame_dbfs, cfg.floor_percentile))
    quiet = np.flatnonzero(frame_dbfs <= floor + cfg.quiet_margin_db)
    if quiet.size < cfg.matched_min_quiet_frames:
        quiet = np.argsort(frame_dbfs)[: cfg.matched_min_quiet_frames]
    level_dbfs = float(np.median(frame_dbfs[quiet]))

    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames[quiet] * window[None, :], axis=1))
    # mean-power envelope: reproduces the reference's per-band energies
    # (percentile envelopes come out several dB darker at 3-6 kHz)
    env = np.sqrt(np.mean(spec.astype(np.float64) ** 2, axis=0))
    env = np.convolve(env, np.ones(3) / 3.0, mode="same")
    env_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    env[env_freqs < cfg.highpass_hz] = 0.0
    return env_freqs, env, level_dbfs


def add_pause_ambient(
    y: np.ndarray,
    sr: int,
    pause_start: int,
    pause_end: int,
    rng: np.random.Generator,
    env_freqs: np.ndarray | None = None,
    env_mag: np.ndarray | None = None,
    level_dbfs: float = DEFAULT_AMBIENT_DBFS,
    ramp_ms: float = DEFAULT_AMBIENT_RAMP_MS,
) -> np.ndarray:
    """Overlay broadband ambience across y[pause_start:pause_end].

    The ambience ramps up/down over ramp_ms UNDER the neighboring speech
    (where it is masked), so the floor reads as continuous instead of
    switching on at the pause edge. Defaults to the built-in template."""
    y = np.asarray(y, dtype=np.float32).copy()
    if env_freqs is None or env_mag is None:
        env_freqs = AMBIENT_TEMPLATE_FREQS
        env_mag = 10.0 ** (AMBIENT_TEMPLATE_DB / 20.0)
    ramp = int(round(sr * ramp_ms / 1000.0))
    a = max(0, pause_start - ramp)
    b = min(y.size, pause_end + ramp)
    n = b - a
    if n <= 0:
        return y
    amb = _shaped_noise(n, sr, rng, env_freqs, env_mag, level_dbfs)
    ramp_in = pause_start - a
    if ramp_in > 0:
        t = np.linspace(0.0, 1.0, ramp_in, endpoint=False, dtype=np.float32)
        amb[:ramp_in] *= 0.5 - 0.5 * np.cos(np.pi * t)
    ramp_out = b - pause_end
    if ramp_out > 0:
        t = np.linspace(1.0, 0.0, ramp_out, endpoint=False, dtype=np.float32)
        amb[-ramp_out:] *= 0.5 - 0.5 * np.cos(np.pi * t)
    y[a:b] += amb
    return y


def load_breath_bank(bank_dir: Path | str, sr: int) -> list[tuple[str, np.ndarray]]:
    """All wavs in bank_dir as (name, 1-D float32 array at sr); empty if none."""
    bank = []
    for f in sorted(Path(bank_dir).glob("*.wav")):
        audio, file_sr = torchaudio.load(str(f))
        if file_sr != sr:
            audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
        bank.append((f.name, audio[0].to(torch.float32).numpy()))
    return bank


def load_event_banks(
    root: Path | str = DEFAULT_BREATH_DIR, sr: int = SAMPLE_RATE
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """{'breath': ..., 'click': ..., 'exhale': ...} from the bank layout
    root/*.wav, root/clicks/*.wav, root/exhales/*.wav."""
    root = Path(root)
    return {
        "breath": load_breath_bank(root, sr) if root.is_dir() else [],
        "click": load_breath_bank(root / "clicks", sr)
        if (root / "clicks").is_dir()
        else [],
        "exhale": load_breath_bank(root / "exhales", sr)
        if (root / "exhales").is_dir()
        else [],
    }


def add_pause_breath(
    y: np.ndarray,
    sr: int,
    pause_start: int,
    pause_end: int,
    rng: np.random.Generator,
    breath: np.ndarray,
    level_dbfs: float,
    end_gap_ms: tuple[float, float] = BREATH_END_GAP_MS,
    fade_ms: float = 25.0,
) -> tuple[np.ndarray, dict | None]:
    """Overlay a breath inside the pause, ending shortly before speech
    resumes (the natural pre-speech inhale position). Returns (y, info);
    info is None when the pause is too short to fit one."""
    y = np.asarray(y, dtype=np.float32).copy()
    gap = int(round(rng.uniform(*end_gap_ms) * sr / 1000.0))
    space = pause_end - pause_start - gap
    if space < int(0.15 * sr):
        return y, None
    b = np.asarray(breath, dtype=np.float32)
    if b.size > space:
        b = b[b.size - space :]  # keep the end: the inhale peaks late
    b = normalize_rms(b, level_dbfs).astype(np.float32)
    b = apply_fade_edges(b, sr, fade_ms)
    end = pause_end - gap
    start = end - b.size
    y[start:end] += b
    return y, {
        "breath_start_sec": start / sr,
        "breath_dur_sec": b.size / sr,
        "breath_dbfs": level_dbfs,
    }


def add_pause_events(
    y: np.ndarray,
    sr: int,
    pause_start: int,
    pause_end: int,
    rng: np.random.Generator,
    ambient_dbfs: float,
    banks: dict[str, list[tuple[str, np.ndarray]]],
    breath_prob: float = DEFAULT_BREATH_PROB,
    click_prob: float = DEFAULT_CLICK_PROB,
    exhale_prob: float = DEFAULT_EXHALE_PROB,
) -> tuple[np.ndarray, list[dict]]:
    """Compose probabilistic mouth events inside a pause, in the natural
    order: [exhale after speech stops] ... [click] [inhale] -> speech.

    Event levels are relative to ambient_dbfs (the pause floor). Returns
    (y, events) with one metadata dict per placed event."""
    y = np.asarray(y, dtype=np.float32).copy()
    events: list[dict] = []
    pause_len = pause_end - pause_start

    def _overlay(arr: np.ndarray, start: int, level: float, fade_ms: float) -> None:
        a = normalize_rms(arr.astype(np.float32), level).astype(np.float32)
        a = apply_fade_edges(a, sr, fade_ms)
        y[start : start + a.size] += a

    inhale_start = None
    breath_bank = banks.get("breath") or []
    if (
        breath_bank
        and pause_len >= int(MIN_PAUSE_FOR_BREATH_S * sr)
        and rng.random() < breath_prob
    ):
        name, arr = breath_bank[int(rng.integers(0, len(breath_bank)))]
        level = min(
            BREATH_MAX_DBFS,
            ambient_dbfs + float(rng.uniform(*BREATH_GAIN_ABOVE_AMBIENT_DB)),
        )
        y, info = add_pause_breath(y, sr, pause_start, pause_end, rng, arr, level)
        if info is not None:
            inhale_start = int(round(info["breath_start_sec"] * sr))
            events.append(
                {
                    "type": "inhale",
                    "file": name,
                    "start_sec": info["breath_start_sec"],
                    "dur_sec": info["breath_dur_sec"],
                    "dbfs": level,
                }
            )

    click_end = None
    click_bank = banks.get("click") or []
    if click_bank and rng.random() < click_prob:
        name, arr = click_bank[int(rng.integers(0, len(click_bank)))]
        level = min(
            CLICK_MAX_DBFS,
            ambient_dbfs + float(rng.uniform(*CLICK_GAIN_ABOVE_AMBIENT_DB)),
        )
        if inhale_start is not None:
            # mouth opens (click), then draws air (inhale)
            end = inhale_start - int(round(rng.uniform(0.0, 0.08) * sr))
        else:
            end = pause_end - int(round(rng.uniform(0.03, 0.12) * sr))
        start = end - arr.size
        if start >= pause_start + int(0.03 * sr):
            _overlay(arr, start, level, fade_ms=3.0)
            click_end = start
            events.append(
                {
                    "type": "click",
                    "file": name,
                    "start_sec": start / sr,
                    "dur_sec": arr.size / sr,
                    "dbfs": level,
                }
            )

    exhale_bank = banks.get("exhale") or []
    if exhale_bank and rng.random() < exhale_prob:
        name, arr = exhale_bank[int(rng.integers(0, len(exhale_bank)))]
        level = min(
            EXHALE_MAX_DBFS,
            ambient_dbfs + float(rng.uniform(*EXHALE_GAIN_ABOVE_AMBIENT_DB)),
        )
        start = pause_start + int(round(rng.uniform(0.03, 0.15) * sr))
        if click_end is not None:
            limit = click_end - int(0.05 * sr)
        elif inhale_start is not None:
            limit = inhale_start - int(0.05 * sr)
        else:
            limit = pause_end - int(0.2 * sr)
        space = limit - start
        if space >= int(0.12 * sr):
            arr = arr[: min(arr.size, space)]  # keep the head: exhale peaks early
            _overlay(arr, start, level, fade_ms=20.0)
            events.append(
                {
                    "type": "exhale",
                    "file": name,
                    "start_sec": start / sr,
                    "dur_sec": arr.size / sr,
                    "dbfs": level,
                }
            )

    events.sort(key=lambda e: e["start_sec"])
    return y, events


def _tpdf_noise(
    num_samples: int, rng: np.random.Generator, noise_dbfs: float
) -> np.ndarray:
    u1 = rng.random(num_samples).astype(np.float32) - 0.5
    u2 = rng.random(num_samples).astype(np.float32) - 0.5
    return normalize_rms(u1 + u2, noise_dbfs).astype(np.float32)


def get_pause_tone(
    x: np.ndarray,
    sr: int,
    num_samples: int,
    rng: np.random.Generator,
    cfg: RoomToneConfig,
    prefer: str | None = None,
) -> tuple[np.ndarray, str, dict]:
    """Tone ladder: tiled room tone -> spectrally matched noise -> TPDF at the
    measured floor. Returns (tone, method, room_tone_region_info)."""
    x = np.asarray(x, dtype=np.float32)
    extracted = extract_room_tone(x, sr, cfg, prefer=prefer)
    if extracted is not None:
        seg, info = extracted
        if seg.size >= int(round(sr * cfg.full_tile_min_ms / 1000.0)):
            return (
                tile_room_tone(seg, num_samples, sr, rng, cfg),
                "tiled_room_tone",
                info,
            )
        # short real segment: synthesize from its own spectrum/level instead
        # of tiling it (repeating a <200ms loop prints visible stripes)
        seg_tone = synthesize_matched_noise(
            seg, sr, num_samples, rng, cfg, target_dbfs=rms_dbfs(seg)
        )
        if seg_tone is not None:
            return seg_tone, "segment_matched_noise", info

    matched = synthesize_matched_noise(x, sr, num_samples, rng, cfg)
    if matched is not None:
        return matched, "matched_noise", {}

    floor = float(
        np.percentile(
            frame_rms_dbfs(x, sr, cfg.frame_ms, cfg.hop_ms), cfg.floor_percentile
        )
    )
    noise_dbfs = float(
        np.clip(
            floor + rng.uniform(-cfg.tpdf_jitter_db, cfg.tpdf_jitter_db),
            cfg.tpdf_dbfs_min,
            cfg.tpdf_dbfs_max,
        )
    )
    return _tpdf_noise(num_samples, rng, noise_dbfs), "tpdf", {}


def crossfade_join(
    before: np.ndarray, tone: np.ndarray, after: np.ndarray, xfade: int
) -> np.ndarray:
    """Splice tone between before/after with equal-power mix-in crossfades.

    tone must be pause_samples + 2*xfade long; its first/last xfade samples
    overlap the neighboring speech, so the output length is exactly
    len(before) + pause_samples + len(after).
    """
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    tone = np.asarray(tone, dtype=np.float32)
    if xfade <= 0:
        return np.concatenate([before, tone, after])
    fade_in, fade_out = _equal_power_ramps(xfade)
    left = before[-xfade:] * fade_out + tone[:xfade] * fade_in
    right = tone[-xfade:] * fade_out + after[:xfade] * fade_in
    return np.concatenate(
        [before[:-xfade], left, tone[xfade : tone.size - xfade], right, after[xfade:]]
    )


def extend_existing_pause(
    audio: np.ndarray,
    sr: int,
    pause_start_sample: int,
    pause_end_sample: int,
    extension_samples: int,
    rng: np.random.Generator,
    cfg: RoomToneConfig,
    splice_xfade_ms: float = 15.0,
) -> tuple[np.ndarray, dict]:
    """Lengthen a natural within-utterance pause without touching its edges.

    The original speech decay and following speech onset are kept bit-for-bit.
    A cut point is chosen at the quietest short window *inside* the supplied
    pause, and native local room tone is inserted there with a short
    equal-power tone-to-tone crossfade.  Unlike the two-utterance construction,
    this path deliberately adds no external ambience and no mouth events.

    Args:
        audio: Mono float waveform.
        sr: Sample rate.
        pause_start_sample/pause_end_sample: Half-open natural-pause span.
        extension_samples: Exact number of samples to add.
        rng: Deterministic generator for native-tone tiling/synthesis.
        cfg: Room-tone extraction/synthesis parameters.
        splice_xfade_ms: Crossfade at the internal quiet cut.  This is shorter
            than the speech-boundary crossfade because both sides are already
            room tone.

    Returns:
        ``(extended_audio, metadata)``.  The returned length is exactly
        ``len(audio) + extension_samples`` and the updated pause span is
        ``[pause_start_sample, pause_end_sample + extension_samples)``.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"audio must be mono 1-D, got shape={x.shape}")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    start = int(pause_start_sample)
    end = int(pause_end_sample)
    extension = int(extension_samples)
    if not (0 <= start < end <= x.size):
        raise ValueError(
            f"invalid pause span [{start}, {end}) for audio length {x.size}"
        )
    if extension <= 0:
        raise ValueError(f"extension_samples must be positive, got {extension}")

    pause = x[start:end]
    pause_level = rms_dbfs(pause)
    if not np.isfinite(pause_level) or pause_level < -95.0:
        raise ValueError(
            "natural pause is digital-zero/too quiet to provide native room tone: "
            f"{pause_level:.2f} dBFS"
        )

    xfade = int(round(float(splice_xfade_ms) * sr / 1000.0))
    # Leave enough native quiet samples on both sides of the cut for the
    # crossfade.  A 5 ms reserve prevents the selected window from touching a
    # VAD boundary even for the shortest accepted pauses.
    reserve = xfade + int(round(0.005 * sr))
    if pause.size <= 2 * reserve:
        xfade = max(1, min(xfade, pause.size // 4))
        reserve = xfade
    if pause.size <= 2 * reserve:
        raise ValueError(
            f"pause ({pause.size} samples) is too short for xfade={xfade}"
        )

    # Find the quietest 30 ms RMS window inside the safe pause core.  The cut
    # is placed at that window's centre, never at a speech/pause boundary.
    safe = pause[reserve : pause.size - reserve]
    probe = max(1, min(int(round(0.030 * sr)), safe.size))
    if safe.size == probe:
        local_start = 0
    else:
        # Sliding energy via cumulative sum avoids constructing a large frame
        # matrix for long pauses.
        power = safe.astype(np.float64) ** 2
        cumulative = np.concatenate(([0.0], np.cumsum(power)))
        energies = cumulative[probe:] - cumulative[:-probe]
        local_start = int(np.argmin(energies))
    cut = start + reserve + local_start + probe // 2
    cut = max(start + xfade, min(cut, end - xfade))

    # Use only this pause's native core as the texture reference.  Long cores
    # retain their actual waveform through randomized tiling; shorter cores use
    # their spectrum and measured level, avoiding a visibly periodic short
    # loop.  No generic TPDF fallback is allowed for this high-quality path.
    native_core = pause[xfade : pause.size - xfade]
    if native_core.size < max(1, int(round(cfg.min_tile_len_ms * sr / 1000.0))):
        native_core = pause
    tone_samples = extension + 2 * xfade
    if native_core.size >= int(round(cfg.full_tile_min_ms * sr / 1000.0)):
        tone = tile_room_tone(native_core, tone_samples, sr, rng, cfg)
        method = "native_tiled_room_tone"
    else:
        tone = synthesize_matched_noise(
            native_core,
            sr,
            tone_samples,
            rng,
            cfg,
            target_dbfs=pause_level,
        )
        if tone is None:
            # This is still a native-waveform fallback, not synthetic TPDF.
            tone = tile_room_tone(native_core, tone_samples, sr, rng, cfg)
            method = "native_short_tiled_room_tone"
        else:
            method = "native_segment_matched_noise"

    # A pause can have a non-zero aggregate RMS because of one residual click
    # while its long "quiet" core is still effectively digital black.  Do not
    # turn that core into a conspicuous zero-valued extension.  For 16-bit
    # source material, a sustained level below -95 dBFS is below one-LSB RMS
    # in practice and is treated as unusable native room tone.
    inserted_tone_dbfs = rms_dbfs(tone[xfade : xfade + extension])
    if not np.isfinite(inserted_tone_dbfs) or inserted_tone_dbfs < -95.0:
        raise ValueError(
            "selected native pause core is digital-zero/too quiet to extend: "
            f"{inserted_tone_dbfs:.2f} dBFS"
        )

    y = crossfade_join(x[:cut], tone, x[cut:], xfade).astype(np.float32)
    expected = x.size + extension
    if y.size != expected:
        raise RuntimeError(
            f"pause extension length mismatch: got {y.size}, expected {expected}"
        )
    metadata = {
        "pause_method": method,
        "original_pause_start_sample": start,
        "original_pause_end_sample": end,
        "extended_pause_start_sample": start,
        "extended_pause_end_sample": end + extension,
        "extension_samples": extension,
        "cut_sample": cut,
        "splice_xfade_samples": xfade,
        "native_pause_dbfs": float(pause_level),
        "inserted_tone_dbfs": float(inserted_tone_dbfs),
    }
    return y, metadata


def build_pause_between(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    pause_samples: int,
    rng: np.random.Generator,
    cfg: RoomToneConfig,
    xfade: int,
) -> tuple[np.ndarray, dict]:
    """Pause tone spanning pause_samples + 2*xfade that blends clip1's ambience
    into clip2's with one equal-power crossfade across the whole pause (this
    also absorbs any noise-floor level mismatch between the clips)."""
    total = pause_samples + 2 * xfade
    # prefer the regions adjacent to the junction (tail of clip1, head of
    # clip2) so the pause continues exactly the silence it touches
    tone1, method1, info1 = get_pause_tone(audio1, sr, total, rng, cfg, prefer="end")
    tone2, method2, info2 = get_pause_tone(audio2, sr, total, rng, cfg, prefer="start")
    fade_in, fade_out = _equal_power_ramps(total)
    pause = tone1 * fade_out + tone2 * fade_in
    # when the two floors differ a lot the raw mix power P1*cos^2 + P2*sin^2
    # drops abruptly near the quiet end; re-shape it onto a dB-linear
    # trajectory between the two tone levels (a no-op when P1 ~= P2)
    eps = 1e-24
    p1 = float(np.mean(tone1.astype(np.float64) ** 2)) + eps
    p2 = float(np.mean(tone2.astype(np.float64) ** 2)) + eps
    t = (np.arange(total, dtype=np.float64) + 0.5) / max(total, 1)
    target = p1 * (p2 / p1) ** t
    actual = p1 * fade_out.astype(np.float64) ** 2 + p2 * fade_in.astype(np.float64) ** 2
    pause = (pause * np.sqrt(target / actual)).astype(np.float32)
    meta = {
        "pause_method_wav1": method1,
        "pause_method_wav2": method2,
        "room_tone_region_wav1": info1 or None,
        "room_tone_region_wav2": info2 or None,
    }
    return pause, meta


def room_tone_config_from_args(args: argparse.Namespace) -> RoomToneConfig:
    return RoomToneConfig(
        frame_ms=args.noise_frame_ms,
        hop_ms=args.noise_hop_ms,
        floor_percentile=args.noise_floor_percentile,
        quiet_margin_db=args.quiet_margin_db,
        min_tile_len_ms=args.min_tile_len_ms,
        tile_xfade_ms=args.tile_xfade_ms,
        transient_db=args.transient_db,
        tpdf_jitter_db=args.noise_jitter_db,
    )


def load_audio_mono(path: Path, sr: int) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if sample_rate != sr:
        audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
    return audio[0].to(torch.float32).contiguous()


def main() -> int:
    args = parse_args()
    if args.min_silence > args.max_silence:
        raise ValueError("--min-silence must be <= --max-silence")
    if args.noise_dbfs_min > args.noise_dbfs_max:
        raise ValueError("--noise-dbfs-min must be <= --noise-dbfs-max")
    if not 0.0 <= args.noise_floor_percentile <= 100.0:
        raise ValueError("--noise-floor-percentile must be in [0, 100]")
    if args.noise_frame_ms <= 0.0:
        raise ValueError("--noise-frame-ms must be > 0")
    if args.noise_hop_ms <= 0.0:
        raise ValueError("--noise-hop-ms must be > 0")
    if args.noise_jitter_db < 0.0:
        raise ValueError("--noise-jitter-db must be >= 0")

    rng = np.random.default_rng(args.seed)
    silence_duration = (
        float(args.silence_duration)
        if args.silence_duration is not None
        else float(rng.uniform(args.min_silence, args.max_silence))
    )

    audio1 = load_audio_mono(args.wav1, args.sample_rate)
    audio2 = load_audio_mono(args.wav2, args.sample_rate)

    if args.noise_type == "room_tone":
        cfg = room_tone_config_from_args(args)
        # Use a child generator for room-tone synthesis.
        tone_rng = rng.spawn(1)[0]
        a1 = audio1.numpy()
        a2 = audio2.numpy()
        pause_samples = int(round(silence_duration * args.sample_rate))
        xfade = int(round(args.boundary_xfade_ms * args.sample_rate / 1000.0))
        xfade = max(0, min(xfade, a1.size, a2.size))
        pause_tone, pause_meta = build_pause_between(
            a1, a2, args.sample_rate, pause_samples, tone_rng, cfg, xfade
        )
        combined_np = crossfade_join(a1, pause_tone, a2, xfade)

        ambient_meta: dict[str, float | str | None] = {
            "ambient_source": None,
            "ambient_dbfs": None,
        }
        if not args.no_pause_ambient:
            env_freqs = env_mag = None
            ambient_level = DEFAULT_AMBIENT_DBFS
            ambient_source = "builtin_template"
            if args.ambient_ref is not None:
                ref = load_audio_mono(args.ambient_ref, args.sample_rate).numpy()
                profile = extract_ambient_profile(ref, args.sample_rate, cfg)
                if profile is None:
                    raise ValueError(f"--ambient-ref too short: {args.ambient_ref}")
                env_freqs, env_mag, ambient_level = profile
                ambient_source = str(args.ambient_ref)
            if args.ambient_dbfs is not None:
                ambient_level = args.ambient_dbfs
            else:
                ambient_level += float(
                    tone_rng.uniform(
                        -DEFAULT_AMBIENT_JITTER_DB, DEFAULT_AMBIENT_JITTER_DB
                    )
                )
            combined_np = add_pause_ambient(
                combined_np,
                args.sample_rate,
                a1.size,
                a1.size + pause_samples,
                tone_rng,
                env_freqs=env_freqs,
                env_mag=env_mag,
                level_dbfs=ambient_level,
                ramp_ms=args.ambient_ramp_ms,
            )
            ambient_meta = {
                "ambient_source": ambient_source,
                "ambient_dbfs": ambient_level,
                "ambient_ramp_ms": args.ambient_ramp_ms,
            }

        breath_meta: dict = {"pause_events": []}
        breath_dir = args.breath_dir
        if breath_dir is None and DEFAULT_BREATH_DIR.is_dir():
            breath_dir = DEFAULT_BREATH_DIR
        if not args.no_breath and breath_dir is not None:
            banks = load_event_banks(breath_dir, args.sample_rate)
            base = (
                ambient_meta["ambient_dbfs"]
                if ambient_meta["ambient_dbfs"] is not None
                else DEFAULT_AMBIENT_DBFS
            )
            combined_np, events = add_pause_events(
                combined_np,
                args.sample_rate,
                a1.size,
                a1.size + pause_samples,
                tone_rng,
                base,
                banks,
                breath_prob=args.breath_prob,
                click_prob=args.click_prob,
                exhale_prob=args.exhale_prob,
            )
            breath_meta = {"pause_events": events}

        combined = torch.from_numpy(combined_np)
        silence = torch.from_numpy(
            combined_np[a1.size : a1.size + pause_samples].copy()
        )
        noise_dbfs = rms_dbfs(silence.numpy())
        noise_metadata = {
            "noise_mode": "room_tone",
            "noise_floor_dbfs_wav1": estimate_noise_floor_dbfs(
                audio1,
                sr=args.sample_rate,
                frame_ms=args.noise_frame_ms,
                hop_ms=args.noise_hop_ms,
                percentile=args.noise_floor_percentile,
            ),
            "noise_floor_dbfs_wav2": estimate_noise_floor_dbfs(
                audio2,
                sr=args.sample_rate,
                frame_ms=args.noise_frame_ms,
                hop_ms=args.noise_hop_ms,
                percentile=args.noise_floor_percentile,
            ),
            "boundary_xfade_ms": args.boundary_xfade_ms,
            "pause_rms_dbfs": noise_dbfs,
            **pause_meta,
            **ambient_meta,
            **breath_meta,
        }
    else:
        noise_dbfs, noise_metadata = choose_noise_dbfs(
            audio1=audio1,
            audio2=audio2,
            sr=args.sample_rate,
            rng=rng,
            args=args,
        )
        if args.noise_type == "tpdf":
            silence, noise_dbfs = generate_tpdf_pause(
                duration_sec=silence_duration,
                sr=args.sample_rate,
                rng=rng,
                fade_ms=args.fade_ms,
                noise_dbfs=noise_dbfs,
            )
        else:
            num_samples = int(round(silence_duration * args.sample_rate))
            if args.noise_type == "pink":
                noise = generate_pinkish_noise(
                    num_samples, args.sample_rate, noise_dbfs, rng
                )
                noise = apply_fade_edges(noise, args.sample_rate, args.fade_ms)
            else:  # zero
                noise = np.zeros(num_samples, dtype=np.float32)
                noise_dbfs = None
            silence = torch.from_numpy(noise)
        combined = torch.cat([audio1, silence, audio2], dim=0)

    silence_nonzero = int(torch.count_nonzero(silence).item())
    silence_rms = (
        float(torch.sqrt(torch.mean(silence**2)).item())
        if silence.numel()
        else 0.0
    )
    silence_min = float(silence.min().item()) if silence.numel() else 0.0
    silence_max = float(silence.max().item()) if silence.numel() else 0.0

    if args.noise_type != "zero" and silence.numel() and silence_nonzero == 0:
        raise RuntimeError(
            "generated silence is all zeros; increase --noise-dbfs-min/max or "
            "check the output encoding"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if args.output.suffix.lower() == ".wav":
        save_kwargs = {"encoding": "PCM_F", "bits_per_sample": 32}
    torchaudio.save(
        str(args.output),
        combined.unsqueeze(0),
        args.sample_rate,
        **save_kwargs,
    )

    metadata = {
        "wav1": str(args.wav1),
        "wav2": str(args.wav2),
        "output": str(args.output),
        "sample_rate": args.sample_rate,
        "noise_type": args.noise_type,
        "silence_duration": silence_duration,
        "noise_dbfs": noise_dbfs,
        "noise_dbfs_min": args.noise_dbfs_min,
        "noise_dbfs_max": args.noise_dbfs_max,
        "noise_floor_percentile": args.noise_floor_percentile,
        "noise_frame_ms": args.noise_frame_ms,
        "noise_hop_ms": args.noise_hop_ms,
        "noise_jitter_db": args.noise_jitter_db,
        **noise_metadata,
        "fade_ms": args.fade_ms,
        "audio1_samples": int(audio1.numel()),
        "silence_samples": int(silence.numel()),
        "audio2_samples": int(audio2.numel()),
        "combined_samples": int(combined.numel()),
        "silence_min": silence_min,
        "silence_max": silence_max,
        "silence_rms": silence_rms,
        "silence_nonzero_samples": silence_nonzero,
    }

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
