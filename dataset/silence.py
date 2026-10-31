from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import librosa
import numpy as np

DEFAULT_MIN_SILENCE = 0.4
DEFAULT_MAX_SILENCE = 1.0

SAMPLE_RATE = 16000
DEFAULT_NOISE_FLOOR_PERCENTILE = 1.0
DEFAULT_NOISE_FRAME_MS = 20.0
DEFAULT_NOISE_HOP_MS = 10.0
DEFAULT_NOISE_JITTER_DB = 3.0

from .room_tone import (
    DEFAULT_AMBIENT_DBFS,
    DEFAULT_AMBIENT_JITTER_DB,
    DEFAULT_BOUNDARY_XFADE_MS,
    DEFAULT_BREATH_DIR,
    DEFAULT_BREATH_PROB,
    DEFAULT_CLICK_PROB,
    DEFAULT_EXHALE_PROB,
    RoomToneConfig,
    add_pause_ambient,
    add_pause_events,
    build_pause_between,
    crossfade_join,
    load_event_banks,
)

# Per-process breath/click/exhale bank cache (DataLoader workers each load once).
_EVENT_BANKS: Optional[dict] = None


def _get_event_banks(breath_dir: Optional[Path] = None) -> dict:
    global _EVENT_BANKS
    if _EVENT_BANKS is None:
        d = Path(breath_dir) if breath_dir is not None else DEFAULT_BREATH_DIR
        _EVENT_BANKS = (
            load_event_banks(d, SAMPLE_RATE)
            if d.is_dir()
            else {"breath": [], "click": [], "exhale": []}
        )
    return _EVENT_BANKS


def _join_with_room_tone(
    audio1: np.ndarray,
    audio2: np.ndarray,
    silence_length: float,
    sr: int,
    rng: np.random.Generator,
    breath_dir: Optional[Path] = None,
    pause_style: str = "legacy",
) -> np.ndarray:
    """Splice ``audio1`` and ``audio2`` with a natural room-tone pause of
    ``silence_length`` seconds, mirroring the offline reference's room_tone
    path: build_pause_between -> crossfade_join -> add_pause_ambient ->
    add_pause_events.  Returns the combined 1-D float32 waveform whose length
    is exactly ``len(audio1) + pause_samples + len(audio2)``.

    The pause tone / ambience / event RNG draws come from a spawned child
    generator, matching the reference (``tone_rng = rng.spawn(1)[0]``), so the
    online mix reproduces the offline processing given the same seed.

    ``pause_style="legacy"`` adds ambience and mouth-event overlays.
    ``pause_style="clean"`` uses locally derived room tone and a 20 ms
    boundary crossfade. The default is ``legacy``.
    """
    if pause_style not in {"legacy", "clean"}:
        raise ValueError(
            f"pause_style must be 'legacy' or 'clean', got {pause_style!r}"
        )

    a1 = np.asarray(audio1, dtype=np.float32)
    a2 = np.asarray(audio2, dtype=np.float32)
    pause_samples = int(round(silence_length * sr))
    if pause_samples <= 0:
        return np.concatenate([a1, a2]).astype(np.float32)

    cfg = RoomToneConfig(
        frame_ms=DEFAULT_NOISE_FRAME_MS,
        hop_ms=DEFAULT_NOISE_HOP_MS,
        floor_percentile=DEFAULT_NOISE_FLOOR_PERCENTILE,
        tpdf_jitter_db=DEFAULT_NOISE_JITTER_DB,
    )
    # Child generator so the pause-tone / ambient / event draws stay independent
    # of the duration draws, exactly as the reference does.
    tone_rng = rng.spawn(1)[0]

    xfade = int(round(DEFAULT_BOUNDARY_XFADE_MS * sr / 1000.0))
    xfade = max(0, min(xfade, a1.size, a2.size))
    pause_tone, _ = build_pause_between(
        a1, a2, sr, pause_samples, tone_rng, cfg, xfade
    )
    y = crossfade_join(a1, pause_tone, a2, xfade)

    if pause_style == "legacy":
        ambient_level = DEFAULT_AMBIENT_DBFS + float(
            tone_rng.uniform(-DEFAULT_AMBIENT_JITTER_DB, DEFAULT_AMBIENT_JITTER_DB)
        )
        y = add_pause_ambient(
            y,
            sr,
            a1.size,
            a1.size + pause_samples,
            tone_rng,
            level_dbfs=ambient_level,
        )

        banks = _get_event_banks(breath_dir)
        if any(banks.get(k) for k in ("breath", "click", "exhale")):
            y, _ = add_pause_events(
                y,
                sr,
                a1.size,
                a1.size + pause_samples,
                tone_rng,
                ambient_level,
                banks,
                breath_prob=DEFAULT_BREATH_PROB,
                click_prob=DEFAULT_CLICK_PROB,
                exhale_prob=DEFAULT_EXHALE_PROB,
            )
    return y.astype(np.float32)


def concat_audios_with_silence(
    paths: Sequence[str],
    silence_length: float,
    sr: int = SAMPLE_RATE,
    seed: Optional[int] = None,
    pause_style: str = "legacy",
) -> np.ndarray:
    """Load each path (mono @ ``sr``) and concatenate, inserting a natural
    room-tone pseudo-silence of ``silence_length`` seconds between consecutive
    segments.  A single-element ``paths`` returns that audio unchanged.

    ``seed`` makes the generated pause reproducible per sample.  Select
    ``pause_style="clean"`` to omit the pause-only ambience and mouth
    events while retaining local room tone and boundary crossfades.
    """
    audios = [
        librosa.load(p, sr=sr, mono=True)[0].astype(np.float32) for p in paths
    ]
    if len(audios) == 1:
        return audios[0]

    rng = np.random.default_rng(seed)
    out = audios[0]
    for k in range(1, len(audios)):
        out = _join_with_room_tone(
            out,
            audios[k],
            silence_length,
            sr,
            rng,
            pause_style=pause_style,
        )
    return out.astype(np.float32)


def concat_audios_with_random_silence(
    paths: Sequence[str],
    sr: int = SAMPLE_RATE,
    seed: Optional[int] = None,
    min_silence: float = DEFAULT_MIN_SILENCE,
    max_silence: float = DEFAULT_MAX_SILENCE,
) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    """Concatenate the segment audios, inserting between each consecutive pair
    a natural room-tone pseudo-silence whose duration is drawn i.i.d. from
    ``Uniform(min_silence, max_silence)``.

    Unlike :func:`concat_audios_with_silence` (fixed length, target reuse) this
    is the RL-side mixer: it picks a fresh duration per gap and reports where
    each pause landed so the caller can build ``source_vad`` pause intervals.

    ``seed=None`` (the default) draws fresh randomness on every call, so a
    sample re-rolls its pauses each epoch; pass an int to make a mix
    reproducible.

    Returns:
        combined : the mixed mono waveform at ``sr``.
        gaps     : one ``(start_sec, end_sec, duration_sec)`` per inserted
                   pause, measured from the *actual* sample offsets in
                   ``combined`` (so rounding is already accounted for). Empty
                   when ``paths`` has a single element.
    """
    audios = [
        librosa.load(p, sr=sr, mono=True)[0].astype(np.float32) for p in paths
    ]
    if len(audios) == 1:
        return audios[0], []

    rng = np.random.default_rng(seed)
    out = audios[0]
    gaps: List[Tuple[float, float, float]] = []
    for k in range(1, len(audios)):
        duration = float(rng.uniform(min_silence, max_silence))
        before = len(out)
        out = _join_with_room_tone(out, audios[k], duration, sr, rng)
        # The pause occupies out[before : before + pause_samples]; the trailing
        # segment audios[k] was appended after it.  Recover the pause span from
        # the total growth minus the segment length (crossfade_join preserves
        # the length identity, so this equals the nominal pause-sample count).
        pause_samples = len(out) - before - len(audios[k])
        start = before / sr
        end = (before + max(pause_samples, 0)) / sr
        gaps.append((start, end, end - start))
    return out.astype(np.float32), gaps
