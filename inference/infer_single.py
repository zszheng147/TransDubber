#!/usr/bin/env python
"""Translate English speech file(s) into isometric, pause-aligned Chinese speech.

    python inference/infer_single.py \
        --ckpt_path /path/to/checkpoint.pt \
        --audio demo/en_1.wav [demo/en_2.wav ...] \
        --out_dir demo_out

Per input ``<name>.<ext>`` the script writes ``<out_dir>/audio/<name>.wav``
(24 kHz) and appends one JSON record (English ASR with/without ``<pause>``,
Chinese translation, codec count, marker positions) to
``<out_dir>/predictions.jsonl``; ``<out_dir>/translated_texts.txt``
(``name<TAB>translation``) is regenerated after every run.  That layout is what
the metric scripts in ``eval/`` consume (``--root <out_dir>``).

Model paths come from ``--base_model`` / ``--token2wav_root`` or the
``base_model`` / ``token2wav_root`` environment variables.

Default decoding protocol: greedy
until ``<s2st_text>`` (deterministic ASR), nucleus sampling afterwards with
top-p 0.85 / temperature 0.85, stop at the second EOS, seed derived from the
file name.  The generated stream must match the training layout exactly::

    ASR <s2st_text> (translation + interleaved codec, including text EOS)
        <s2st_eos> remaining codec final-EOS

otherwise the sample is reported as a ``StructureError`` instead of being
rendered.  All codec tokens are then vocoded with the Fun-CosyVoice3 token2wav
stack, using the input clip as the speaker prompt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Qwen3-ASR-1.7B base directory (HF layout) and the Fun-CosyVoice3-0.5B
# directory holding flow.pt / hift.pt / campplus.onnx / speech_tokenizer_v3.onnx.
DEFAULT_BASE_MODEL = os.environ.get("base_model", "/path/to/Qwen3-ASR-1.7B")
DEFAULT_TOKEN2WAV_ROOT = os.environ.get(
    "token2wav_root", "/path/to/Fun-CosyVoice3-0.5B"
)
SPECIAL_TOKENS = ["<s2st_text>", "<s2st_eos>", "<|speech_codec|>", "<pause>"]
SPEECH_CODEC_OFFSET = 151936
SPEECH_CODEC_SIZE = 6561
OUTPUT_SAMPLE_RATE = 24000


class StructureError(ValueError):
    """The generated stream does not match the supervised target layout."""


@dataclass(frozen=True)
class ParsedGeneration:
    asr_ids: list[int]
    translation_ids: list[int]
    speech_codes: list[int]
    s2st_text_position: int
    text_eos_position: int
    s2st_eos_position: int
    final_eos_position: int
    codec_count: int
    generated_token_count: int


def _positions(values: Sequence[int], wanted: int) -> list[int]:
    return [index for index, value in enumerate(values) if value == wanted]


def parse_and_validate_generation(
    generated_ids: Sequence[int],
    *,
    s2st_text_id: int,
    s2st_eos_id: int,
    eos_id: int,
    codec_offset: int = SPEECH_CODEC_OFFSET,
    codec_size: int = SPEECH_CODEC_SIZE,
    min_codec_count: int = 1,
) -> ParsedGeneration:
    """Validate and split one generated stream.

    Validation is intentionally strict.  In particular, there must be exactly
    one delimiter of each kind, exactly two EOS tokens, the second EOS must be
    the final generated token, and only codec tokens may occur after the text
    EOS (apart from ``<s2st_eos>`` itself).  A malformed sample is recorded as
    an error instead of being silently converted into a plausible wav.
    """
    ids = [int(value) for value in generated_ids]
    if not ids:
        raise StructureError("empty generated sequence")
    if len({s2st_text_id, s2st_eos_id, eos_id}) != 3:
        raise ValueError("s2st_text, s2st_eos, and eos ids must be distinct")
    if codec_size <= 0:
        raise ValueError(f"codec_size must be positive, got {codec_size}")
    if min_codec_count < 0:
        raise ValueError(
            f"min_codec_count must be non-negative, got {min_codec_count}"
        )

    text_markers = _positions(ids, s2st_text_id)
    speech_end_markers = _positions(ids, s2st_eos_id)
    eos_positions = _positions(ids, eos_id)
    problems: list[str] = []
    if len(text_markers) != 1:
        problems.append(f"expected 1 <s2st_text>, found {len(text_markers)}")
    if len(speech_end_markers) != 1:
        problems.append(f"expected 1 <s2st_eos>, found {len(speech_end_markers)}")
    if len(eos_positions) != 2:
        problems.append(f"expected 2 EOS tokens, found {len(eos_positions)}")
    if problems:
        raise StructureError("; ".join(problems))

    s2st_text_pos = text_markers[0]
    text_eos_pos, final_eos_pos = eos_positions
    s2st_eos_pos = speech_end_markers[0]
    if not (s2st_text_pos < text_eos_pos < s2st_eos_pos < final_eos_pos):
        raise StructureError(
            "invalid marker order: expected <s2st_text> < text-EOS < "
            f"<s2st_eos> < final-EOS, got {s2st_text_pos}, "
            f"{text_eos_pos}, {s2st_eos_pos}, {final_eos_pos}"
        )
    if final_eos_pos != len(ids) - 1:
        raise StructureError(
            f"final EOS is at {final_eos_pos}, but sequence ends at {len(ids) - 1}"
        )

    codec_end = codec_offset + codec_size

    def is_codec(token_id: int) -> bool:
        return codec_offset <= token_id < codec_end

    codec_positions = [index for index, token_id in enumerate(ids) if is_codec(token_id)]
    invalid_codec_positions = [
        index
        for index in codec_positions
        if not (s2st_text_pos < index < final_eos_pos)
    ]
    if invalid_codec_positions:
        raise StructureError(
            "codec tokens found outside the translation/speech region at "
            f"positions {invalid_codec_positions[:10]}"
        )

    post_text_non_codec = [
        (index, ids[index])
        for index in range(text_eos_pos + 1, s2st_eos_pos)
        if not is_codec(ids[index])
    ]
    post_marker_non_codec = [
        (index, ids[index])
        for index in range(s2st_eos_pos + 1, final_eos_pos)
        if not is_codec(ids[index])
    ]
    if post_text_non_codec:
        raise StructureError(
            "non-codec tokens occur between text-EOS and <s2st_eos>: "
            f"{post_text_non_codec[:10]}"
        )
    if post_marker_non_codec:
        raise StructureError(
            "non-codec tokens occur between <s2st_eos> and final-EOS: "
            f"{post_marker_non_codec[:10]}"
        )

    speech_codes = [ids[index] - codec_offset for index in codec_positions]
    if len(speech_codes) < min_codec_count:
        raise StructureError(
            f"only {len(speech_codes)} codec tokens; minimum is {min_codec_count}"
        )
    if any(code < 0 or code >= codec_size for code in speech_codes):
        raise AssertionError("internal error: parsed codec code outside configured range")

    interleaved_before_text_eos = ids[s2st_text_pos + 1 : text_eos_pos]
    translation_ids = [
        token_id for token_id in interleaved_before_text_eos if not is_codec(token_id)
    ]
    asr_ids = ids[:s2st_text_pos]
    forbidden_asr = [
        token_id
        for token_id in asr_ids
        if is_codec(token_id) or token_id in (s2st_eos_id, eos_id)
    ]
    if forbidden_asr:
        raise StructureError(
            "ASR region contains codec/EOS markers: "
            f"{forbidden_asr[:10]}"
        )

    return ParsedGeneration(
        asr_ids=asr_ids,
        translation_ids=translation_ids,
        speech_codes=speech_codes,
        s2st_text_position=s2st_text_pos,
        text_eos_position=text_eos_pos,
        s2st_eos_position=s2st_eos_pos,
        final_eos_position=final_eos_pos,
        codec_count=len(speech_codes),
        generated_token_count=len(ids),
    )


def stable_seed(base_seed: int, name: str) -> int:
    """Return a process-independent torch-compatible seed derived from the name."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**31)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU torch, and all visible CUDA devices."""
    random.seed(seed)
    # Heavy dependencies stay lazy so the parser tests remain CPU-only.
    import numpy as np
    import torch

    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(value: object) -> str:
    return " ".join(str(value).replace("\t", " ").splitlines()).strip()


def wav_is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 44


# --------------------------------------------------------------------------- #
#  Model loading                                                               #
# --------------------------------------------------------------------------- #
def load_full_stack(args: argparse.Namespace, device: str):
    """Load the ASR/S2ST model and the token2wav stack."""
    import torch
    from models import (
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
        build_model,
    )

    token_root = Path(args.token2wav_root)
    token2wav_model, token2wav_frontend = build_model(
        flow_model=str(token_root / "flow.pt"),
        hift_model=str(token_root / "hift.pt"),
        campplus_model=str(token_root / "campplus.onnx"),
        speech_tokenizer_model=str(token_root / "speech_tokenizer_v3.onnx"),
        fp16=False,
    )
    token2wav_model.to(device)
    if hasattr(token2wav_model, "eval"):
        token2wav_model.eval()

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation=os.environ.get("attn_impl", "flash_attention_2"),
    )
    processor = Qwen3ASRProcessor.from_pretrained(
        args.base_model, fix_mistral_regex=True,
    )
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    model.thinker.resize_token_embeddings(SPEECH_CODEC_OFFSET + SPEECH_CODEC_SIZE)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {args.ckpt_path}")
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    if not state_dict:
        raise ValueError(f"empty model state dict in {args.ckpt_path}")
    load_message = model.load_state_dict(state_dict, strict=False)
    print(f"[{device}] load_state_dict: {load_message}", flush=True)
    model = model.to(device).eval()

    tokenizer = processor.tokenizer
    token_ids = {
        "s2st_text": tokenizer.convert_tokens_to_ids("<s2st_text>"),
        "s2st_eos": tokenizer.convert_tokens_to_ids("<s2st_eos>"),
        "pause": tokenizer.convert_tokens_to_ids("<pause>"),
        "eos": tokenizer.eos_token_id,
    }
    if len(set(token_ids.values())) != len(token_ids):
        raise RuntimeError(f"special token ids are not distinct: {token_ids}")
    return model, processor, token2wav_model, token2wav_frontend, token_ids


# --------------------------------------------------------------------------- #
#  Decoding helpers                                                            #
# --------------------------------------------------------------------------- #
class ASRGreedyProcessor:
    """Force greedy decoding before ``<s2st_text>``, then pass logits through."""

    def __init__(self, s2st_text_id: int, prompt_len: int):
        self.s2st_text_id = s2st_text_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        import torch

        for row in range(scores.shape[0]):
            generated = input_ids[row, self.prompt_len :]
            if (generated == self.s2st_text_id).any():
                continue
            top_id = scores[row].argmax().item()
            greedy_scores = torch.full_like(scores[row], float("-inf"))
            greedy_scores[top_id] = 0.0
            scores[row] = greedy_scores
        return scores


class CodecRepetitionPenaltyProcessor:
    """Apply repetition penalty only to generated codec ids."""

    def __init__(
        self, penalty: float, s2st_text_id: int, prompt_len: int,
        codec_offset: int = SPEECH_CODEC_OFFSET,
        codec_size: int = SPEECH_CODEC_SIZE,
    ):
        self.penalty = penalty
        self.s2st_text_id = s2st_text_id
        self.prompt_len = prompt_len
        self.codec_offset = codec_offset
        self.codec_end = codec_offset + codec_size

    def __call__(self, input_ids, scores):
        import torch

        if self.penalty == 1.0:
            return scores
        for row in range(scores.shape[0]):
            generated = input_ids[row, self.prompt_len :]
            if not (generated == self.s2st_text_id).any():
                continue
            unique = generated.unique()
            codec = unique[
                (unique >= self.codec_offset) & (unique < self.codec_end)
            ]
            if codec.numel() == 0:
                continue
            values = scores[row, codec]
            scores[row, codec] = torch.where(
                values > 0, values / self.penalty, values * self.penalty,
            )
        return scores


class StopAtSecondEos:
    """Stop batch-size-one inference once two generated EOS tokens exist."""

    def __init__(self, eos_id: int, prompt_len: int):
        self.eos_id = eos_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return all(
            int((row[self.prompt_len :] == self.eos_id).sum().item()) >= 2
            for row in input_ids
        )


# --------------------------------------------------------------------------- #
#  One sample                                                                  #
# --------------------------------------------------------------------------- #
def infer_one(
    *,
    model,
    processor,
    token2wav_model,
    token2wav_frontend,
    token_ids: dict[str, int],
    device: str,
    audio_path: Path,
    output_wav: Path,
    name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import librosa
    import torch
    import torchaudio
    from transformers import LogitsProcessorList, StoppingCriteriaList
    from models import prepare_prompt_conditioning, token2wav

    sample_seed = stable_seed(args.seed, name)
    seed_everything(sample_seed)
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]
    prompt = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False,
    )[0]
    inputs = processor(
        text=[prompt], audio=[audio], return_tensors="pt", padding=True,
    )
    inputs = {
        key: (
            value.to(device=device, dtype=torch.bfloat16)
            if value.is_floating_point()
            else value.to(device)
        )
        for key, value in inputs.items()
    }
    prompt_len = int(inputs["input_ids"].shape[1])
    processors: list[Any] = [
        ASRGreedyProcessor(token_ids["s2st_text"], prompt_len)
    ]
    if args.repetition_penalty != 1.0:
        processors.append(
            CodecRepetitionPenaltyProcessor(
                args.repetition_penalty, token_ids["s2st_text"], prompt_len,
            )
        )

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            eos_token_id=[],
            pad_token_id=processor.tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList(processors),
            stopping_criteria=StoppingCriteriaList(
                [StopAtSecondEos(token_ids["eos"], prompt_len)]
            ),
        )
    sequences = result.sequences if hasattr(result, "sequences") else result
    if int(sequences.shape[0]) != 1:
        raise RuntimeError(f"expected one generated sequence, got {sequences.shape[0]}")
    generated_ids = sequences[0, prompt_len:].tolist()
    parsed = parse_and_validate_generation(
        generated_ids,
        s2st_text_id=token_ids["s2st_text"],
        s2st_eos_id=token_ids["s2st_eos"],
        eos_id=token_ids["eos"],
        min_codec_count=args.min_codec_count,
    )

    tokenizer = processor.tokenizer
    asr_text_no_pause = clean_text(
        tokenizer.decode(parsed.asr_ids, skip_special_tokens=True)
    )
    asr_text_with_pause = clean_text(
        tokenizer.decode(parsed.asr_ids, skip_special_tokens=False)
    )
    translation_text = clean_text(
        tokenizer.decode(parsed.translation_ids, skip_special_tokens=True)
    )

    # Give prompt extraction and the vocoder their own deterministic sub-seed.
    seed_everything(stable_seed(sample_seed, "token2wav"))
    prompt_conditioning = prepare_prompt_conditioning(
        token2wav_frontend, str(audio_path),
    )
    speech_codes = torch.tensor(
        parsed.speech_codes, dtype=torch.long, device="cpu",
    ).unsqueeze(0)
    with torch.inference_mode():
        waveform = token2wav(
            token2wav_model,
            speech_codes,
            prompt_conditioning=prompt_conditioning,
        )
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] <= 0:
        raise RuntimeError(f"token2wav returned invalid shape {tuple(waveform.shape)}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temporary_wav = output_wav.with_name(
        f".{output_wav.stem}.{os.getpid()}.tmp.wav"
    )
    try:
        torchaudio.save(str(temporary_wav), waveform.detach().cpu(), OUTPUT_SAMPLE_RATE)
        if not wav_is_valid(temporary_wav):
            raise RuntimeError(f"saved wav is empty or invalid: {temporary_wav}")
        os.replace(temporary_wav, output_wav)
    finally:
        temporary_wav.unlink(missing_ok=True)

    parsed_dict = asdict(parsed)
    # Codec contents are not needed after vocoding; keep the record compact.
    parsed_dict.pop("speech_codes")
    parsed_dict.pop("translation_ids")
    return {
        "status": "ok",
        "structure_valid": True,
        "seed": sample_seed,
        "asr_ids": parsed.asr_ids,
        "asr_pause_count": sum(
            token_id == token_ids["pause"] for token_id in parsed.asr_ids
        ),
        "asr_text_no_pause": asr_text_no_pause,
        "asr_text_with_pause": asr_text_with_pause,
        "translation_text": translation_text,
        "wav_relpath": str(output_wav.relative_to(Path(args.out_dir).resolve())),
        "wav_num_frames": int(waveform.shape[1]),
        "wav_sample_rate": OUTPUT_SAMPLE_RATE,
        **parsed_dict,
    }


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def write_translated_texts(out_dir: Path) -> int:
    """Regenerate ``translated_texts.txt`` (latest successful record per name)."""
    latest: dict[str, dict[str, Any]] = {}
    records_path = out_dir / "predictions.jsonl"
    if records_path.is_file():
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["name"]] = row
    rows = [row for row in latest.values() if row.get("status") == "ok"]
    with (out_dir / "translated_texts.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['name']}\t{row['translation_text']}\n")
    return len(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True,
                        help="Lightning .ckpt (SFT) or GSPO .pt checkpoint")
    parser.add_argument("--audio", nargs="+", required=True,
                        help="input English speech file(s); any format librosa can read")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--token2wav_root", default=DEFAULT_TOKEN2WAV_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=896)
    parser.add_argument("--min_codec_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        if not Path(audio).is_file():
            raise FileNotFoundError(f"input audio not found: {audio}")

    model, processor, t2w_model, t2w_frontend, token_ids = load_full_stack(args, args.device)

    failures = 0
    with (out_dir / "predictions.jsonl").open("a", encoding="utf-8") as records:
        for audio in args.audio:
            audio_path = Path(audio).resolve()
            name = audio_path.stem
            output_wav = out_dir / "audio" / f"{name}.wav"
            output_wav.unlink(missing_ok=True)  # never pair a stale wav with a new record
            started = time.time()
            try:
                row = infer_one(
                    model=model, processor=processor,
                    token2wav_model=t2w_model, token2wav_frontend=t2w_frontend,
                    token_ids=token_ids, device=args.device,
                    audio_path=audio_path, output_wav=output_wav, name=name, args=args,
                )
                row.update({"name": name, "input_audio": str(audio_path),
                            "seconds": round(time.time() - started, 2)})
                duration = row["wav_num_frames"] / row["wav_sample_rate"]
                print(f"[{name}] ASR : {row['asr_text_with_pause']}")
                print(f"[{name}] ZH  : {row['translation_text']}")
                print(f"[{name}] wav : {output_wav}  ({duration:.2f}s, "
                      f"{row['codec_count']} codec tokens, {row['seconds']}s)")
            except StructureError as exc:
                failures += 1
                row = {"name": name, "input_audio": str(audio_path), "status": "error",
                       "structure_valid": False, "error_type": "StructureError",
                       "error": str(exc)}
                print(f"[{name}] ERROR: malformed generation ({exc})", file=sys.stderr)
            records.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.flush()

    n_ok = write_translated_texts(out_dir)
    print(f"done: {n_ok} translation(s) in {out_dir / 'translated_texts.txt'}, "
          f"audio in {out_dir / 'audio'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
