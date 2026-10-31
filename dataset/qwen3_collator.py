from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import torch


def load_audio(path, sr: int = 16000):
    # Multi-audio samples are pre-mixed (segments concatenated with silence) in
    # the dataset and arrive as a ready waveform array; pass those through.
    # Single-audio samples arrive as a file path and are loaded here.
    if isinstance(path, np.ndarray):
        return path
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


class Qwen3ASRCollator:
    """Load audio, build interleaved text+speech sequences, and mask prefix tokens in labels."""

    TEXT_CHUNK = 5
    SPEECH_CHUNK = 10

    def __init__(self, processor, sampling_rate: int = 16000):
        self.processor = processor
        self.sampling_rate = sampling_rate

        tok = processor.tokenizer
        tok.padding_side = "left"

        self.eos_token_id = tok.eos_token_id
        self.pad_token_id = tok.pad_token_id
        self.s2st_eos_id = tok.convert_tokens_to_ids("<s2st_eos>")

    # -------------------------------------------------------------- #
    #  Interleaving                                                    #
    # -------------------------------------------------------------- #

    @staticmethod
    def _interleave_with_speech_positions(
        text_ids: List[int],
        speech_ids: List[int],
        text_chunk: int,
        speech_chunk: int,
    ) -> Tuple[List[int], int, List[int]]:
        """Interleave tokens and retain each consumed codec token's position.

        The returned positions are ordered by codec index: element ``k`` is
        the position in ``mixed`` occupied by ``speech_ids[k]``.  Keeping this
        mapping explicitly is important because codec ids may repeat, and
        pause-region supervision must remain correct on both sides of the
        interleaving boundary.
        """
        mixed: List[int] = []
        speech_positions: List[int] = []
        ti, si = 0, 0

        while ti < len(text_ids):
            end = min(ti + text_chunk, len(text_ids))
            mixed.extend(text_ids[ti:end])
            n = end - ti
            ti = end

            if si < len(speech_ids) and (ti < len(text_ids) or n == text_chunk):
                s_end = min(si + speech_chunk, len(speech_ids))
                speech_positions.extend(range(len(mixed), len(mixed) + s_end - si))
                mixed.extend(speech_ids[si:s_end])
                si = s_end

        return mixed, si, speech_positions

    @staticmethod
    def _interleave(
        text_ids: List[int],
        speech_ids: List[int],
        text_chunk: int,
        speech_chunk: int,
    ) -> Tuple[List[int], int]:
        """Round-robin merge of *text_chunk* text and *speech_chunk* codec tokens."""
        mixed, consumed, _ = Qwen3ASRCollator._interleave_with_speech_positions(
            text_ids, speech_ids, text_chunk, speech_chunk,
        )
        return mixed, consumed

    def _build_target_ids_with_speech_positions(
        self, target_text: str, speech_codes: torch.Tensor | None,
    ) -> Tuple[List[int], List[int]]:
        """Build a target and map every codec index to its target-local position."""
        tok = self.processor.tokenizer
        text_ids = tok(
            target_text + tok.eos_token, add_special_tokens=False,
        )["input_ids"]

        if speech_codes is None:
            return text_ids + [self.s2st_eos_id, self.eos_token_id], []

        speech_ids = [int(c) for c in speech_codes.tolist()]
        mixed, consumed, speech_positions = self._interleave_with_speech_positions(
            text_ids, speech_ids, self.TEXT_CHUNK, self.SPEECH_CHUNK,
        )
        mixed.append(self.s2st_eos_id)
        speech_positions.extend(
            range(len(mixed), len(mixed) + len(speech_ids) - consumed)
        )
        mixed.extend(speech_ids[consumed:])
        mixed.append(self.eos_token_id)
        if len(speech_positions) != len(speech_ids):
            raise RuntimeError(
                "internal error: target builder lost codec positions "
                f"({len(speech_positions)} != {len(speech_ids)})"
            )
        return mixed, speech_positions

    def _build_target_ids(
        self, target_text: str, speech_codes: torch.Tensor | None,
    ) -> List[int]:
        """Encode one sample's target: interleaved text+speech -> <s2st_eos> -> remaining speech -> <eos>."""
        target_ids, _ = self._build_target_ids_with_speech_positions(
            target_text, speech_codes,
        )
        return target_ids

    @staticmethod
    def _pause_code_indices(
        spans: Any, num_speech_codes: int,
    ) -> List[int]:
        """Validate half-open codec spans and return their unique indices.

        Dataset rows use ``target_pause_code_spans`` relative to the original
        target codec stream, before text/speech interleaving.  A span may be a
        ``(start, end)`` pair or a mapping with ``start_code``/``end_code``.
        """
        if spans is None:
            return []
        selected: set[int] = set()
        for span in spans:
            if isinstance(span, dict):
                start = span.get("start_code")
                end = span.get("end_code")
            else:
                try:
                    start, end = span
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "target_pause_code_spans entries must be (start, end) "
                        "pairs or {start_code, end_code} mappings"
                    ) from exc
            if (
                isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, np.integer))
                or not isinstance(end, (int, np.integer))
            ):
                raise TypeError(
                    "target pause codec span boundaries must be integers, got "
                    f"({start!r}, {end!r})"
                )
            start, end = int(start), int(end)
            if not (0 <= start < end <= num_speech_codes):
                raise ValueError(
                    "target pause codec span is outside the target codec stream: "
                    f"[{start}, {end}) vs {num_speech_codes} codes"
                )
            selected.update(range(start, end))
        return sorted(selected)

    # -------------------------------------------------------------- #
    #  Collate                                                         #
    # -------------------------------------------------------------- #

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [load_audio(f["audio"], sr=self.sampling_rate) for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        source_texts = [f["source_text"] for f in features]
        context_texts = [p + s for p, s in zip(prefix_texts, source_texts)]

        ctx = self.processor(
            text=context_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        pfx = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_lens = pfx["attention_mask"].sum(dim=1).tolist()

        composed: List[List[int]] = []
        context_lens: List[int] = []
        loss_modes: List[str] = []
        sample_loss_weights: List[float] = []
        # Unpadded positions in each complete (context + target) sequence.
        pause_speech_positions: List[List[int]] = []
        for i, feat in enumerate(features):
            seq_len = int(ctx["attention_mask"][i].sum().item())
            ctx_ids = ctx["input_ids"][i, -seq_len:].tolist()
            loss_mode = feat.get("loss_mode", "all")
            if loss_mode == "asr_only":
                # The source context already ends in <s2st_text>.  Ending the
                # sequence here makes the supervision unambiguously ASR-only
                # and avoids wasting compute on masked dummy target tokens.
                tgt_ids = []
                speech_positions: List[int] = []
            elif loss_mode == "text_only":
                # AST rows: translated text then <eos> — no <s2st_eos> and no
                # speech codes.  The ASR transcript lives in the (unmasked)
                # source context, so each row supervises ASR + AST jointly.
                tok = self.processor.tokenizer
                tgt_ids = tok(
                    feat["target_text"] + tok.eos_token,
                    add_special_tokens=False,
                )["input_ids"]
                speech_positions = []
            elif loss_mode == "all":
                tgt_ids, speech_positions = self._build_target_ids_with_speech_positions(
                    feat["target_text"], feat["target_speech_code"],
                )
            else:
                raise ValueError(f"Unsupported SFT loss_mode: {loss_mode!r}")
            composed.append(ctx_ids + tgt_ids)
            context_lens.append(len(ctx_ids))
            loss_modes.append(loss_mode)
            sample_loss_weights.append(float(feat.get("sample_loss_weight", 1.0)))
            pause_code_indices = self._pause_code_indices(
                feat.get("target_pause_code_spans"), len(speech_positions),
            )
            pause_speech_positions.append([
                len(ctx_ids) + speech_positions[code_index]
                for code_index in pause_code_indices
            ])

        padded = self.processor.tokenizer.pad(
            {"input_ids": composed},
            padding=True, return_tensors="pt", padding_side="left",
        )
        ctx["input_ids"] = padded["input_ids"]
        ctx["attention_mask"] = padded["attention_mask"]

        labels = ctx["input_ids"].clone()
        pause_speech_mask = torch.zeros_like(labels, dtype=torch.bool)
        width = labels.size(1)
        for i, pl in enumerate(prefix_lens):
            seq_len = int(ctx["attention_mask"][i].sum().item())
            start = width - seq_len
            labels[i, start : start + int(pl)] = -100
            if loss_modes[i] == "asr_only":
                labels[i, start + context_lens[i] :] = -100
            for position in pause_speech_positions[i]:
                pause_speech_mask[i, start + position] = True
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        ctx["labels"] = labels
        ctx["pause_speech_mask"] = pause_speech_mask
        ctx["sample_loss_weights"] = torch.tensor(
            sample_loss_weights, dtype=torch.float32,
        )
        ctx["loss_modes"] = loss_modes
        return ctx
