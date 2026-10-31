import bisect
import math

import numpy as np
from .base_dataset import BaseDataset
from .silence import concat_audios_with_silence, concat_audios_with_random_silence

SAMPLE_RATE = 16000


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

class T2STDataset(BaseDataset):
    def __init__(
        self,
        config,
        split,
        processor=None,
        paths=None,
        target_pause_span_source="none",
    ):
        super().__init__(config, split)
        self.split = split
        self.processor = processor
        self.sampling_rate = getattr(config, "sampling_rate", SAMPLE_RATE)
        # Select the pause synthesis style used by the target dataset.
        self.pause_audio_style = (
            getattr(config, "pause_audio_style", "legacy") or "legacy"
        )
        if self.pause_audio_style not in {"legacy", "clean"}:
            raise ValueError(
                "pause_audio_style must be 'legacy' or 'clean', got "
                f"{self.pause_audio_style!r}"
            )
        # Optional pause marker (e.g. "<pause>") inserted between the segment
        # transcripts of multi-audio rows so the ASR text carries an explicit,
        # text-level anchor for the inserted source silence.  Must also be
        # registered in model.special_tokens (appended AFTER the existing
        # specials to preserve their token IDs).
        self.pause_token = getattr(config, "pause_token", None) or None
        # Source-side pause markers remain part of the ASR supervision.  The
        # target marker is optional because the global fixed-ratio text/speech
        # interleaver cannot place it at the true target codec boundary.
        self.add_pause_to_target_text = bool(
            getattr(config, "add_pause_to_target_text", True)
        )
        # Priority-2 pause supervision is intentionally opt-in per dataset.
        # Only parquets with an audited target pause interval should enable it;
        # no VAD or codec-ID heuristic is run in the training workers.
        self.target_pause_span_source = target_pause_span_source
        if self.target_pause_span_source not in {
            "none", "inserted", "audited", "audited_multi",
        }:
            raise ValueError(
                "target_pause_span_source must be one of "
                "'none', 'inserted', 'audited', or 'audited_multi'"
            )
        self.supervise_target_pause_speech = (
            self.target_pause_span_source != "none"
        )
        self.pause_speech_boundary_trim_s = float(
            getattr(config, "pause_speech_boundary_trim_s", 0.08)
        )
        self.speech_codec_rate_hz = float(
            getattr(config, "speech_codec_rate_hz", 25.0)
        )
        if self.pause_speech_boundary_trim_s < 0.0:
            raise ValueError("pause_speech_boundary_trim_s must be >= 0")
        if self.speech_codec_rate_hz <= 0.0:
            raise ValueError("speech_codec_rate_hz must be > 0")

        parquet_paths = paths
        if parquet_paths is None:
            parquet_paths = (
                config.train_parquet_dir
                if split == "train"
                else config.dev_parquet_dir
            )

        requested_columns = [
            "en_text", "en_path", "en_length",
            "zh_text", "zh_codes", "zh_length", "silence_length",
        ]
        if self.target_pause_span_source == "audited":
            requested_columns.extend([
                "post_target_pause_start_sample",
                "post_target_pause_end_sample",
                "rendered_num_samples",
                "sample_rate",
            ])
        elif self.target_pause_span_source == "audited_multi":
            # Multi-pause Parquet rows carry every target pause of the clip in
            # target_pause_spans_s (list<struct<start_s, end_s>>, seconds on
            # the rendered target timeline).
            requested_columns.extend([
                "target_pause_spans_s",
                "rendered_num_samples",
                "sample_rate",
            ])

        cols = self.load_parquet_columns(
            paths=parquet_paths,
            columns=requested_columns,
        )
        # self.en_codes = cols["en_codes"]
        self.en_texts = cols["en_text"]
        self.en_paths = cols["en_path"]
        self.en_lengths = cols["en_length"]

        # self.zh_paths = cols["zh_path"]
        self.zh_lengths = cols["zh_length"]
        self.zh_texts = cols["zh_text"]
        self.zh_codes = cols["zh_codes"]
        self.silence_lengths = cols["silence_length"]

        if self.target_pause_span_source == "audited":
            self.target_pause_start_samples = cols[
                "post_target_pause_start_sample"
            ]
            self.target_pause_end_samples = cols[
                "post_target_pause_end_sample"
            ]
            self.target_rendered_num_samples = cols["rendered_num_samples"]
            self.target_sample_rates = cols["sample_rate"]
            required_pause_columns = (
                self.target_pause_start_samples,
                self.target_pause_end_samples,
                self.target_rendered_num_samples,
                self.target_sample_rates,
            )
            if not self.target_pause_start_samples or any(
                value is None
                for column in required_pause_columns
                for value in column
            ):
                raise ValueError(
                    "Paired pause-speech supervision requires "
                    "post_target_pause_start_sample/end_sample, "
                    "rendered_num_samples, and sample_rate on every row."
                )
        elif self.target_pause_span_source == "audited_multi":
            self.target_pause_spans_s = cols["target_pause_spans_s"]
            self.target_rendered_num_samples = cols["rendered_num_samples"]
            self.target_sample_rates = cols["sample_rate"]
            if not self.target_pause_spans_s or any(
                spans is None or len(spans) == 0
                or self.target_rendered_num_samples[i] is None
                or self.target_sample_rates[i] is None
                for i, spans in enumerate(self.target_pause_spans_s)
            ):
                raise ValueError(
                    "audited_multi pause supervision requires a non-empty "
                    "target_pause_spans_s plus rendered_num_samples and "
                    "sample_rate on every row."
                )

        # Two supported parquet layouts:
        #   single-audio: scalar columns, one audio/sample.
        #   multi-audio: list columns grouped by speaker,
        #     where a sample concatenates 1+ source audios with a generated
        #     silence of `silence_length` seconds between them; `zh_codes` is a
        #     single binary already covering the combined target (kept as-is).
        # Detected from the column type: list<> for multi-audio, scalar for
        # single-audio.
        self.is_multi_audio = (
            len(self.en_paths) > 0 and isinstance(self.en_paths[0], list)
        )

        if self.is_multi_audio:
            self._init_multi_audio()
        else:
            self.lengths = self.en_lengths

        # self.normalizer = BasicTextNormalizer()
        # self.english_normalizer = EnglishTextNormalizer()

    def _init_multi_audio(self):
        """Normalize the list-typed multi-audio schema.

        Computes per-sample combined durations (sum of segment lengths +
        silence) for the dynamic batch sampler, and drops samples whose
        combined source or target duration falls outside [2, 20]s — mirroring
        the scalar pushdown filter applied to single-audio parquet in
        `load_parquet_columns` (which is skipped for list columns).
        """
        keep, lengths = [], []
        for i in range(len(self.en_paths)):
            sil = self.silence_lengths[i] or 0.0
            en_total = sum(self.en_lengths[i]) + sil
            zh_total = sum(self.zh_lengths[i]) + sil
            if 2.0 <= en_total <= 20.0 and 2.0 <= zh_total <= 20.0:
                keep.append(i)
                lengths.append(en_total)

        def _take(lst):
            return [lst[i] for i in keep]

        self.en_texts = _take(self.en_texts)
        self.en_paths = _take(self.en_paths)
        self.en_lengths = _take(self.en_lengths)
        self.zh_lengths = _take(self.zh_lengths)
        self.zh_texts = _take(self.zh_texts)
        self.zh_codes = _take(self.zh_codes)
        self.silence_lengths = _take(self.silence_lengths)
        if self.target_pause_span_source == "audited":
            self.target_pause_start_samples = _take(
                self.target_pause_start_samples
            )
            self.target_pause_end_samples = _take(
                self.target_pause_end_samples
            )
            self.target_rendered_num_samples = _take(
                self.target_rendered_num_samples
            )
            self.target_sample_rates = _take(self.target_sample_rates)
        elif self.target_pause_span_source == "audited_multi":
            self.target_pause_spans_s = _take(self.target_pause_spans_s)
            self.target_rendered_num_samples = _take(
                self.target_rendered_num_samples
            )
            self.target_sample_rates = _take(self.target_sample_rates)
        self.lengths = lengths

    def __len__(self):
        return len(self.en_paths)

    def _target_pause_code_spans(self, idx, num_codes):
        """Map known target pause intervals from samples to codec indices.

        The returned interval is half-open and relative to
        ``target_speech_code`` before text/speech interleaving.  Trimming both
        boundaries keeps cross-fades and adjacent phonemes out of the
        high-confidence silence supervision.
        """
        if self.target_pause_span_source == "none":
            return []

        if self.target_pause_span_source == "audited":
            sample_rate = int(self.target_sample_rates[idx])
            total_samples = int(self.target_rendered_num_samples[idx])
            sample_spans = [(
                int(self.target_pause_start_samples[idx]),
                int(self.target_pause_end_samples[idx]),
            )]
        elif self.target_pause_span_source == "audited_multi":
            # Every pause of the clip is supervised. Spans are seconds on
            # the rendered target timeline (isometric with the source).
            sample_rate = int(self.target_sample_rates[idx])
            total_samples = int(self.target_rendered_num_samples[idx])
            sample_spans = [
                (
                    int(round(float(span["start_s"]) * sample_rate)),
                    int(round(float(span["end_s"]) * sample_rate)),
                )
                for span in self.target_pause_spans_s[idx]
            ]
        else:
            # The clean two-utterance target was encoded from exactly this
            # timeline.  crossfade_join preserves length, and the parquet's
            # segment durations are sample-exact, so these boundaries are
            # known without VAD.  Single-audio rows correctly yield no span.
            sample_rate = int(self.sampling_rate)
            lengths = self.zh_lengths[idx]
            if not isinstance(lengths, list) or len(lengths) <= 1:
                return []
            pause_samples = int(round(
                float(self.silence_lengths[idx] or 0.0) * sample_rate
            ))
            if pause_samples <= 0:
                return []
            segment_samples = [
                int(round(float(length) * sample_rate)) for length in lengths
            ]
            sample_spans = []
            cursor = segment_samples[0]
            for segment_num_samples in segment_samples[1:]:
                sample_spans.append((cursor, cursor + pause_samples))
                cursor += pause_samples + segment_num_samples
            total_samples = cursor

        trim_samples = int(round(
            self.pause_speech_boundary_trim_s * sample_rate
        ))
        code_spans = []
        for start_sample, end_sample in sample_spans:
            inner_start = start_sample + trim_samples
            inner_end = end_sample - trim_samples
            if not (
                0 <= start_sample < end_sample <= total_samples
                and sample_rate > 0
                and num_codes > 0
            ):
                raise ValueError(
                    "Invalid target pause interval: "
                    f"start={start_sample}, end={end_sample}, "
                    f"total={total_samples}, sr={sample_rate}, codes={num_codes}"
                )
            if inner_end <= inner_start:
                raise ValueError(
                    "pause_speech_boundary_trim_s removes the entire target "
                    f"pause: duration={(end_sample - start_sample) / sample_rate:.3f}s, "
                    f"trim={self.pause_speech_boundary_trim_s:.3f}s per side"
                )

            # Codec frames are anchored to the start of the utterance.  Use
            # ceil/floor so only frames fully inside the trimmed pause are
            # selected; clamp handles the tokenizer's possible tail frame.
            code_start = math.ceil(
                inner_start * self.speech_codec_rate_hz / sample_rate
            )
            code_end = math.floor(
                inner_end * self.speech_codec_rate_hz / sample_rate
            )
            code_start = max(0, min(num_codes, code_start))
            code_end = max(0, min(num_codes, code_end))
            if code_end <= code_start:
                raise ValueError(
                    "Target pause maps to no interior codec tokens: "
                    f"sample_span=({inner_start}, {inner_end}), codes={num_codes}"
                )
            code_spans.append((code_start, code_end))
        return code_spans

    def _build_item(
        self,
        audio,
        source_text,
        target_text,
        target_speech_code,
        target_pause_code_spans=None,
    ):
        prompt = ""
        prefix_msgs = build_prefix_messages(prompt)
        prefix_text = self.processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": audio,
            "prefix_text": prefix_text,
            "source_text": source_text,
            "target_text": target_text,
            "target_speech_code": target_speech_code,
            "target_pause_code_spans": target_pause_code_spans or [],
            "loss_mode": "all",
            "sample_loss_weight": 1.0,
        }

    def __getitem__(self, idx):
        if self.is_multi_audio:
            return self._getitem_multi(idx)

        source_path = self.en_paths[idx]
        source_text = self.en_texts[idx].strip()
        # source_text = self.english_normalizer(source_text)
        source_text = f"language English<asr_text>{source_text}<s2st_text>"

        target_text = self.zh_texts[idx].strip()
        target_speech_code = np.frombuffer(self.zh_codes[idx], dtype=np.uint16)
        target_speech_code = target_speech_code.astype(np.uint32) + 151936
        if abs(len(target_speech_code) / 25 - self.zh_lengths[idx]) > 0.1:
            print(source_path, self.zh_lengths[idx], len(target_speech_code) / 25)
        # target_speech_code = None

        return self._build_item(
            source_path,
            source_text,
            target_text,
            target_speech_code,
            self._target_pause_code_spans(idx, len(target_speech_code)),
        )

    def _getitem_multi(self, idx):
        """Multi-audio sample: concatenate the segment audios (silence between)
        and join the per-segment texts.

        - source_text : "text A" + " " + "text B"  (single-space join), or
                        "text A <pause> text B" when ``pause_token`` is set
                        and a silence was inserted between the segments
        - target_text : "text A" + "text B"; optionally uses the same
                        ``<pause>`` join when ``add_pause_to_target_text`` is
                        enabled
        - audio       : seg0 + silence + seg1 + ...  (mixed in the dataloader)
        - zh_codes    : kept as-is (already the combined target speech codes)
        """
        en_texts = [(t or "").strip() for t in self.en_texts[idx]]
        zh_texts = [(t or "").strip() for t in self.zh_texts[idx]]
        has_pause = (self.silence_lengths[idx] or 0.0) > 0.0 and len(en_texts) > 1
        if self.pause_token and has_pause:
            source_text = f" {self.pause_token} ".join(en_texts)
        else:
            source_text = " ".join(en_texts)
        source_text = f"language English<asr_text>{source_text}<s2st_text>"
        # The source ASR marker still conditions target generation.  Keep the
        # marker out of translated text when requested; with the current global
        # 5-text/10-code interleaver it otherwise appears much earlier than the
        # actual silence span in the target speech codes.
        if self.pause_token and has_pause and self.add_pause_to_target_text:
            target_text = f" {self.pause_token} ".join(zh_texts)
        else:
            target_text = "".join(zh_texts)

        target_speech_code = np.frombuffer(self.zh_codes[idx], dtype=np.uint16)
        target_speech_code = target_speech_code.astype(np.uint32) + 151936
        sil = self.silence_lengths[idx] or 0.0
        zh_total = sum(self.zh_lengths[idx]) + sil
        if abs(len(target_speech_code) / 25 - zh_total) > 0.1:
            print(self.en_paths[idx], zh_total, len(target_speech_code) / 25)

        audio = concat_audios_with_silence(
            self.en_paths[idx], silence_length=sil,
            sr=self.sampling_rate, seed=idx,
            pause_style=self.pause_audio_style,
        )

        return self._build_item(
            audio,
            source_text,
            target_text,
            target_speech_code,
            self._target_pause_code_spans(idx, len(target_speech_code)),
        )


class ASROnlyPauseDataset(BaseDataset):
    """Real single-audio pause examples with ASR-only supervision.

    Input parquets must contain a text column (``pause_text`` by default) in
    which real, VAD-confirmed pause locations have already been aligned to word
    boundaries and rendered as ``<pause>``.  No translated text or target
    speech is required.  ``Qwen3ASRCollator`` stops the sequence at
    ``<s2st_text>`` for these rows, so the loss covers only the ASR transcript
    (including the pause markers).

    Pause labels are expected to be prepared offline: this dataset reads the
    supplied transcript as-is and performs no VAD, alignment, or marker
    insertion.
    """

    def __init__(self, config, split, processor=None):
        super().__init__(config, split)
        self.processor = processor
        self.pause_token = getattr(config, "pause_token", None) or "<pause>"
        self.text_column = getattr(config, "asr_only_text_column", "pause_text")
        self.require_pause = bool(getattr(config, "asr_only_require_pause", True))
        self.sample_loss_weight = float(
            getattr(config, "asr_only_loss_weight", 1.0)
        )

        path_key = (
            "asr_only_train_parquet_dir"
            if split == "train"
            else "asr_only_dev_parquet_dir"
        )
        paths = list(getattr(config, path_key, []) or [])
        if not paths:
            raise ValueError(f"{path_key} is empty")

        cols = self.load_parquet_columns(
            paths=paths,
            columns=["en_path", "en_length", self.text_column],
        )
        raw_texts = cols[self.text_column]
        if not any(text is not None for text in raw_texts):
            raise ValueError(
                f"ASR-only pause parquet is missing required column "
                f"{self.text_column!r}. Prepare this transcript column "
                "offline with <pause> already inserted."
            )

        self.audio_paths = []
        self.pause_texts = []
        self.lengths = []
        for path_entry, length_entry, text_entry in zip(
            cols["en_path"], cols["en_length"], raw_texts,
        ):
            # The real-pause parquets use list columns for schema compatibility,
            # but each ASR-only row must refer to exactly one continuous file.
            if isinstance(path_entry, list):
                if len(path_entry) != 1:
                    continue
                audio_path = path_entry[0]
            else:
                audio_path = path_entry

            if isinstance(length_entry, list):
                if len(length_entry) != 1:
                    continue
                raw_duration = length_entry[0]
            else:
                raw_duration = length_entry
            if raw_duration is None:
                continue
            duration = float(raw_duration)

            if isinstance(text_entry, list):
                if len(text_entry) != 1:
                    continue
                pause_text = text_entry[0]
            else:
                pause_text = text_entry
            pause_text = (pause_text or "").strip()

            if not audio_path or not pause_text or not (2.0 <= duration <= 20.0):
                continue
            if self.require_pause and self.pause_token not in pause_text:
                continue

            self.audio_paths.append(audio_path)
            self.pause_texts.append(pause_text)
            self.lengths.append(duration)

        if not self.audio_paths:
            raise ValueError(
                "No usable ASR-only pause rows remained after requiring one "
                "continuous 2-20s audio and an aligned pause marker."
            )

        prefix_msgs = build_prefix_messages("")
        self._prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False,
        )[0]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        source_text = (
            f"language English<asr_text>{self.pause_texts[idx]}<s2st_text>"
        )
        return {
            "prompt": "",
            "audio": self.audio_paths[idx],
            "prefix_text": self._prefix_text,
            "source_text": source_text,
            # ASR-only rows end at <s2st_text>; these placeholders are never
            # appended by the collator and therefore receive no loss.
            "target_text": "",
            "target_speech_code": None,
            "target_pause_code_spans": [],
            "loss_mode": "asr_only",
            "sample_loss_weight": self.sample_loss_weight,
        }


class ASTTextDataset(BaseDataset):
    """Audio -> (ASR transcript + translated text) rows, no target speech.

    Stage-1 curriculum data: teaches listening + translation before any
    speech generation.  Parquet columns: en_path / en_text / en_length /
    zh_text (scalar strings; length-1 lists also accepted).  The collator
    appends the translation and <eos> only (loss_mode="text_only"); the
    English transcript sits in the unmasked source context, so ASR and AST
    are supervised jointly by every row.
    """

    def __init__(self, config, split, processor=None):
        super().__init__(config, split)
        self.processor = processor
        self.sample_loss_weight = float(
            getattr(config, "ast_text_loss_weight", 1.0)
        )

        path_key = (
            "ast_text_train_parquet_dir"
            if split == "train"
            else "ast_text_dev_parquet_dir"
        )
        paths = list(getattr(config, path_key, []) or [])
        if not paths:
            raise ValueError(f"{path_key} is empty")

        cols = self.load_parquet_columns(
            paths=paths,
            columns=["en_path", "en_text", "en_length", "zh_text"],
        )

        def _scalar(entry):
            if isinstance(entry, list):
                return entry[0] if len(entry) == 1 else None
            return entry

        self.audio_paths = []
        self.en_texts = []
        self.zh_texts = []
        self.lengths = []
        for path_e, en_e, len_e, zh_e in zip(
            cols["en_path"], cols["en_text"], cols["en_length"],
            cols["zh_text"],
        ):
            audio_path = _scalar(path_e)
            en_text = (_scalar(en_e) or "").strip()
            zh_text = (_scalar(zh_e) or "").strip()
            raw_duration = _scalar(len_e)
            if raw_duration is None:
                continue
            duration = float(raw_duration)
            if not audio_path or not en_text or not zh_text \
                    or not (2.0 <= duration <= 20.0):
                continue
            self.audio_paths.append(audio_path)
            self.en_texts.append(en_text)
            self.zh_texts.append(zh_text)
            self.lengths.append(duration)

        if not self.audio_paths:
            raise ValueError(
                "No usable AST text rows remained after requiring one "
                "continuous 2-20s audio with non-empty en/zh text."
            )

        prefix_msgs = build_prefix_messages("")
        self._prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False,
        )[0]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        source_text = (
            f"language English<asr_text>{self.en_texts[idx]}<s2st_text>"
        )
        return {
            "prompt": "",
            "audio": self.audio_paths[idx],
            "prefix_text": self._prefix_text,
            "source_text": source_text,
            "target_text": self.zh_texts[idx],
            "target_speech_code": None,
            "target_pause_code_spans": [],
            "loss_mode": "text_only",
            "sample_loss_weight": self.sample_loss_weight,
        }


class MixedSFTDataset:
    """Concatenate SFT datasets while exposing aligned duration hints.

    ``DynamicBatchSampler`` requires a single ``lengths`` list.  Logical
    repetition lets the ASR-only corpus be up/down-sampled without copying its
    in-memory columns.
    """

    def __init__(self, datasets, repeats=None):
        repeats = repeats or [1] * len(datasets)
        if len(datasets) != len(repeats):
            raise ValueError("datasets and repeats must have the same length")

        self._segments = []
        self._ends = []
        self.lengths = []
        total = 0
        for dataset, repeat in zip(datasets, repeats):
            for _ in range(int(repeat)):
                if len(dataset) == 0:
                    continue
                self._segments.append(dataset)
                total += len(dataset)
                self._ends.append(total)
                self.lengths.extend(dataset.lengths)
        if not self._segments:
            raise ValueError("MixedSFTDataset needs at least one non-empty dataset")

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        segment = bisect.bisect_right(self._ends, idx)
        start = 0 if segment == 0 else self._ends[segment - 1]
        return self._segments[segment][idx - start]


class T2STRLDataset(BaseDataset):
    """Lightweight dataset for GRPO / RL training.

    Loads source audio paths, durations, source transcript and target
    reference translation from parquet — no target speech codes, since RL
    generates its own speech.  `source_text` is the source-side transcript
    used as the "source" input to translation-quality reward models (so ASR
    errors don't leak into the reward); `target_text` is the human reference
    translation used by reference-based metrics like MetricX.

    Two supported parquet layouts (detected from `en_path`'s column type):

    * single-audio (scalar columns): one audio/sample; `text` is the source
      transcript, `vad` carries precomputed source-silence intervals.
    * multi-audio (list columns grouped by speaker): a sample
      concatenates 2+ same-speaker source audios with a freshly generated
      pseudo-silence between each consecutive pair (duration drawn per gap
      from `Uniform(min_silence, max_silence)`). `source_text` joins the
      per-segment English transcripts with single spaces, `target_text`
      concatenates the per-segment Chinese references, and `source_vad` is
      built from the inserted silences: one `(start, end)` interval per gap,
      positioned at the true segment boundary and lengthened by
      `source_vad_offset` seconds (split ±half on each side) so it matches the
      slightly-wider pause the VAD reports on decoded speech. The waveform is
      mixed here and passed through the collator as a ready array.

      Single-element rows using the list schema are valid: nothing is
      inserted, the lone audio
      passes through unchanged and `source_vad` is emitted empty. The
      pause_only reward then detects the source's *real* pauses by running
      its own Silero VAD on `audio_path` (see
      `rewards.PauseOnlyReward._source_silence_intervals`), so in-audio
      silences score the same way inserted ones do.
    """

    def __init__(self, config, split, processor=None):
        super().__init__(config, split)
        self.processor = processor
        self.sampling_rate = getattr(config, "sampling_rate", SAMPLE_RATE)
        self.seed = int(getattr(config, "seed", 42))
        self.epoch = 0
        # Per-gap silence sampling + source_vad widening (multi-audio only).
        self.min_silence = float(getattr(config, "rl_min_silence", 0.4))
        self.max_silence = float(getattr(config, "rl_max_silence", 1.0))
        self.source_vad_offset = float(getattr(config, "rl_source_vad_offset", 0.2))

        cols = self.load_parquet_columns(
            paths=config.train_parquet_dir if split == "train" else config.dev_parquet_dir,
            columns=["en_path", "en_length", "text", "en_text", "zh_text", "vad"],
        )
        self.en_paths = cols["en_path"]
        self.en_lengths = cols["en_length"]
        self.transcripts = cols["text"]
        self.en_texts = cols["en_text"]
        self.references = cols["zh_text"]
        self.source_vads = cols["vad"]

        self.is_multi_audio = (
            len(self.en_paths) > 0 and isinstance(self.en_paths[0], list)
        )
        if self.is_multi_audio:
            self._init_multi_audio()
        else:
            self.lengths = self.en_lengths

        prefix_msgs = build_prefix_messages("")
        self._prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False,
        )[0]

    def _init_multi_audio(self):
        """Drop multi-audio rows whose mixed source duration can fall outside
        [2, 20]s (the scalar pushdown filter in `load_parquet_columns` is
        skipped for list columns). The upper bound uses the worst-case silence
        (`max_silence` per gap) so no sampled mix ever exceeds 20s; `lengths`
        (dynamic-sampler hint) uses the mean silence.
        """
        keep, lengths = [], []
        for i in range(len(self.en_paths)):
            n_gaps = max(len(self.en_paths[i]) - 1, 0)
            base = sum(self.en_lengths[i])
            max_total = base + n_gaps * self.max_silence
            mean_total = base + n_gaps * (self.min_silence + self.max_silence) / 2.0
            if base >= 2.0 and max_total <= 20.0:
                keep.append(i)
                lengths.append(mean_total)

        def _take(lst):
            return [lst[i] for i in keep]

        self.en_paths = _take(self.en_paths)
        self.en_lengths = _take(self.en_lengths)
        self.en_texts = _take(self.en_texts)
        self.references = _take(self.references)
        self.transcripts = _take(self.transcripts)
        self.source_vads = _take(self.source_vads)
        self.lengths = lengths

    def __len__(self):
        return len(self.en_paths)

    def set_epoch(self, epoch: int):
        """Select deterministic per-epoch online pause augmentation."""
        self.epoch = int(epoch)

    def _getitem_multi(self, idx):
        """Mix the segment audios with per-gap generated silence and build the
        joined texts + source_vad pause intervals (see class docstring)."""
        paths = self.en_paths[idx]
        en_texts = [(t or "").strip() for t in self.en_texts[idx]]
        zh_texts = [(t or "").strip() for t in self.references[idx]]
        source_text = " ".join(en_texts)
        target_text = "".join(zh_texts)

        # Vary training mixtures by epoch while preserving reproducibility.
        # Development mixtures stay fixed across evaluations.
        if self.split == "train":
            mix_seed = self.seed + self.epoch * len(self) + int(idx)
        else:
            mix_seed = self.seed + int(idx)
        audio, gaps = concat_audios_with_random_silence(
            paths,
            sr=self.sampling_rate,
            seed=mix_seed,
            min_silence=self.min_silence,
            max_silence=self.max_silence,
        )
        total_dur = len(audio) / self.sampling_rate

        half = self.source_vad_offset / 2.0
        source_vad = [
            (max(0.0, start - half), min(total_dur, end + half))
            for (start, end, _dur) in gaps
        ]

        return {
            "audio": audio,             # pre-mixed waveform for the model input
            # Real path for token2wav speaker conditioning.  For 1-element
            # rows this is also the exact source waveform, which the reward
            # VADs to recover source pauses (source_vad below stays empty).
            "audio_path": paths[0],
            "prefix_text": self._prefix_text,
            "source_length": total_dur,
            "source_text": source_text,
            "target_text": target_text,
            "source_vad": source_vad,
        }

    def __getitem__(self, idx):
        if self.is_multi_audio:
            return self._getitem_multi(idx)

        transcript = self.transcripts[idx]
        reference = self.references[idx]
        return {
            "audio": self.en_paths[idx],
            "audio_path": self.en_paths[idx],
            "prefix_text": self._prefix_text,
            "source_length": self.en_lengths[idx],
            "source_text": (transcript or "").strip(),
            "target_text": (reference or "").strip(),
            "source_vad": self.source_vads[idx] or [],
        }
