"""
GSPO (Group Sequence Policy Optimization) Training for Qwen3-ASR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Masked-subsequence GSPO with either one reward or a combined objective.
  Single-reward runs select one *trained* subsequence via `mask_mode`:

        mask_mode = "text"   — translation tokens between
                                <s2st_text> and <s2st_eos>
        mask_mode = "speech" — codec tokens after <s2st_text>
        mask_mode = "all"    — every generated position after <s2st_text>

  We form a sequence-level importance ratio over that subsequence

        ρ_i = exp( (1/N_i) Σ_t m_{i,t} [logπ_θ(y_t|·) − logπ_old(y_t|·)] )

  and clip *at the subsequence level*.  Group-relative advantages use
  std-floor normalisation within each sampled group.

        L = − 1/B Σ_i  min( ρ_i A_i, clip(ρ_i, 1−εl, 1+εh) A_i )

  In `combined` mode, translation quality gets its own text-masked GSPO
  loss, while pause and duration each get a speech-masked GSPO loss.  The
  independently standardised component losses are weighted and averaged;
  KL is applied once over the union of text and speech positions.

Usage:
    torchrun --nproc_per_node=${NUM_GPUS} gspo_train.py --config-name gspo_stage4_rl
"""
import json
import math
import os
import random
import re
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import hydra
import librosa
import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from models import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor  # noqa: F401
from model import patch_outer_forward
from dataset import T2STDataset, T2STRLDataset
from dataset.qwen3_collator import Qwen3ASRCollator, load_audio
from rewards import RewardServer, compute_extreme_mask

# ---------------------------------------------------------------------------
# inlined from grpo_train.py (shared GRPO/GSPO infrastructure) — verbatim
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# generation helpers  (matching infer_single.py)
# ---------------------------------------------------------------------------
class CodecRepetitionPenaltyProcessor(LogitsProcessor):
    """Apply repetition penalty only to speech codec tokens.

    Text tokens are left untouched. The penalty only activates after
    <s2st_text> (i.e. during the translation+speech generation phase).
    """

    def __init__(
        self,
        penalty: float,
        s2st_text_id: int,
        prompt_len: int,
        codec_offset: int,
        codec_size: int,
    ):
        if penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {penalty}")
        self.penalty = penalty
        self.s2st_text_id = s2st_text_id
        self.prompt_len = prompt_len
        self.codec_offset = codec_offset
        self.codec_end = codec_offset + codec_size

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.penalty == 1.0:
            return scores
        for i in range(scores.shape[0]):
            gen = input_ids[i, self.prompt_len:]
            if not (gen == self.s2st_text_id).any():
                continue
            unique_tokens = gen.unique()
            codec_tokens = unique_tokens[
                (unique_tokens >= self.codec_offset) & (unique_tokens < self.codec_end)
            ]
            if codec_tokens.numel() == 0:
                continue
            token_scores = scores[i, codec_tokens]
            scores[i, codec_tokens] = torch.where(
                token_scores > 0,
                token_scores / self.penalty,
                token_scores * self.penalty,
            )
        return scores


class ASRGreedyProcessor(LogitsProcessor):
    """Greedy decoding during the ASR phase, pass-through afterwards.

    Before the model emits <s2st_text>, every token is decoded greedily
    (collapse logits to argmax).  Once <s2st_text> appears in the generated
    sequence the ASR phase is over; all subsequent tokens (translation text
    and speech codec) are left untouched for nucleus / top-k sampling.
    """

    def __init__(self, s2st_text_id: int, prompt_len: int):
        self.s2st_text_id = s2st_text_id
        self.prompt_len = prompt_len

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        for i in range(scores.shape[0]):
            gen = input_ids[i, self.prompt_len:]
            if (gen == self.s2st_text_id).any():
                continue
            top = scores[i].argmax().item()
            greedy = torch.full_like(scores[i], float("-inf"))
            greedy[top] = 0.0
            scores[i] = greedy
        return scores


class StopAtSecondEos(StoppingCriteria):
    """Continue generation past the first <eos> (end of text) and stop at the
    second <eos> (end of speech), matching the training target layout."""

    def __init__(self, eos_token_id: int, prompt_len: int):
        self.eos_token_id = eos_token_id
        self.prompt_len = prompt_len

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kw,
    ) -> bool:
        for i in range(input_ids.shape[0]):
            gen = input_ids[i, self.prompt_len:]
            if (gen == self.eos_token_id).sum().item() < 2:
                return False
        return True


def parse_interleaved_output(
    generated_ids: List[int],
    s2st_text_id: int,
    s2st_eos_id: int,
    eos_id: int,
    codec_offset: int,
    codec_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Split generated token ids into ASR ids, translation text ids, and
    speech codec ids.

    Expected layout:
      {asr tokens} <s2st_text> {interleaved text+speech} <s2st_eos> {remaining speech} <eos>
    """
    s2st_text_pos, s2st_eos_pos = None, None
    for i, tid in enumerate(generated_ids):
        if tid == s2st_text_id and s2st_text_pos is None:
            s2st_text_pos = i
        if tid == s2st_eos_id and s2st_eos_pos is None:
            s2st_eos_pos = i

    asr_ids = generated_ids[:s2st_text_pos] if s2st_text_pos is not None else []

    if s2st_text_pos is not None and s2st_eos_pos is not None:
        interleaved = generated_ids[s2st_text_pos + 1: s2st_eos_pos]
        remaining = generated_ids[s2st_eos_pos + 1:]
    elif s2st_text_pos is not None:
        interleaved = generated_ids[s2st_text_pos + 1:]
        remaining = []
    else:
        interleaved = generated_ids
        remaining = []

    codec_end = codec_offset + codec_size
    is_codec = lambda t: codec_offset <= t < codec_end

    text_ids = [t for t in interleaved if not is_codec(t) and t != eos_id]
    speech_codes = [t - codec_offset for t in interleaved if is_codec(t)]
    speech_codes += [t - codec_offset for t in remaining if is_codec(t)]

    return asr_ids, text_ids, speech_codes


def build_gen_token_mask(
    gen_ids: List[int],
    prompt_len: int,
    mode: str,
    s2st_text_id: int,
    s2st_eos_id: int,
    eos_id: int,
    codec_offset: int,
    codec_size: int,
) -> List[int]:
    """Per-token mask over [prompt + gen_ids] selecting which positions
    contribute to the GRPO loss.

    mode:
      "all"    — every generated position after <s2st_text> (interleaved text +
                 speech codec + remaining speech). Use with isometric reward
                 or any reward that depends on the full output.
      "text"   — only translation text tokens between <s2st_text> and
                 <s2st_eos> (no codec, no eos). Use with SeedX-style rewards
                 that score translation quality.
      "speech" — only speech codec tokens after <s2st_text>.
    """
    full_len = prompt_len + len(gen_ids)
    mask = [0] * full_len

    s2st_text_pos, s2st_eos_pos = None, None
    for k, tid in enumerate(gen_ids):
        if tid == s2st_text_id and s2st_text_pos is None:
            s2st_text_pos = k
        if tid == s2st_eos_id and s2st_eos_pos is None:
            s2st_eos_pos = k

    if s2st_text_pos is None:
        return mask

    codec_end = codec_offset + codec_size

    if mode == "all":
        for k in range(s2st_text_pos + 1, len(gen_ids)):
            mask[prompt_len + k] = 1
    elif mode == "text":
        end = s2st_eos_pos if s2st_eos_pos is not None else len(gen_ids)
        for k in range(s2st_text_pos + 1, end):
            tid = gen_ids[k]
            if tid != eos_id and not (codec_offset <= tid < codec_end):
                mask[prompt_len + k] = 1
    elif mode == "speech":
        for k in range(s2st_text_pos + 1, len(gen_ids)):
            tid = gen_ids[k]
            if codec_offset <= tid < codec_end:
                mask[prompt_len + k] = 1
    else:
        raise ValueError(f"unknown train_token_mode: {mode}")

    return mask


# ---------------------------------------------------------------------------
# collator  (only prepares the source side — targets are generated by RL)
# ---------------------------------------------------------------------------
def _load_audio(path, sr: int = 16000):
    # Multi-audio RL samples arrive pre-mixed (segments concatenated with
    # generated silence) as a ready waveform array; pass those through.
    # Single-audio samples arrive as a file path and are loaded here.
    if isinstance(path, np.ndarray):
        return path
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


class GRPOCollator:
    def __init__(self, processor, sampling_rate: int = 16000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, features):
        audios = [_load_audio(f["audio"], sr=self.sampling_rate) for f in features]
        prefix_texts = [f["prefix_text"] for f in features]

        batch = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        batch["source_lengths"] = torch.tensor(
            [f["source_length"] for f in features], dtype=torch.float32)
        # Reward token2wav conditioning needs a real file path; for multi-audio
        # the model input `audio` is a pre-mixed array, so fall back to the
        # dedicated `audio_path` (a representative source segment).
        batch["audio_paths"] = [f.get("audio_path", f["audio"]) for f in features]
        batch["source_vads"] = [f.get("source_vad", []) for f in features]
        batch["source_texts"] = [f.get("source_text", "") for f in features]
        batch["target_texts"] = [f.get("target_text", "") for f in features]
        return batch


# ---------------------------------------------------------------------------
# log-prob helper
# ---------------------------------------------------------------------------
def compute_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor = None,
):
    """Per-token log-probs for a causal LM.  Returns shape [B, T-1].

    When `mask` (shape [B, T-1], 1 at positions whose log-probs are needed)
    is given, only those positions go through the full-vocab softmax; the
    rest are zero-filled.  The GRPO loss multiplies by the same mask, so
    this is loss-equivalent while skipping a softmax over the prompt
    region (mel features + text prefix), which dominates T.
    """
    shifted = logits[:, :-1]
    targets = input_ids[:, 1:]

    if mask is None:
        B, T, V = shifted.shape
        nll = F.cross_entropy(
            shifted.reshape(B * T, V).float(),
            targets.reshape(B * T),
            reduction="none",
        )
        return -nll.view(B, T)

    mask_bool = mask.bool()
    active_logits = shifted[mask_bool]            # [N_active, V]
    active_targets = targets[mask_bool]           # [N_active]

    nll_active = F.cross_entropy(
        active_logits.float(), active_targets, reduction="none",
    )

    out = torch.zeros(targets.shape, dtype=torch.float32, device=targets.device)
    out[mask_bool] = -nll_active
    return out


# ---------------------------------------------------------------------------
# model loading helper
# ---------------------------------------------------------------------------
_MODEL_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_model_param_dtype(config, freeze: bool) -> torch.dtype:
    """Resolve policy/reference parameter storage dtype.

    Parameter storage defaults to BF16. The configuration can select FP32
    master parameters for the trainable policy while
    keeping the frozen reference in BF16 via ``train_param_dtype`` and
    ``ref_param_dtype`` respectively.
    """
    key = "ref_param_dtype" if freeze else "train_param_dtype"
    raw = str(config.model.get(key, "bfloat16")).strip().lower()
    try:
        return _MODEL_DTYPE_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            f"model.{key} must be one of {sorted(_MODEL_DTYPE_ALIASES)}, got {raw!r}"
        ) from exc


def _validate_special_token_ids(config, tokenizer) -> Dict[str, int]:
    """Fail fast when tokenizer IDs do not match the SFT checkpoint semantics.

    Tensor shapes alone cannot detect a reordered additional-special-token
    list. When provided, ``expected_special_token_ids`` validates the token
    mapping against the checkpoint's expected IDs.
    """
    special_tokens = [t for t in list(config.model.get("special_tokens", [])) if t]
    if special_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )

    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    missing = [token for token, token_id in actual.items() if token_id is None]
    if missing:
        raise ValueError(f"special tokens were not registered: {missing}")

    expected = dict(config.model.get("expected_special_token_ids", {}) or {})
    mismatched = {
        token: {"expected": int(token_id), "actual": actual.get(token)}
        for token, token_id in expected.items()
        if actual.get(token) != int(token_id)
    }
    if mismatched:
        raise ValueError(
            "special-token IDs do not match the SFT checkpoint semantics: "
            f"{mismatched}"
        )

    codec_offset = int(config.model.speech_codec_offset)
    if len(tokenizer) > codec_offset:
        raise ValueError(
            f"text tokenizer size {len(tokenizer)} overlaps codec offset {codec_offset}"
        )
    if any(token_id >= codec_offset for token_id in actual.values()):
        raise ValueError(
            f"special-token IDs must stay below codec offset {codec_offset}: {actual}"
        )
    return actual


def _build_model(config, device, freeze=False):
    """Load and prepare a Qwen3-ASR model.  Set freeze=True for the ref copy."""
    label = "ref" if freeze else "policy"
    param_dtype = _resolve_model_param_dtype(config, freeze=freeze)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        config.model.model_path, dtype=param_dtype, device_map=None,
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3ASRProcessor.from_pretrained(
        config.model.model_path, fix_mistral_regex=True,
    )

    codec_offset = config.model.speech_codec_offset
    codec_size = config.model.speech_codec_size
    special_token_ids = _validate_special_token_ids(config, processor.tokenizer)
    model.thinker.resize_token_embeddings(codec_offset + codec_size)

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    if config.checkpoint.resume_ckpt_path:
        checkpoint_path = os.fspath(config.checkpoint.resume_ckpt_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"initial checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        strict_load = bool(config.checkpoint.get("strict_model_load", False))
        msg = model.load_state_dict(sd, strict=strict_load)
        del sd, ckpt
        print(f"[{label}] loaded checkpoint: {msg}")

    model = model.to(device)
    print(
        f"[{label}] parameter_dtype={param_dtype}; "
        f"special_token_ids={special_token_ids}; tokenizer_size={len(processor.tokenizer)}"
    )

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if config.optimization.get("compile_ref", False):
            print(f"[{label}] torch.compile: ref forward (mode=reduce-overhead)")
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    elif config.optimization.get("compile_policy", False):
        # HF generate() under torch.compile is fragile (cudagraph + dynamic
        # shapes often misbehave). Only flip this on after confirming
        # generation still works for this codebase.
        print(f"[{label}] torch.compile: policy forward (mode=reduce-overhead)")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    return model, processor


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------
LAST_CKPT_NAME = "ckpt_last.pt"


def find_latest_train_ckpt(out_dir: str) -> Optional[str]:
    """Return path to the latest checkpoint to resume from.

    Prefers `ckpt_last.pt`, which is written at every save event. If it is
    absent, selects the highest-step `ckpt_step<N>.pt` checkpoint.
    """
    if not os.path.isdir(out_dir):
        return None
    last_path = os.path.join(out_dir, LAST_CKPT_NAME)
    if os.path.isfile(last_path):
        return last_path
    best_step = -1
    best_path = None
    for fname in os.listdir(out_dir):
        m = re.match(r"ckpt_step(\d+)\.pt$", fname)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = os.path.join(out_dir, fname)
    return best_path


def save_train_state(
    path: str, policy_model, optimizer, scheduler,
    epoch: int, global_step: int, step_in_epoch: int,
    also_write_last: bool = True,
):
    """Save full training state to `path`. If `also_write_last`, also copy to
    `<dir>/ckpt_last.pt` so resume always picks up the freshest state.
    Uses an atomic rename so a mid-write SIGTERM cannot corrupt the file.
    """
    sd = {"model." + k: v for k, v in policy_model.state_dict().items()}
    state = {
        "state_dict": sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "step_in_epoch": step_in_epoch,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    if also_write_last:
        last_path = os.path.join(os.path.dirname(path), LAST_CKPT_NAME)
        if os.path.abspath(path) != os.path.abspath(last_path):
            tmp_last = last_path + ".tmp"
            shutil.copyfile(path, tmp_last)
            os.replace(tmp_last, last_path)


def load_train_state(path: str, policy_model, optimizer, scheduler, device):
    """Load a full-state ckpt into model+optimizer+scheduler. Returns
    (epoch, global_step, step_in_epoch). Optimizer/scheduler are skipped if
    missing (back-compat with model-only ckpts)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
    policy_model.load_state_dict(sd, strict=False)

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        # Move optimizer state tensors to the right device (load_state_dict
        # restores them on CPU).
        for group in optimizer.state.values():
            for k, v in group.items():
                if isinstance(v, torch.Tensor):
                    group[k] = v.to(device)
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return (
        int(state.get("epoch", 0)),
        int(state.get("step", 0)),
        int(state.get("step_in_epoch", 0)),
    )


def read_wandb_run_id(out_dir: str) -> Optional[str]:
    p = os.path.join(out_dir, "wandb_run_id.txt")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        rid = f.read().strip()
    return rid or None


def write_wandb_run_id(out_dir: str, run_id: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run_id)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class GSPOConfig:
    num_samples: int = 8
    max_new_tokens: int = 896
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 0
    num_gen_batches: int = 4
    # Number of retained candidates forwarded at once during RL training.
    # 0 disables candidate micro-chunking.  This does not change rollout count,
    # extreme filtering, group advantages, or the optimizer-step batch size.
    train_micro_batch_size: int = 0
    speech_token_rate: float = 25.0
    repetition_penalty: float = 1.0
    # Configure silence-code sets for the model in use; the fallback is not
    # a substitute for audio-domain validation (rewards.SPEECH_SILENCE_CODES).
    speech_silence_tokens: Tuple[int, ...] = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
    # Recall-first superset used ONLY by the token backend's anti-hack
    # occupancy term; None -> same as speech_silence_tokens.
    speech_occupancy_silence_tokens: Optional[Tuple[int, ...]] = None
    # FP32 policy master weights can still use BF16 forward/backward compute.
    # BF16 autocast needs no GradScaler.
    compute_dtype: str = "bfloat16"

    # GSPO clipping (asymmetric supported)
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2

    # KL penalty against frozen reference (0 disables when adaptive is off)
    kl_coef: float = 0.0

    # Adaptive KL: when enabled, kl_coef is auto-tuned each step via a
    # proportional controller (Schulman 2017 §3.4 / TRL).  ref_model is
    # always loaded when adaptive is on, even if kl_coef starts at 0.
    kl_adaptive: bool = False
    kl_target: float = 0.01            # desired per-token KL
    kl_horizon: float = 10000.0        # update smoothing horizon (steps)
    kl_coef_min: float = 1e-6
    kl_coef_max: float = 1.0

    # reward: single component or text+speech multi-objective training
    reward_type: str = "translation_quality"  # "isometric" | "translation_quality" | "pause_only" | "combined"

    # Combined mode weights independently standardised component GSPO losses:
    #   (w_q L_quality + w_p L_pause + w_d L_duration) / sum(weights)
    # Keeping duration below pause initially makes it a soft global constraint
    # rather than letting length matching dominate pause placement.
    quality_weight: float = 1.0
    pause_weight: float = 1.0
    duration_weight: float = 0.5
    duration_eps_free: float = 0.05
    duration_speech_only: bool = False
    duration_speech_ratio_center: float = 1.0

    # MetricX-24-Hybrid (only used by translation_quality)
    metricx_model_dir: str = ""
    metricx_repo: str = ""
    metricx_tokenizer: str = "google/mt5-xl"
    metricx_max_input_length: int = 1536
    metricx_max_score: float = 25.0
    metricx_batch_size: int = 16

    # Token2wav renderer, used only by the audio pause backend
    # (pause_backend="audio").  The token backend never renders.
    # Set these in the yaml (gspo.token2wav_*); see config/gspo_stage4_rl.yaml.
    token2wav_flow_model: str = ""
    token2wav_hift_model: str = ""
    token2wav_campplus_model: str = ""
    token2wav_speech_tokenizer_model: str = ""
    token2wav_fp16: bool = False
    token2wav_n_timesteps: int = 2
    token2wav_cfg: bool = False
    # Silero VAD parameters (mirror eval/silence_iou.py).  Used by the audio
    # backend on decoded candidates and by BOTH backends to recover source
    # pauses of single-audio samples.
    pause_vad_sample_rate: int = 16000
    pause_vad_threshold: float = 0.8
    pause_min_speech_ms: int = 250
    pause_min_silence_ms: int = 100
    pause_speech_pad_ms: int = 30
    # --- pause_only reward (rewards.compute_pause_alignment) ---
    # pause_backend="token" scores pauses directly on codec silence-token
    # runs (speech_silence_tokens at speech_token_rate tokens/s):
    # deterministic, no token2wav/VAD noise inside the group ranking, and no
    # rendering cost in the training loop. "audio" uses token2wav rendering
    # and Silero VAD for waveform-based scoring.
    pause_backend: str = "token"
    timeline_w_miss: float = 1.5
    timeline_w_fa: float = 1.0
    timeline_w_overhang: float = 2.0
    timeline_w_drift: float = 0.0
    timeline_drift_tol: float = 3.0
    # Strict threshold matches the external eval (>400 ms); the dense
    # threshold provides sub-threshold shaping (>200 ms) blended in with
    # pause_dense_weight.
    pause_strict_min_silence_ms: int = 400
    pause_dense_min_silence_ms: int = 200
    pause_dense_weight: float = 0.25
    # reward = followed - lambda * added, where followed blends the strict
    # and dense collared coverage of source-pause cores and added is the
    # dense false-pause seconds normalized by the target core (capped).
    pause_collar: float = 0.10
    pause_added_silence_lambda: float = 0.3
    pause_added_silence_cap: float = 2.0
    # Per-utterance free allowance for natural micro-pauses: this fraction
    # of the source SPEECH seconds is deducted from the added occupancy
    # before normalization, so long fluent utterances are not flood-gated.
    pause_added_free_fraction: float = 0.05
    # Hard flood-gate threshold on the added ratio; sits above the penalty
    # cap so added-measurement noise near the cap costs at most the capped
    # penalty rather than flipping the candidate to the floor.
    pause_silence_flood_ratio: float = 3.0
    # Token backend: merge credit-set runs separated by <= this many tokens
    # (the silence-code set has a long tail; a single missed code must not
    # fragment one acoustic pause below the strict threshold).
    pause_bridge_gap_tokens: int = 2
    # Exploration shaping: flat penalty when the source has pauses but the
    # candidate emitted no strict pause at all (creates "try pausing"
    # within-group variance on hard real data). 0 disables.
    pause_no_generated_penalty: float = 0.0
    pause_target_core_floor: float = 0.4
    # Hard anti-hack gate: candidates whose duration ratio leaves the
    # ±tolerance band are floored, so stretching the utterance cannot game
    # the source-timeline mapping.  Inside the band duration contributes
    # zero gradient (pause-only probing).
    pause_duration_gate_tolerance: float = 0.25
    pause_duration_gate_floor: float = -1.0
    pause_drop_edge_silence: bool = True
    pause_no_speech_reward: float = 0.0
    pause_eps: float = 1e-6
    # Single-audio samples have no inserted silences, so their source_vad
    # arrives empty; when true, the reward fills it by running the same
    # Silero VAD (same thresholds + coercion as external eval) on the
    # source audio file, cached per path.  Multi-audio samples are
    # unaffected (they always carry the inserted-gap intervals).
    pause_source_vad_from_audio: bool = True
    # Combined mode: if |generated_duration / source_duration - 1| exceeds
    # this tolerance, force pause reward to the duration-gate floor.
    pause_duration_tolerance: float = 0.10

    # translation_quality reward: maps raw MetricX to (max_score-raw)/max_score
    # in [0, 1]. quality_use_reference=False switches MetricX to QE mode.
    quality_use_reference: bool = True
    # Advantage shaping: (r - group_mean) / max(group_std, adv_std_floor),
    # optionally clipped. This keeps a useful scale without letting tiny
    # within-group std amplify reward-model noise too aggressively.
    adv_std_floor: float = 0.02
    adv_clip: float = 2.0
    # Optional post-winsorization centering, applied independently inside each
    # rollout group (and each local event). Disabled by default.
    adv_recenter_after_clip: bool = False

    # Oversample candidates, score them, then train only on the top-K best +
    # bottom-K worst per group (rest masked as NaN and dropped by the
    # finite-mask filter in generate_step). Applies to all reward types.
    extreme_filter: bool = False
    keep_top_k: int = 4
    keep_bottom_k: int = 4

    # Drop all-tied (zero reward variance) groups before the policy forward.
    # Always True for training; evaluate() overrides it to False so eval
    # means are computed over every sampled group, not the non-tied subset.
    drop_zero_signal_groups: bool = True

    # subsequence selector — leave empty to auto-derive from reward_type
    mask_mode: str = ""                      # "" | "text" | "speech" | "all"

    # CE/SFT auxiliary loss on ASR + translation text tokens, computed on a
    # separate SFT-format dataset. 0 disables. Typical 0.05–0.2.
    ce_coef: float = 0.0
    # CE fires once per N RL micro-batches inside an optimizer step (1=every).
    ce_every_n_microbatches: int = 1

    # Samples without speech codec tokens carry an all-zero gen_mask and
    # receive no direct policy gradient. Their negative reward lowers the
    # group baseline, increasing codec-emitting siblings' relative advantage.
    no_codec_reward: float = -2.0

    # --- held-out eval on dataset.dev_parquet_dir ---
    # Every eval_every_n_steps optimizer steps, generate on the dev set and log
    # the held-out counterpart of the train/* reward metrics (reward_mean,
    # pause_followed_silence, pause_added_silence, ...). 0 disables eval. Eval
    # reuses generate_step (same parse + reward server) but with the sampling
    # knobs below and extreme_filter forced OFF, so the means cover *all* sampled
    # candidates, not just the top/bottom-K kept for training. >=2 samples are
    # required because generate_step forms reward groups.
    eval_every_n_steps: int = 0
    eval_num_batches: int = 0       # cap dev micro-batches/rank (0 = full pass)
    eval_num_samples: int = 8
    eval_temperature: float = 0.85
    eval_top_p: float = 0.85
    eval_top_k: int = 0
    eval_seed: int = 42


TEXT_REWARD_TYPES = ("translation_quality",)

TEXT_MASK_REWARD_TYPES = ("translation_quality",)
SPEECH_MASK_REWARD_TYPES = ("isometric", "pause_only")
COMBINED_REWARD_TYPE = "combined"


def _model_compute_context(device: torch.device, raw_dtype: str):
    """Return the configured CUDA autocast context for model compute."""
    name = str(raw_dtype).strip().lower()
    if name in ("bf16", "bfloat16"):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    if name in ("fp32", "float32"):
        return nullcontext()
    raise ValueError(
        "model.compute_dtype must be bf16/bfloat16 or fp32/float32, "
        f"got {raw_dtype!r}"
    )


def _audit_train_state_dtypes(policy_model, optimizer, require_fp32: bool) -> str:
    """Audit policy parameters, live gradients, and Adam moment precision."""
    param_dtypes = {
        p.dtype for p in policy_model.parameters() if p.requires_grad
    }
    grad_dtypes = {
        p.grad.dtype
        for p in policy_model.parameters()
        if p.requires_grad and p.grad is not None
    }
    moment_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for key, value in state.items()
        if key in ("exp_avg", "exp_avg_sq") and isinstance(value, torch.Tensor)
    }
    if require_fp32:
        expected = {torch.float32}
        if param_dtypes != expected:
            raise RuntimeError(f"FP32 policy requested but parameter dtypes are {param_dtypes}")
        if grad_dtypes != expected:
            raise RuntimeError(f"FP32 policy requested but gradient dtypes are {grad_dtypes}")
        if moment_dtypes != expected:
            raise RuntimeError(f"FP32 policy requested but Adam moment dtypes are {moment_dtypes}")
    return (
        f"params={sorted(map(str, param_dtypes))}; "
        f"grads={sorted(map(str, grad_dtypes))}; "
        f"Adam moments={sorted(map(str, moment_dtypes))}"
    )


# ---------------------------------------------------------------------------
# Adaptive KL controller (proportional, Schulman 2017 §3.4 / TRL)
# ---------------------------------------------------------------------------
class AdaptiveKLController:
    """Auto-tune kl_coef so that observed KL stays near `target`.

    update():  proportional_error = clip(current/target - 1, -0.2, 0.2)
               mult = 1 + proportional_error * n_steps / horizon
               value *= mult, then clamp into [coef_min, coef_max].

    `horizon` controls how aggressively coef moves; smaller → faster.  NOTE we
    drive this with n_steps=1 per optimizer step (see training loop), so
    `horizon` is measured in OPTIMIZER STEPS, not in samples like TRL.  TRL
    passes n_steps=batch_size, so its 10_000 default is ~batch_size× too slow
    here and effectively freezes the coef.  Max per-step change is
    0.2/horizon; for short (300–500 step) runs use ~30 (coef ~doubles over
    ~100 steps), scaling up with run length.
    """

    def __init__(
        self,
        init_kl_coef: float,
        target: float,
        horizon: float,
        coef_min: float = 1e-6,
        coef_max: float = 1.0,
    ):
        self.value = float(init_kl_coef)
        self.target = float(target)
        self.horizon = float(horizon)
        self.coef_min = float(coef_min)
        self.coef_max = float(coef_max)

    def update(self, current_kl: float, n_steps: int = 1) -> float:
        target = max(self.target, 1e-8)
        proportional_error = max(min(current_kl / target - 1.0, 0.2), -0.2)
        mult = 1.0 + proportional_error * n_steps / max(self.horizon, 1.0)
        self.value = max(self.coef_min, min(self.coef_max, self.value * mult))
        return self.value


class _PhaseTimer:
    """Accumulate elapsed time for a named phase into a shared dict.

    `torch.cuda.synchronize` is called on enter and exit so GPU work
    launched inside the block is fully attributed to this phase rather
    than spilling into the next.
    """

    def __init__(self, device, store: Dict[str, float], key: str):
        self._device = device
        self._store = store
        self._key = key

    def __enter__(self):
        if self._device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        self._store[self._key] = (
            self._store.get(self._key, 0.0) + time.perf_counter() - self._t0
        )
        return False


KL_STATE_FNAME = "kl_state.json"


def save_kl_state(out_dir: str, controller: "AdaptiveKLController") -> None:
    """Atomically write controller state next to checkpoints. Tiny JSON file
    so it survives mid-write SIGTERMs (rename is atomic on the same FS)."""
    path = os.path.join(out_dir, KL_STATE_FNAME)
    payload = {
        "value": float(controller.value),
        "target": float(controller.target),
        "horizon": float(controller.horizon),
        "coef_min": float(controller.coef_min),
        "coef_max": float(controller.coef_max),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_kl_state(out_dir: str, controller: "AdaptiveKLController") -> bool:
    """Restore `controller.value` from sidecar. Returns False if no sidecar.

    Only the live `value` is restored; static knobs (target/horizon/bounds)
    keep the *current* config so re-tuning between resumes works as expected.
    """
    path = os.path.join(out_dir, KL_STATE_FNAME)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        state = json.load(f)
    controller.value = max(
        controller.coef_min, min(controller.coef_max, float(state["value"]))
    )
    return True


# ---------------------------------------------------------------------------
# GSPO loss  (sequence-level ratio over a single subsequence)
# ---------------------------------------------------------------------------
def _group_standardize_rewards(
    rewards: torch.Tensor,
    group_sizes: List[int],
    adv_std_floor: float = 0.02,
    adv_clip: float = 2.0,
    recenter_after_clip: bool = False,
) -> torch.Tensor:
    """Group-relative advantage with a std floor and optional clipping.

    Formula per group:
        adv = (r - group_mean) / max(group_std, adv_std_floor)
        adv = clamp(adv, -adv_clip, adv_clip) when adv_clip > 0
        adv = adv - mean(adv) when recenter_after_clip

    Singleton groups yield 0, though valid groups are normally filtered to
    size >= 2 upstream.
    """
    adv = torch.empty_like(rewards)
    std_floor = max(float(adv_std_floor), 1e-8)
    offset = 0
    for g in group_sizes:
        chunk = rewards[offset: offset + g]
        if g >= 2:
            centered = chunk - chunk.mean()
            denom = torch.clamp(chunk.std(unbiased=False), min=std_floor)
            values = centered / denom
            if adv_clip > 0:
                values = values.clamp(-float(adv_clip), float(adv_clip))
            if recenter_after_clip:
                values = values - values.mean()
            adv[offset: offset + g] = values
        else:
            adv[offset: offset + g] = 0.0
        offset += g
    return adv


def _compute_gspo_loss_from_advantages(
    pol_lp: torch.Tensor,                 # [B, T-1]
    gen_lp: torch.Tensor,                 # [B, T-1]
    ref_lp: Optional[torch.Tensor],       # [B, T-1] or None when kl_coef == 0
    advantages: torch.Tensor,             # [B], standardized over the full group
    gen_mask: torch.Tensor,               # [B, T] unshifted, selects the trained subsequence
    clip_eps_low: float,
    clip_eps_high: float,
    kl_coef: float,
    adv_std_floor: float,
    adv_clip: float,
    n_valid_normalizer: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute a GSPO loss from already-standardized advantages.

    ``n_valid_normalizer`` normally defaults to the local batch's number of
    valid sequences.  Candidate micro-chunking instead passes the full retained
    batch's count, so summing chunk losses is exactly the unsplit objective.
    """
    mask = gen_mask[:, 1:].float()
    n_tok = mask.sum(-1)                              # [B]
    valid = (n_tok > 0).float()
    local_n_valid = valid.sum()
    if n_valid_normalizer is None:
        n_valid = local_n_valid.clamp(min=1)
    else:
        n_valid = torch.as_tensor(
            max(float(n_valid_normalizer), 1.0),
            dtype=torch.float32,
            device=pol_lp.device,
        )
    adv = advantages

    log_ratio = pol_lp - gen_lp                       # zero outside lp_mask
    seq_log_ratio = (log_ratio * mask).sum(-1) / n_tok.clamp(min=1)
    ratio = seq_log_ratio.exp()                       # [B]

    ratio_clipped = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high)
    surr = torch.min(ratio * adv, ratio_clipped * adv) * valid
    policy_loss = -(surr.sum() / n_valid)

    # Per-token KL (Schulman k3 estimator) averaged over the trained
    # subsequence, then averaged over valid sequences. Skip when kl_coef==0.
    if kl_coef > 0 and ref_lp is not None:
        kl_log_ratio = ref_lp - pol_lp
        kl_per_tok = torch.exp(kl_log_ratio) - kl_log_ratio - 1
        seq_kl = (kl_per_tok * mask).sum(-1) / n_tok.clamp(min=1)
        kl_mean = (seq_kl * valid).sum() / n_valid
    else:
        kl_mean = torch.zeros((), device=pol_lp.device)

    loss = policy_loss + kl_coef * kl_mean

    with torch.no_grad():
        clip_frac = (
            (((ratio < 1 - clip_eps_low) | (ratio > 1 + clip_eps_high)).float() * valid).sum()
            / n_valid
        )
        ratio_mean = ((ratio * valid).sum() / n_valid)
        adv_mean = ((adv * valid).sum() / n_valid)

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_mean": kl_mean.item(),
        "kl_loss": (kl_coef * kl_mean).item(),
        "advantage_mean": adv_mean.item(),
        "advantage_std": adv.std().item() if adv.numel() > 1 else 0.0,
        "adv_std_floor": float(adv_std_floor),
        "adv_clip": float(adv_clip),
        "ratio_mean": ratio_mean.item(),
        "clip_fraction": clip_frac.item(),
        "n_valid": local_n_valid.item(),
    }
    return loss, metrics


def compute_gspo_loss(
    pol_lp: torch.Tensor,                 # [B, T-1]
    gen_lp: torch.Tensor,                 # [B, T-1]
    ref_lp: Optional[torch.Tensor],       # [B, T-1] or None when kl_coef == 0
    rewards: torch.Tensor,                # [B]
    gen_mask: torch.Tensor,               # [B, T] unshifted, selects the trained subsequence
    group_sizes: List[int],
    clip_eps_low: float,
    clip_eps_high: float,
    kl_coef: float,
    adv_std_floor: float,
    adv_clip: float,
    adv_recenter_after_clip: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    advantages = _group_standardize_rewards(
        rewards, group_sizes, adv_std_floor, adv_clip,
        adv_recenter_after_clip,
    )
    return _compute_gspo_loss_from_advantages(
        pol_lp, gen_lp, ref_lp, advantages, gen_mask,
        clip_eps_low, clip_eps_high, kl_coef, adv_std_floor, adv_clip,
    )


def compute_masked_kl(
    pol_lp: torch.Tensor,
    ref_lp: Optional[torch.Tensor],
    gen_mask: torch.Tensor,
    n_valid_normalizer: Optional[float] = None,
) -> torch.Tensor:
    """Schulman k3 KL, averaged over the union of selected tokens once.

    Combined training has three policy losses but must not triple-count the
    reference-model regulariser.  This helper mirrors the KL term in
    ``compute_gspo_loss`` and is used only for that combined path.
    """
    if ref_lp is None:
        return torch.zeros((), device=pol_lp.device)
    mask = gen_mask[:, 1:].float()
    n_tok = mask.sum(-1)
    valid = (n_tok > 0).float()
    if n_valid_normalizer is None:
        n_valid = valid.sum().clamp(min=1)
    else:
        n_valid = torch.as_tensor(
            max(float(n_valid_normalizer), 1.0),
            dtype=torch.float32,
            device=pol_lp.device,
        )
    kl_log_ratio = ref_lp - pol_lp
    kl_per_tok = torch.exp(kl_log_ratio) - kl_log_ratio - 1
    seq_kl = (kl_per_tok * mask).sum(-1) / n_tok.clamp(min=1)
    return (seq_kl * valid).sum() / n_valid


def combined_group_score(
    component_rewards: Dict[str, torch.Tensor],
    group_sizes: List[int],
    cfg: GSPOConfig,
) -> torch.Tensor:
    """Scale-free group score used only for combined top/bottom selection.

    Policy advantages remain separate per component.  This score prevents the
    raw [-4, 0] duration range from dominating [0, 1] MetricX during candidate
    filtering.
    """
    weighted = torch.zeros_like(component_rewards["translation_quality"])
    total_weight = 0.0
    for name, weight in (
        ("translation_quality", cfg.quality_weight),
        ("pause", cfg.pause_weight),
        ("duration", cfg.duration_weight),
    ):
        if weight <= 0:
            continue
        weighted = weighted + float(weight) * _group_standardize_rewards(
            component_rewards[name], group_sizes,
            cfg.adv_std_floor, cfg.adv_clip,
            cfg.adv_recenter_after_clip,
        )
        total_weight += float(weight)
    if total_weight <= 0:
        raise ValueError("combined reward needs at least one positive component weight")
    return weighted / total_weight


def gate_pause_reward_by_duration(
    pause_rewards: torch.Tensor,
    source_lengths: List[float],
    generated_speech_lengths: List[int],
    target_token_rate: float,
    tolerance: float,
    minimum_reward: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Floor pause reward when generated/source duration gap is too large.

    The duration definition intentionally matches ``compute_isometric_reward``:
    generated codec length divided by the configured speech token rate. The
    returned boolean mask is aligned with candidates for logging/filtering.
    """
    if pause_rewards.numel() == 0:
        return pause_rewards, torch.zeros_like(pause_rewards, dtype=torch.bool)
    if not (
        len(source_lengths) == len(generated_speech_lengths) == pause_rewards.numel()
    ):
        raise ValueError("duration gate inputs must align with pause rewards")
    source = torch.tensor(
        source_lengths, dtype=torch.float32, device=pause_rewards.device,
    ).clamp(min=1e-5)
    generated = torch.tensor(
        generated_speech_lengths, dtype=torch.float32, device=pause_rewards.device,
    ) / float(target_token_rate)
    relative_gap = torch.abs(generated / source - 1.0)
    # Small numerical slack preserves the intended strict "> tolerance"
    # boundary for values such as an exactly 1.10 duration ratio in fp32.
    gated = relative_gap > float(tolerance) + 1e-6
    out = pause_rewards.clone()
    # Floor, never raise: a candidate already scoring below the floor (e.g.
    # silence-flooded) must not be rewarded for ALSO violating duration.
    out[gated] = torch.minimum(
        out[gated],
        torch.full_like(out[gated], float(minimum_reward)),
    )
    return out, gated


# ---------------------------------------------------------------------------
# CE/SFT collator — interleaved SFT-format target with text-only labels
# ---------------------------------------------------------------------------
class CECollator(Qwen3ASRCollator):
    """Build full SFT-style sequences (interleaved translation text + speech
    codec, matching the model's inference layout) and produce labels with
    `-100` everywhere except ASR text + translation text tokens.

    Masked: prompt + `language English<asr_text>` static prefix + speech codec
    tokens + `<s2st_text>` / `<s2st_eos>` / `<eos>` markers + padding. So CE
    gradient flows ONLY through text-content tokens, while the model still
    sees the full interleaved context (so the CE objective doesn't fight the
    interleaved generation behavior).
    """

    ASR_PREFIX = "language English<asr_text>"

    def __init__(self, processor, codec_offset: int, codec_size: int,
                 sampling_rate: int = 16000):
        super().__init__(processor, sampling_rate)
        self.codec_offset = codec_offset
        self.codec_size = codec_size
        self.s2st_text_id = processor.tokenizer.convert_tokens_to_ids("<s2st_text>")

    def __call__(self, features):
        audios = [load_audio(f["audio"], sr=self.sampling_rate) for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        source_texts = [f["source_text"] for f in features]

        context_texts = [p + s for p, s in zip(prefix_texts, source_texts)]
        # mfx = prompt + static ASR prefix; its attention_mask sum gives the
        # number of positions that should be masked (audio prompt + the
        # constant `language English<asr_text>` lead-in).
        mask_texts = [p + self.ASR_PREFIX for p in prefix_texts]

        ctx = self.processor(
            text=context_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        mfx = self.processor(
            text=mask_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        mask_prefix_lens = mfx["attention_mask"].sum(dim=1).tolist()

        composed: List[List[int]] = []
        for i, feat in enumerate(features):
            seq_len = int(ctx["attention_mask"][i].sum().item())
            ctx_ids = ctx["input_ids"][i, -seq_len:].tolist()
            tgt_ids = self._build_target_ids(feat["target_text"], feat["target_speech_code"])
            composed.append(ctx_ids + tgt_ids)

        padded = self.processor.tokenizer.pad(
            {"input_ids": composed},
            padding=True, return_tensors="pt", padding_side="left",
        )
        ctx["input_ids"] = padded["input_ids"]
        ctx["attention_mask"] = padded["attention_mask"]

        labels = ctx["input_ids"].clone()
        width = labels.size(1)

        # Mask prompt + static ASR prefix (positions, not by token id — token
        # boundary may collide with real ASR content otherwise).
        for i, pl in enumerate(mask_prefix_lens):
            seq_len = int(ctx["attention_mask"][i].sum().item())
            start = width - seq_len
            labels[i, start : start + int(pl)] = -100

        # Mask padding.
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        # Mask speech codec id range (those are interleaved into the target
        # but should not contribute to CE — the user asked for text-only CE).
        codec_end = self.codec_offset + self.codec_size
        codec_mask = (labels >= self.codec_offset) & (labels < codec_end)
        labels[codec_mask] = -100

        # Mask structural markers and EOS.
        labels[labels == self.s2st_text_id] = -100
        labels[labels == self.s2st_eos_id] = -100
        labels[labels == self.eos_token_id] = -100

        ctx["labels"] = labels
        return ctx


def _cycle_ce_loader(loader: DataLoader, sampler: DistributedSampler):
    """Infinite generator over a DistributedSampler-wrapped loader. Bumps
    the sampler epoch each pass so shuffling re-rolls. Used to pair one CE
    batch with each RL micro-batch regardless of CE-dataset size."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# ---------------------------------------------------------------------------
# generation step
# ---------------------------------------------------------------------------
PAUSE_REWARD_METRIC_LOG_NAMES = {
    "translation_quality_reward": "quality_reward_mean",
    "duration_reward": "duration_reward_mean",
    "duration_within_2": "duration_within_2_frac",
    "duration_within_5": "duration_within_5_frac",
    "duration_within_10": "duration_within_10_frac",
    "pause_reward": "pause_reward_mean",
    "pause_followed_silence": "pause_followed_silence_mean",
    "pause_dense_followed_silence": "pause_dense_followed_silence_mean",
    "pause_followed_shaped": "pause_followed_shaped_mean",
    "pause_added_silence": "pause_added_silence_mean",
    "pause_dense_added_silence": "pause_dense_added_silence_mean",
    "pause_added_balanced": "pause_added_balanced_mean",
    "pause_added_ratio": "pause_added_ratio_mean",
    "pause_silence_flood_gated": "pause_silence_flood_gated_frac",
    "pause_no_generated_penalized": "pause_no_generated_penalized_frac",
    "pause_source_silence_total": "pause_source_silence_total_mean",
    "pause_generated_silence_total": "pause_generated_silence_total_mean",
    "pause_dense_generated_silence_total": (
        "pause_dense_generated_silence_total_mean"
    ),
    "pause_source_duration": "pause_source_duration_mean",
    "pause_generated_duration": "pause_generated_duration_mean",
    "pause_duration_ratio": "pause_duration_ratio_mean",
    "pause_timeline_scale": "pause_timeline_scale_mean",
    "pause_silence_code_fraction": "pause_silence_code_fraction_mean",
    "num_src_pauses": "num_src_pauses_mean",
    "num_gen_pauses": "num_gen_pauses_mean",
    "num_dense_gen_pauses": "num_dense_gen_pauses_mean",
    "pause_no_source": "pause_no_source_frac",
    "pause_no_generated": "pause_no_generated_frac",
    "pause_no_speech": "pause_no_speech_frac",
    "pause_source_vad_detected": "pause_source_vad_detected_frac",
    "pause_duration_gated": "pause_duration_gated_frac",
    "timeline_drift": "timeline_drift_mean",
    # Group-level ceiling diagnostics (broadcast per candidate, so the
    # candidate-weighted mean equals a group-size-weighted group mean):
    # followed_group_max is the best-of-N sampling ceiling the policy can
    # already reach; mean -> max convergence is what pause RL can harvest.
    "pause_followed_group_max": "pause_followed_group_max_mean",
    "pause_reward_group_max": "pause_reward_group_max_mean",
    "pause_reward_group_std": "pause_reward_group_std_mean",
    # Scalar diagnostic injected directly into reward_metric_means (never a
    # per-candidate row); listed here so evaluate() aggregates it too.
    "pause_zero_signal_group_frac": "pause_zero_signal_group_frac",
}


def _add_group_ceiling_metrics(
    metrics: Dict[str, List[float]],
    group_sizes: List[int],
) -> None:
    """Broadcast per-group max/std diagnostics onto candidate-aligned rows."""
    if not metrics:
        return
    n_items = sum(group_sizes)
    followed = metrics.get("pause_followed_silence")
    reward = metrics.get("pause_reward")
    if reward is None or len(reward) != n_items:
        return
    if followed is None or len(followed) != n_items:
        followed = None
    followed_max: List[float] = []
    reward_max: List[float] = []
    reward_std: List[float] = []
    offset = 0
    for g in group_sizes:
        group_reward = reward[offset: offset + g]
        r_max = max(group_reward)
        r_std = (
            float(torch.tensor(group_reward).std(unbiased=False).item())
            if g > 1 else 0.0
        )
        f_max = max(followed[offset: offset + g]) if followed is not None else 0.0
        followed_max.extend([f_max] * g)
        reward_max.extend([r_max] * g)
        reward_std.extend([r_std] * g)
        offset += g
    if followed is not None:
        metrics["pause_followed_group_max"] = followed_max
    metrics["pause_reward_group_max"] = reward_max
    metrics["pause_reward_group_std"] = reward_std


def _reward_metrics_are_aligned(
    metrics: Dict[str, List[float]],
    n_items: int,
) -> bool:
    return bool(metrics) and all(len(values) == n_items for values in metrics.values())


def _take_reward_metric_rows(
    metrics: Dict[str, List[float]],
    indices: List[int],
) -> Dict[str, List[float]]:
    if not metrics:
        return {}
    return {
        key: [float(values[i]) for i in indices]
        for key, values in metrics.items()
    }


def _summarize_reward_metrics(
    metrics: Dict[str, List[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for src_key, log_key in PAUSE_REWARD_METRIC_LOG_NAMES.items():
        values = metrics.get(src_key, [])
        finite_values = [
            float(v) for v in values
            if math.isfinite(float(v))
        ]
        if finite_values:
            out[log_key] = sum(finite_values) / len(finite_values)
    return out


def generate_step(
    policy_model, batch, cfg: GSPOConfig,
    codec_offset: int, codec_size: int, tok, device,
    reward_server: RewardServer,
    timings: Optional[Dict[str, float]] = None,
):
    """Sample cfg.num_samples completions per source, parse, and request rewards.

    Returns gen_data dict or None when the rank has no valid samples (the
    reward request is still issued to keep collectives lockstep).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    input_features = batch["input_features"].to(device=device, dtype=torch.bfloat16)
    feature_attention_mask = batch["feature_attention_mask"].to(device)
    source_lengths = batch["source_lengths"]
    audio_paths = batch.get("audio_paths", [""] * input_ids.shape[0])
    source_vads = batch.get("source_vads", [[] for _ in range(input_ids.shape[0])])
    source_texts = batch.get("source_texts", [""] * input_ids.shape[0])
    target_texts = batch.get("target_texts", [""] * input_ids.shape[0])

    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    eos_id = tok.eos_token_id
    s2st_text_id = tok.convert_tokens_to_ids("<s2st_text>")
    s2st_eos_id = tok.convert_tokens_to_ids("<s2st_eos>")

    all_full_ids: List[torch.Tensor] = []
    all_feat_idx: List[int] = []
    all_gen_masks: List[List[int]] = []
    all_text_masks: List[List[int]] = []
    all_speech_masks: List[List[int]] = []
    all_speech_lens: List[int] = []
    all_src_lens: List[float] = []
    all_source_texts: List[str] = []
    all_trans_texts: List[str] = []
    all_reference_texts: List[str] = []
    all_audio_paths: List[str] = []
    all_source_silences: List[List[Tuple[float, float]]] = []
    all_speech_codes: List[List[int]] = []
    group_sizes: List[int] = []
    no_codec_flags: List[bool] = []
    is_combined = cfg.reward_type == COMBINED_REWARD_TYPE
    uses_text_reward = cfg.reward_type in TEXT_REWARD_TYPES or is_combined
    uses_waveform_reward = cfg.reward_type == "pause_only" or is_combined

    silence_token_ids = {codec_offset + t for t in cfg.speech_silence_tokens}

    # Resolve subsequence mode.  Auto-pick from reward_type when not set.
    if cfg.mask_mode:
        mask_mode = cfg.mask_mode
    elif cfg.reward_type in TEXT_MASK_REWARD_TYPES:
        mask_mode = "text"
    elif cfg.reward_type in SPEECH_MASK_REWARD_TYPES:
        mask_mode = "speech"
    else:
        mask_mode = "all"

    with torch.no_grad():
        for i in range(B):
            single = {
                "input_ids": input_ids[i: i + 1],
                "attention_mask": attention_mask[i: i + 1],
                "input_features": input_features[i: i + 1],
                "feature_attention_mask": feature_attention_mask[i: i + 1],
            }

            processors = [ASRGreedyProcessor(s2st_text_id, prompt_len)]
            if cfg.repetition_penalty != 1.0:
                processors.append(CodecRepetitionPenaltyProcessor(
                    penalty=cfg.repetition_penalty,
                    s2st_text_id=s2st_text_id,
                    prompt_len=prompt_len,
                    codec_offset=codec_offset,
                    codec_size=codec_size,
                ))
            stop = StopAtSecondEos(eos_id, prompt_len)

            with _model_compute_context(device, cfg.compute_dtype):
                result = policy_model.generate(
                    **single,
                    num_return_sequences=cfg.num_samples,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    temperature=cfg.temperature,
                    eos_token_id=[],
                    pad_token_id=tok.pad_token_id,
                    logits_processor=LogitsProcessorList(processors),
                    stopping_criteria=StoppingCriteriaList([stop]),
                )

            sequences = result.sequences if hasattr(result, "sequences") else result
            valid_count = 0
            for j in range(sequences.shape[0]):
                gen_ids = sequences[j, prompt_len:].tolist()

                if len(gen_ids) >= 5 and all(t in silence_token_ids for t in gen_ids[-5:]):
                    while gen_ids and gen_ids[-1] in silence_token_ids:
                        gen_ids.pop()
                    gen_ids.append(eos_id)

                eos_count = 0
                for pos, tid in enumerate(gen_ids):
                    if tid == eos_id:
                        eos_count += 1
                        if eos_count == 2:
                            gen_ids = gen_ids[: pos + 1]
                            break

                asr_ids, trans_ids, speech_codes = parse_interleaved_output(
                    gen_ids, s2st_text_id, s2st_eos_id, eos_id, codec_offset, codec_size,
                )
                is_no_codec = len(speech_codes) == 0

                actual_mask = attention_mask[i].bool()
                actual_prompt = input_ids[i][actual_mask]
                full = torch.cat([
                    actual_prompt,
                    torch.tensor(gen_ids, dtype=torch.long, device=device),
                ])
                prompt_len_actual = actual_prompt.shape[0]

                if is_no_codec and not is_combined:
                    # Keep the row but with an all-zero mask: it acts as a
                    # group-baseline anchor (negative reward is injected after
                    # the reward request) without contributing direct policy
                    # gradient. Without this the failure mode is invisible.
                    # Length must match build_gen_token_mask (prompt + gen) so
                    # the packing slice in padded_gen_mask lines up.
                    token_mask = [0] * (prompt_len_actual + len(gen_ids))
                elif is_combined:
                    text_mask = build_gen_token_mask(
                        gen_ids, prompt_len_actual, "text",
                        s2st_text_id, s2st_eos_id, eos_id,
                        codec_offset, codec_size,
                    )
                    speech_mask = build_gen_token_mask(
                        gen_ids, prompt_len_actual, "speech",
                        s2st_text_id, s2st_eos_id, eos_id,
                        codec_offset, codec_size,
                    )
                    token_mask = [
                        int(text_on or speech_on)
                        for text_on, speech_on in zip(text_mask, speech_mask)
                    ]
                    if sum(token_mask) == 0:
                        continue
                else:
                    token_mask = build_gen_token_mask(
                        gen_ids, prompt_len_actual, mask_mode,
                        s2st_text_id, s2st_eos_id, eos_id,
                        codec_offset, codec_size,
                    )
                    if sum(token_mask) == 0:
                        continue

                all_full_ids.append(full)
                all_feat_idx.append(i)
                all_gen_masks.append(token_mask)
                if is_combined:
                    all_text_masks.append(text_mask)
                    all_speech_masks.append(speech_mask)
                # Pad to >=1 so reward server math (e.g. isometric ratios)
                # can run without div-by-zero on no-codec rows; the reward
                # is overridden post-hoc anyway.
                all_speech_lens.append(1 if is_no_codec else len(speech_codes))
                all_src_lens.append(source_lengths[i].item())
                # The translation-quality reward models take
                # the dataset transcript as the source so ASR errors don't
                # leak into the reward; the model-generated translation is
                # the hypothesis.
                if uses_text_reward:
                    all_source_texts.append(source_texts[i])
                    if is_no_codec and not is_combined:
                        all_trans_texts.append(" ")  # placeholder; reward overridden
                    else:
                        all_trans_texts.append(tok.decode(trans_ids, skip_special_tokens=True))
                    all_reference_texts.append(target_texts[i])
                if uses_waveform_reward:
                    all_audio_paths.append(audio_paths[i])
                    all_source_silences.append(source_vads[i])
                    all_speech_codes.append(list(speech_codes))
                no_codec_flags.append(is_no_codec)
                valid_count += 1

            if valid_count >= 2:
                group_sizes.append(valid_count)
            else:
                for _ in range(valid_count):
                    all_full_ids.pop()
                    all_feat_idx.pop()
                    all_gen_masks.pop()
                    if is_combined:
                        all_text_masks.pop()
                        all_speech_masks.pop()
                    all_speech_lens.pop()
                    all_src_lens.pop()
                    if uses_text_reward:
                        all_source_texts.pop()
                        all_trans_texts.pop()
                        all_reference_texts.pop()
                    if uses_waveform_reward:
                        all_audio_paths.pop()
                        all_source_silences.pop()
                        all_speech_codes.pop()
                    no_codec_flags.pop()

    # Always call the reward server so gather/scatter stays lockstep.
    if timings is not None:
        with _PhaseTimer(device, timings, "reward_wait"):
            reward_result = reward_server.request(
                source_texts=all_source_texts,
                trans_texts=all_trans_texts,
                reference_texts=all_reference_texts,
                src_lens=all_src_lens,
                speech_lens=all_speech_lens,
                audio_paths=all_audio_paths,
                source_silences=all_source_silences,
                speech_codes=all_speech_codes,
                device=device,
                group_sizes=group_sizes,
            )
    else:
        reward_result = reward_server.request(
            source_texts=all_source_texts,
            trans_texts=all_trans_texts,
            reference_texts=all_reference_texts,
            src_lens=all_src_lens,
            speech_lens=all_speech_lens,
            audio_paths=all_audio_paths,
            source_silences=all_source_silences,
            speech_codes=all_speech_codes,
            device=device,
            group_sizes=group_sizes,
        )

    component_rewards: Dict[str, torch.Tensor] = {}
    pause_duration_gated: Optional[torch.Tensor] = None
    if is_combined:
        if not isinstance(reward_result, dict):
            raise TypeError("combined RewardServer must return component scores")
        component_rewards = reward_result

        component_rewards["pause"], pause_duration_gated = (
            gate_pause_reward_by_duration(
                component_rewards["pause"],
                all_src_lens,
                all_speech_lens,
                cfg.speech_token_rate,
                cfg.pause_duration_tolerance,
                minimum_reward=cfg.pause_duration_gate_floor,
            )
        )

        # Codec-less samples can still carry a valid translation and therefore
        # keep their text mask. Give only the pause component an explicit
        # negative baseline: duration already saturates naturally at -4 for
        # the one-token placeholder, so overriding it with -2 would make the
        # failure look artificially better on the duration objective.
        if any(no_codec_flags):
            nc_idx = torch.tensor(
                [k for k, flag in enumerate(no_codec_flags) if flag],
                dtype=torch.long, device=device,
            )
            component_rewards["pause"] = component_rewards["pause"].clone()
            component_rewards["pause"][nc_idx] = cfg.no_codec_reward
            # Track no-codec separately; this gate metric is for otherwise
            # valid speech whose duration misses the ±tolerance band.
            pause_duration_gated[nc_idx] = False

        positive_weights = [
            ("translation_quality", cfg.quality_weight),
            ("pause", cfg.pause_weight),
            ("duration", cfg.duration_weight),
        ]
        total_weight = sum(float(w) for _, w in positive_weights if w > 0)
        if total_weight <= 0:
            raise ValueError("combined reward needs at least one positive component weight")
        rewards = sum(
            float(w) * component_rewards[name]
            for name, w in positive_weights if w > 0
        ) / total_weight

        # Extreme selection is based on a scale-free composite of group-wise
        # component advantages.  Component rewards themselves stay untouched
        # for their separate policy losses.
        if cfg.extreme_filter and rewards.numel() > 0:
            keep = compute_extreme_mask(
                combined_group_score(component_rewards, group_sizes, cfg),
                group_sizes,
                top_k=cfg.keep_top_k,
                bottom_k=cfg.keep_bottom_k,
            )
            rewards = rewards.clone()
            rewards[~keep] = float("nan")
    else:
        if not isinstance(reward_result, torch.Tensor):
            raise TypeError("single RewardServer must return one reward tensor")
        rewards = reward_result

    reward_metrics = reward_server.get_last_reward_metrics()
    if is_combined:
        reward_metrics["translation_quality_reward"] = (
            component_rewards["translation_quality"].detach().float().cpu().tolist()
        )
        reward_metrics["pause_reward"] = (
            component_rewards["pause"].detach().float().cpu().tolist()
        )
        reward_metrics["duration_reward"] = (
            component_rewards["duration"].detach().float().cpu().tolist()
        )
        reward_metrics["pause_duration_gated"] = (
            pause_duration_gated.detach().float().cpu().tolist()
        )
    if not _reward_metrics_are_aligned(reward_metrics, int(rewards.numel())):
        reward_metrics = {}

    # The translation-quality reward server can mark oversampled middle
    # candidates as NaN after COMET/MetricX consensus ranking.  Drop them
    # before no-codec override and before packing for the policy forward.
    finite_mask = torch.isfinite(rewards)
    if rewards.numel() > 0 and not bool(finite_mask.all().item()):
        finite = finite_mask.detach().cpu().tolist()
        kept_idx: List[int] = []
        kept_groups: List[int] = []
        offset = 0
        for g in group_sizes:
            group_keep = [offset + k for k in range(g) if finite[offset + k]]
            if len(group_keep) >= 2:
                kept_idx.extend(group_keep)
                kept_groups.append(len(group_keep))
            offset += g

        def _take_finite(lst):
            return [lst[i] for i in kept_idx]

        all_full_ids = _take_finite(all_full_ids)
        all_feat_idx = _take_finite(all_feat_idx)
        all_gen_masks = _take_finite(all_gen_masks)
        if is_combined:
            all_text_masks = _take_finite(all_text_masks)
            all_speech_masks = _take_finite(all_speech_masks)
        all_speech_lens = _take_finite(all_speech_lens)
        all_src_lens = _take_finite(all_src_lens)
        if uses_text_reward:
            all_source_texts = _take_finite(all_source_texts)
            all_trans_texts = _take_finite(all_trans_texts)
            all_reference_texts = _take_finite(all_reference_texts)
        if uses_waveform_reward:
            all_audio_paths = _take_finite(all_audio_paths)
            all_source_silences = _take_finite(all_source_silences)
            all_speech_codes = _take_finite(all_speech_codes)
        no_codec_flags = _take_finite(no_codec_flags)
        if kept_idx:
            kept_tensor = torch.tensor(
                kept_idx, dtype=torch.long, device=rewards.device,
            )
            rewards = rewards[kept_tensor]
            if is_combined:
                component_rewards = {
                    name: values[kept_tensor]
                    for name, values in component_rewards.items()
                }
            reward_metrics = _take_reward_metric_rows(reward_metrics, kept_idx)
        else:
            rewards = rewards[:0]
            if is_combined:
                component_rewards = {
                    name: values[:0]
                    for name, values in component_rewards.items()
                }
            reward_metrics = {}
        group_sizes = kept_groups

    # Negative rewards on no-codec rows shift the group baseline. Their
    # all-zero gen_mask prevents a direct gradient on the failed row.
    if any(no_codec_flags) and not is_combined:
        no_codec_idx = [k for k, f in enumerate(no_codec_flags) if f]
        nc_idx = torch.tensor(
            no_codec_idx,
            dtype=torch.long, device=rewards.device,
        )
        rewards = rewards.clone()
        rewards[nc_idx] = cfg.no_codec_reward
        if "pause_reward" in reward_metrics:
            for k in no_codec_idx:
                reward_metrics["pause_reward"][k] = float(cfg.no_codec_reward)

    # Snapshot the logged means BEFORE the zero-signal filter below, so
    # train/* and eval/* metrics (including the group-max ceiling
    # diagnostics) cover every finite sampled candidate.  The filter only
    # decides what is worth a policy forward, not what is measured —
    # otherwise all-tied groups (e.g. all-no-pause early in training, or
    # all-perfect near convergence) silently vanish from the metrics and
    # bias the ceiling readout.
    if "pause_reward" in reward_metrics:
        _add_group_ceiling_metrics(reward_metrics, group_sizes)
    reward_metric_means = _summarize_reward_metrics(reward_metrics)
    n_groups_before_signal_filter = len(group_sizes)

    # Drop groups whose rewards are (near-)identical — they yield zero GSPO
    # advantage and would otherwise dilute n_valid and contribute KL gradient
    # without policy-loss signal. Saves the policy/ref forward on those rows.
    # Eval disables the drop (cfg.drop_zero_signal_groups=False): there is no
    # policy forward to save, and dropping all-tied groups would condition
    # every eval mean on within-group variance (a shifting subpopulation).
    n_zero_signal_groups = 0
    if group_sizes:
        kept_idx: List[int] = []
        kept_groups: List[int] = []
        offset = 0
        for g in group_sizes:
            if is_combined:
                has_signal = any(
                    values[offset : offset + g].std(unbiased=False).item() > 1e-8
                    for values in component_rewards.values()
                )
            else:
                has_signal = (
                    rewards[offset : offset + g].std(unbiased=False).item() > 1e-8
                )
            if has_signal:
                kept_idx.extend(range(offset, offset + g))
                kept_groups.append(g)
            else:
                n_zero_signal_groups += 1
            offset += g
        if cfg.drop_zero_signal_groups and len(kept_idx) < rewards.numel():
            def _take(lst):
                return [lst[i] for i in kept_idx]
            all_full_ids = _take(all_full_ids)
            all_feat_idx = _take(all_feat_idx)
            all_gen_masks = _take(all_gen_masks)
            if is_combined:
                all_text_masks = _take(all_text_masks)
                all_speech_masks = _take(all_speech_masks)
            all_speech_lens = _take(all_speech_lens)
            all_src_lens = _take(all_src_lens)
            if uses_text_reward:
                all_source_texts = _take(all_source_texts)
                all_trans_texts = _take(all_trans_texts)
                all_reference_texts = _take(all_reference_texts)
            if uses_waveform_reward:
                all_audio_paths = _take(all_audio_paths)
                all_source_silences = _take(all_source_silences)
                all_speech_codes = _take(all_speech_codes)
            no_codec_flags = _take(no_codec_flags)
            if kept_idx:
                kept_tensor = torch.tensor(
                    kept_idx, dtype=torch.long, device=rewards.device,
                )
                rewards = rewards[kept_tensor]
                if is_combined:
                    component_rewards = {
                        name: values[kept_tensor]
                        for name, values in component_rewards.items()
                    }
                reward_metrics = _take_reward_metric_rows(reward_metrics, kept_idx)
            else:
                rewards = rewards[:0]
                if is_combined:
                    component_rewards = {
                        name: values[:0]
                        for name, values in component_rewards.items()
                    }
                reward_metrics = {}
            group_sizes = kept_groups
    if n_groups_before_signal_filter > 0:
        # Stall visibility: fraction of this batch's groups with zero
        # within-group reward variance (nothing to train on).
        reward_metric_means["pause_zero_signal_group_frac"] = (
            n_zero_signal_groups / n_groups_before_signal_filter
        )

    if not all_full_ids:
        return None

    N = len(all_full_ids)
    max_seq_len = max(s.shape[0] for s in all_full_ids)
    padded_ids = torch.full((N, max_seq_len), tok.pad_token_id, dtype=torch.long, device=device)
    padded_attn = torch.zeros(N, max_seq_len, dtype=torch.long, device=device)
    padded_gen_mask = torch.zeros(N, max_seq_len, dtype=torch.long, device=device)
    padded_text_mask = None
    padded_speech_mask = None
    if is_combined:
        padded_text_mask = torch.zeros_like(padded_gen_mask)
        padded_speech_mask = torch.zeros_like(padded_gen_mask)

    for n in range(N):
        L = all_full_ids[n].shape[0]
        off = max_seq_len - L
        padded_ids[n, off:] = all_full_ids[n]
        padded_attn[n, off:] = 1
        padded_gen_mask[n, off:] = torch.tensor(all_gen_masks[n], dtype=torch.long, device=device)
        if is_combined:
            padded_text_mask[n, off:] = torch.tensor(
                all_text_masks[n], dtype=torch.long, device=device,
            )
            padded_speech_mask[n, off:] = torch.tensor(
                all_speech_masks[n], dtype=torch.long, device=device,
            )

    max_feat_len = feature_attention_mask.shape[1]
    n_mels = input_features.shape[1]
    padded_feats = torch.zeros(N, n_mels, max_feat_len, device=device, dtype=input_features.dtype)
    padded_feat_mask = torch.zeros(N, max_feat_len, dtype=torch.long, device=device)
    for n in range(N):
        idx = all_feat_idx[n]
        padded_feats[n] = input_features[idx]
        padded_feat_mask[n] = feature_attention_mask[idx]

    out = {
        "padded_ids": padded_ids,
        "padded_attn": padded_attn,
        "padded_feats": padded_feats,
        "padded_feat_mask": padded_feat_mask,
        "gen_mask": padded_gen_mask,
        "rewards": rewards,
        "group_sizes": group_sizes,
        "reward_metric_means": reward_metric_means,
    }
    if is_combined:
        out["text_gen_mask"] = padded_text_mask
        out["speech_gen_mask"] = padded_speech_mask
        out["component_rewards"] = component_rewards
    return out


# ---------------------------------------------------------------------------
# training step
# ---------------------------------------------------------------------------
def _compute_ce_loss(policy_model, ce, cfg: GSPOConfig, device):
    """Return the auxiliary CE loss and number of labelled sequences."""
    with _model_compute_context(device, cfg.compute_dtype):
        ce_out = policy_model(
            input_ids=ce["input_ids"],
            attention_mask=ce["attention_mask"],
            input_features=ce["input_features"],
            feature_attention_mask=ce["feature_attention_mask"],
        )
    shifted = ce_out.logits[:, :-1].float()
    tgt = ce["labels"][:, 1:]
    per_tok = F.cross_entropy(
        shifted.reshape(-1, shifted.size(-1)),
        tgt.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(tgt.shape)
    tok_mask = (tgt != -100).float()
    n_tok_per_seq = tok_mask.sum(-1)
    seq_nll = (per_tok * tok_mask).sum(-1) / n_tok_per_seq.clamp(min=1)
    has_label = (n_tok_per_seq > 0).float()
    n_seq = has_label.sum().clamp(min=1)
    ce_loss = (seq_nll * has_label).sum() / n_seq
    return ce_loss, has_label.sum().item()


def gspo_train_step(policy_model, ref_model, gen_data, cfg: GSPOConfig, device):
    """Policy forward + (optional) ref forward + GSPO loss.

    On-policy (sync=1): gen_lp = pol_lp.detach(). When cfg.kl_coef > 0 the
    frozen ref model is also forwarded to compute the KL penalty.
    """
    padded_ids = gen_data["padded_ids"]
    padded_attn = gen_data["padded_attn"]
    padded_feats = gen_data["padded_feats"]
    padded_feat_mask = gen_data["padded_feat_mask"]
    gen_mask = gen_data["gen_mask"]
    rewards = gen_data["rewards"]
    group_sizes = gen_data["group_sizes"]

    is_combined = cfg.reward_type == COMBINED_REWARD_TYPE
    if is_combined:
        # Exclude a modality from both policy and KL when all of its component
        # weights are disabled.
        active_mask = torch.zeros_like(gen_mask)
        if cfg.quality_weight > 0:
            active_mask = torch.maximum(active_mask, gen_data["text_gen_mask"])
        if cfg.pause_weight > 0 or cfg.duration_weight > 0:
            active_mask = torch.maximum(active_mask, gen_data["speech_gen_mask"])
        gen_mask = active_mask

    # Restrict the full-vocab softmax to positions that contribute to the loss.
    lp_mask = gen_mask[:, 1:]

    if cfg.kl_coef > 0 and ref_model is not None:
        with torch.no_grad():
            with _model_compute_context(device, cfg.compute_dtype):
                ref_out = ref_model(
                    input_ids=padded_ids, attention_mask=padded_attn,
                    input_features=padded_feats, feature_attention_mask=padded_feat_mask,
                )
            ref_lp = compute_token_log_probs(ref_out.logits, padded_ids, mask=lp_mask)
    else:
        ref_lp = None

    with _model_compute_context(device, cfg.compute_dtype):
        policy_out = policy_model(
            input_ids=padded_ids, attention_mask=padded_attn,
            input_features=padded_feats, feature_attention_mask=padded_feat_mask,
        )
    pol_lp = compute_token_log_probs(policy_out.logits, padded_ids, mask=lp_mask)

    if is_combined:
        component_specs = (
            (
                "quality", "translation_quality", cfg.quality_weight,
                gen_data["text_gen_mask"],
            ),
            ("pause", "pause", cfg.pause_weight, gen_data["speech_gen_mask"]),
            ("duration", "duration", cfg.duration_weight, gen_data["speech_gen_mask"]),
        )
        weighted_policy_loss = torch.zeros((), device=device)
        total_weight = 0.0
        component_metrics: Dict[str, Dict[str, float]] = {}
        for log_name, reward_name, weight, component_mask in component_specs:
            if weight <= 0:
                continue
            component_loss, component_stat = compute_gspo_loss(
                pol_lp, pol_lp.detach(), None,
                gen_data["component_rewards"][reward_name],
                component_mask, group_sizes,
                cfg.clip_eps_low, cfg.clip_eps_high,
                0.0, cfg.adv_std_floor, cfg.adv_clip,
                cfg.adv_recenter_after_clip,
            )
            weighted_policy_loss = weighted_policy_loss + float(weight) * component_loss
            total_weight += float(weight)
            component_metrics[log_name] = component_stat
        if total_weight <= 0:
            raise ValueError("combined reward needs at least one positive component weight")
        policy_loss = weighted_policy_loss / total_weight
        kl_mean = compute_masked_kl(pol_lp, ref_lp, gen_mask)
        loss = policy_loss + cfg.kl_coef * kl_mean

        def _weighted_metric(name: str) -> float:
            return sum(
                float(weight) * component_metrics[log_name][name]
                for log_name, _reward_name, weight, _mask in component_specs
                if weight > 0
            ) / total_weight

        union_valid = (gen_mask[:, 1:].sum(-1) > 0).sum().item()
        loss_metrics = {
            "policy_loss": policy_loss.item(),
            "kl_mean": kl_mean.item(),
            "kl_loss": (cfg.kl_coef * kl_mean).item(),
            "advantage_mean": _weighted_metric("advantage_mean"),
            "advantage_std": _weighted_metric("advantage_std"),
            "adv_std_floor": float(cfg.adv_std_floor),
            "adv_clip": float(cfg.adv_clip),
            "ratio_mean": _weighted_metric("ratio_mean"),
            "clip_fraction": _weighted_metric("clip_fraction"),
            "n_valid": float(union_valid),
        }
        for log_name, stats in component_metrics.items():
            for metric_name in (
                "policy_loss", "advantage_mean", "advantage_std",
                "ratio_mean", "clip_fraction", "n_valid",
            ):
                loss_metrics[f"{log_name}_{metric_name}"] = stats[metric_name]
    else:
        loss, loss_metrics = compute_gspo_loss(
            pol_lp, pol_lp.detach(), ref_lp,
            rewards, gen_mask, group_sizes,
            cfg.clip_eps_low, cfg.clip_eps_high,
            cfg.kl_coef, cfg.adv_std_floor, cfg.adv_clip,
            cfg.adv_recenter_after_clip,
        )

    metrics = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "num_samples": rewards.shape[0],
        "num_groups": len(group_sizes),
        **loss_metrics,
    }
    metrics.update(gen_data.get("reward_metric_means", {}))

    if cfg.ce_coef > 0 and gen_data.get("ce_batch") is not None:
        ce_loss, ce_n_seq = _compute_ce_loss(
            policy_model, gen_data["ce_batch"], cfg, device,
        )
        loss = loss + cfg.ce_coef * ce_loss
        metrics["ce_loss"] = ce_loss.item()
        metrics["ce_n_seq"] = ce_n_seq

    return loss, metrics


_ADDITIVE_GSPO_METRICS = (
    "policy_loss", "kl_mean", "kl_loss", "advantage_mean",
    "ratio_mean", "clip_fraction", "n_valid",
)


def _accumulate_gspo_metrics(total: Dict[str, float], part: Dict[str, float]):
    for key in _ADDITIVE_GSPO_METRICS:
        total[key] = total.get(key, 0.0) + float(part[key])


def _backward_sync_context(policy_model, should_sync: bool):
    """Use DDP no_sync for every backward except the optimizer-step's last."""
    if should_sync or not hasattr(policy_model, "no_sync"):
        return nullcontext()
    return policy_model.no_sync()


def gspo_train_step_microbatched(
    policy_model,
    ref_model,
    gen_data,
    cfg: GSPOConfig,
    device,
    *,
    loss_scale: float,
    sync_gradients: bool,
):
    """Backward a retained rollout batch in candidate micro-chunks.

    Reward standardisation and all valid-sequence denominators are computed on
    the complete retained batch before slicing.  Each chunk loss is therefore
    an additive piece of the unsplit GSPO objective.  Backward happens before
    moving to the next chunk, which releases its logits/activations while all
    retained candidates still contribute to the same optimizer update.
    """
    rewards = gen_data["rewards"]
    group_sizes = gen_data["group_sizes"]
    n_samples = int(rewards.shape[0])
    chunk_size = int(cfg.train_micro_batch_size)
    if chunk_size <= 0 or chunk_size >= n_samples:
        raise ValueError(
            "candidate micro-batching requires 0 < train_micro_batch_size "
            f"< num retained samples; got {chunk_size} vs {n_samples}"
        )
    if sum(group_sizes) != n_samples:
        raise ValueError(
            f"group_sizes sum to {sum(group_sizes)}, expected {n_samples}"
        )

    is_combined = cfg.reward_type == COMBINED_REWARD_TYPE
    gen_mask = gen_data["gen_mask"]
    if is_combined:
        active_mask = torch.zeros_like(gen_mask)
        if cfg.quality_weight > 0:
            active_mask = torch.maximum(active_mask, gen_data["text_gen_mask"])
        if cfg.pause_weight > 0 or cfg.duration_weight > 0:
            active_mask = torch.maximum(active_mask, gen_data["speech_gen_mask"])
        gen_mask = active_mask

    def _valid_count(mask: torch.Tensor) -> float:
        return float((mask[:, 1:].sum(-1) > 0).sum().item())

    component_specs = ()
    component_advantages: Dict[str, torch.Tensor] = {}
    component_normalizers: Dict[str, float] = {}
    component_metrics: Dict[str, Dict[str, float]] = {}
    if is_combined:
        component_specs = (
            (
                "quality", "translation_quality", cfg.quality_weight,
                gen_data["text_gen_mask"],
            ),
            ("pause", "pause", cfg.pause_weight, gen_data["speech_gen_mask"]),
            ("duration", "duration", cfg.duration_weight, gen_data["speech_gen_mask"]),
        )
        for log_name, reward_name, weight, component_mask in component_specs:
            if weight <= 0:
                continue
            component_advantages[log_name] = _group_standardize_rewards(
                gen_data["component_rewards"][reward_name],
                group_sizes,
                cfg.adv_std_floor,
                cfg.adv_clip,
                cfg.adv_recenter_after_clip,
            )
            component_normalizers[log_name] = _valid_count(component_mask)
            component_metrics[log_name] = {}
        total_weight = sum(
            float(weight) for _log_name, _reward_name, weight, _mask in component_specs
            if weight > 0
        )
        union_normalizer = _valid_count(gen_mask)
    else:
        advantages = _group_standardize_rewards(
            rewards, group_sizes, cfg.adv_std_floor, cfg.adv_clip,
            cfg.adv_recenter_after_clip,
        )
        single_normalizer = _valid_count(gen_mask)
        single_metrics: Dict[str, float] = {}

    has_ce = cfg.ce_coef > 0 and gen_data.get("ce_batch") is not None
    kl_normalizer = _valid_count(gen_mask)
    total_loss_value = 0.0
    kl_mean_value = 0.0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_gen_mask = gen_mask[start:end]
        lp_mask = chunk_gen_mask[:, 1:]
        is_last_rl_chunk = end == n_samples
        should_sync = sync_gradients and is_last_rl_chunk and not has_ce

        with _backward_sync_context(policy_model, should_sync):
            if cfg.kl_coef > 0 and ref_model is not None:
                with torch.no_grad():
                    with _model_compute_context(device, cfg.compute_dtype):
                        ref_out = ref_model(
                            input_ids=gen_data["padded_ids"][start:end],
                            attention_mask=gen_data["padded_attn"][start:end],
                            input_features=gen_data["padded_feats"][start:end],
                            feature_attention_mask=gen_data["padded_feat_mask"][start:end],
                        )
                    ref_lp = compute_token_log_probs(
                        ref_out.logits,
                        gen_data["padded_ids"][start:end],
                        mask=lp_mask,
                    )
                    del ref_out
            else:
                ref_lp = None

            with _model_compute_context(device, cfg.compute_dtype):
                policy_out = policy_model(
                    input_ids=gen_data["padded_ids"][start:end],
                    attention_mask=gen_data["padded_attn"][start:end],
                    input_features=gen_data["padded_feats"][start:end],
                    feature_attention_mask=gen_data["padded_feat_mask"][start:end],
                )
            pol_lp = compute_token_log_probs(
                policy_out.logits,
                gen_data["padded_ids"][start:end],
                mask=lp_mask,
            )

            if is_combined:
                weighted_policy_loss = torch.zeros((), device=device)
                chunk_component_stats: Dict[str, Dict[str, float]] = {}
                for log_name, _reward_name, weight, component_mask in component_specs:
                    if weight <= 0:
                        continue
                    component_loss, stats = _compute_gspo_loss_from_advantages(
                        pol_lp,
                        pol_lp.detach(),
                        None,
                        component_advantages[log_name][start:end],
                        component_mask[start:end],
                        cfg.clip_eps_low,
                        cfg.clip_eps_high,
                        0.0,
                        cfg.adv_std_floor,
                        cfg.adv_clip,
                        n_valid_normalizer=component_normalizers[log_name],
                    )
                    weighted_policy_loss = (
                        weighted_policy_loss + float(weight) * component_loss
                    )
                    chunk_component_stats[log_name] = stats
                policy_loss = weighted_policy_loss / total_weight
                chunk_kl = compute_masked_kl(
                    pol_lp,
                    ref_lp,
                    chunk_gen_mask,
                    n_valid_normalizer=kl_normalizer,
                )
                chunk_loss = policy_loss + cfg.kl_coef * chunk_kl
            else:
                chunk_loss, chunk_stats = _compute_gspo_loss_from_advantages(
                    pol_lp,
                    pol_lp.detach(),
                    ref_lp,
                    advantages[start:end],
                    chunk_gen_mask,
                    cfg.clip_eps_low,
                    cfg.clip_eps_high,
                    cfg.kl_coef,
                    cfg.adv_std_floor,
                    cfg.adv_clip,
                    n_valid_normalizer=single_normalizer,
                )

            chunk_loss_value = chunk_loss.item()
            (chunk_loss * float(loss_scale)).backward()

        total_loss_value += chunk_loss_value
        if is_combined:
            kl_mean_value += chunk_kl.item()
            for log_name, stats in chunk_component_stats.items():
                _accumulate_gspo_metrics(component_metrics[log_name], stats)
        else:
            _accumulate_gspo_metrics(single_metrics, chunk_stats)
        del policy_out, pol_lp, ref_lp, chunk_loss
        if is_combined:
            # These locals otherwise keep the just-backwarded graph alive until
            # the next loop assignment, defeating candidate micro-chunking's
            # activation/logit memory reduction.
            del weighted_policy_loss, component_loss, policy_loss, chunk_kl

    ce_loss_value = None
    ce_n_seq = None
    if has_ce:
        with _backward_sync_context(policy_model, sync_gradients):
            ce_loss, ce_n_seq = _compute_ce_loss(
                policy_model, gen_data["ce_batch"], cfg, device,
            )
            weighted_ce_loss = cfg.ce_coef * ce_loss
            ce_loss_value = ce_loss.item()
            (weighted_ce_loss * float(loss_scale)).backward()
        total_loss_value += cfg.ce_coef * ce_loss_value

    if is_combined:
        for log_name, _reward_name, weight, _mask in component_specs:
            if weight <= 0:
                continue
            adv = component_advantages[log_name]
            component_metrics[log_name]["advantage_std"] = (
                adv.std().item() if adv.numel() > 1 else 0.0
            )
            component_metrics[log_name]["adv_std_floor"] = float(cfg.adv_std_floor)
            component_metrics[log_name]["adv_clip"] = float(cfg.adv_clip)

        def _weighted_metric(name: str) -> float:
            return sum(
                float(weight) * component_metrics[log_name][name]
                for log_name, _reward_name, weight, _mask in component_specs
                if weight > 0
            ) / total_weight

        policy_loss_value = _weighted_metric("policy_loss")
        loss_metrics = {
            "policy_loss": policy_loss_value,
            "kl_mean": kl_mean_value,
            "kl_loss": cfg.kl_coef * kl_mean_value,
            "advantage_mean": _weighted_metric("advantage_mean"),
            "advantage_std": _weighted_metric("advantage_std"),
            "adv_std_floor": float(cfg.adv_std_floor),
            "adv_clip": float(cfg.adv_clip),
            "ratio_mean": _weighted_metric("ratio_mean"),
            "clip_fraction": _weighted_metric("clip_fraction"),
            "n_valid": union_normalizer,
        }
        for log_name, stats in component_metrics.items():
            for metric_name in (
                "policy_loss", "advantage_mean", "advantage_std",
                "ratio_mean", "clip_fraction", "n_valid",
            ):
                loss_metrics[f"{log_name}_{metric_name}"] = stats[metric_name]
    else:
        single_metrics["advantage_std"] = (
            advantages.std().item() if advantages.numel() > 1 else 0.0
        )
        single_metrics["adv_std_floor"] = float(cfg.adv_std_floor)
        single_metrics["adv_clip"] = float(cfg.adv_clip)
        loss_metrics = single_metrics

    metrics = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "num_samples": rewards.shape[0],
        "num_groups": len(group_sizes),
        **loss_metrics,
    }
    metrics.update(gen_data.get("reward_metric_means", {}))
    if ce_loss_value is not None:
        metrics["ce_loss"] = ce_loss_value
        metrics["ce_n_seq"] = ce_n_seq
    return total_loss_value, metrics


# ---------------------------------------------------------------------------
# held-out evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    policy_model,
    eval_loader: DataLoader,
    eval_cfg: GSPOConfig,
    codec_offset: int,
    codec_size: int,
    tok,
    device,
    reward_server: RewardServer,
    train_group,
    train_world_size: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Generate on the dev set and return mean reward metrics, reduced across
    train ranks. No gradients, no optimizer step.

    Reuses generate_step so eval matches training exactly (same parsing, same
    reward server, same generation code path) — only the sampling knobs differ
    (carried by eval_cfg) and extreme_filter is forced OFF so the logged means
    cover *every* sampled candidate, not just the top/bottom-K kept for
    training. The canonical pause-metric key set is packed in a fixed order so
    the single cross-rank all_reduce stays lockstep even when a given rank/batch
    produced only a subset of the metrics.

    Lockstep note: each rank iterates the same number of dev batches (the
    DistributedSampler hands every rank an equal count, and max_batches caps
    them identically), so the per-batch reward requests (server reward types)
    and the final all_reduce stay in lockstep across all train ranks.
    """
    was_training = policy_model.training
    policy_model.eval()
    prev_extreme = reward_server.extreme_filter
    reward_server.extreme_filter = False

    # Evaluation uses fixed per-rank/per-batch sampling streams, then restores
    # every RNG so running eval does not perturb subsequent training samples.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    eval_rank = dist.get_rank(group=train_group)
    rank_seed = int(eval_cfg.eval_seed) + eval_rank * 1_000_000

    metric_keys = list(PAUSE_REWARD_METRIC_LOG_NAMES.values())
    reward_sum = 0.0
    reward_n = 0.0
    n_batches = 0.0
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_wsums = {k: 0.0 for k in metric_keys}
    try:
        for bi, batch in enumerate(eval_loader):
            if max_batches > 0 and bi >= max_batches:
                break
            batch_seed = (rank_seed + bi) % (2**32)
            random.seed(batch_seed)
            np.random.seed(batch_seed)
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            n_batches += 1.0
            gen_data = generate_step(
                policy_model, batch, eval_cfg,
                codec_offset, codec_size, tok, device,
                reward_server=reward_server, timings=None,
            )
            if gen_data is None:
                continue
            rewards = gen_data["rewards"]
            w = float(rewards.numel())
            if w == 0.0:
                continue
            reward_sum += float(rewards.sum().item())
            reward_n += w
            for k, v in gen_data.get("reward_metric_means", {}).items():
                if k in metric_sums:
                    metric_sums[k] += float(v) * w
                    metric_wsums[k] += w
    finally:
        reward_server.extreme_filter = prev_extreme
        if was_training:
            policy_model.train()
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    # Single lockstep reduction across train ranks. Layout:
    #   [reward_sum, reward_n, n_batches, (sum_k, wsum_k) for k in metric_keys]
    flat = [reward_sum, reward_n, n_batches]
    for k in metric_keys:
        flat.append(metric_sums[k])
        flat.append(metric_wsums[k])
    t = torch.tensor(flat, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=train_group)
    vals = t.cpu().tolist()

    out: Dict[str, float] = {
        "reward_mean": (vals[0] / vals[1]) if vals[1] > 0 else 0.0,
        "n_samples": vals[1],
        "n_batches": vals[2],
    }
    idx = 3
    for k in metric_keys:
        s, w = vals[idx], vals[idx + 1]
        idx += 2
        if w > 0:
            out[k] = s / w
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _validate_required_init_checkpoint(config: DictConfig) -> None:
    """Enforce an experiment's declared fresh-start checkpoint.

    A later ``ckpt_last.pt`` in the experiment's own output directory may
    still resume that same trajectory. This guard only prevents silently
    changing the model used to start a fresh trajectory (for example, from an
    SFT checkpoint to an intermediate checkpoint from a different RL run).
    """
    checkpoint_cfg = config.get("checkpoint", {})
    required = checkpoint_cfg.get("required_init_checkpoint", "")
    if not required:
        return

    configured = checkpoint_cfg.get("resume_ckpt_path", "")
    if not configured:
        raise ValueError(
            "checkpoint.required_init_checkpoint is set, but "
            "checkpoint.resume_ckpt_path is empty"
        )

    required_path = os.path.realpath(os.fspath(required))
    configured_path = os.path.realpath(os.fspath(configured))
    if configured_path != required_path:
        raise ValueError(
            "initial-checkpoint invariant violated: this experiment must "
            f"start from {required_path}, got {configured_path}"
        )
    if not os.path.isfile(required_path):
        raise FileNotFoundError(
            f"required initial checkpoint not found: {required_path}"
        )

    if checkpoint_cfg.get("forbid_existing_rl_checkpoint", False):
        out_dir = os.path.join(
            os.fspath(config.common.output_dir), "checkpoints",
        )
        existing = (
            sorted(
                os.path.join(out_dir, name)
                for name in os.listdir(out_dir)
                if name.endswith(".pt")
            )
            if os.path.isdir(out_dir) else []
        )
        if existing:
            raise FileExistsError(
                "fresh-SFT invariant violated: this profile cannot resume an "
                f"existing RL checkpoint ({existing[-1]}). Choose a new RUN_NAME."
            )


@hydra.main(config_path="config", config_name="gspo_stage4_rl", version_base=None)
def main(config: DictConfig):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    _validate_required_init_checkpoint(config)

    cfg = GSPOConfig(
        num_samples=config.gspo.num_samples,
        max_new_tokens=config.gspo.max_new_tokens,
        temperature=config.gspo.temperature,
        top_p=config.gspo.top_p,
        top_k=config.gspo.top_k,
        num_gen_batches=config.gspo.get("num_gen_batches", 4),
        train_micro_batch_size=config.gspo.get("train_micro_batch_size", 0),
        speech_token_rate=config.gspo.get("speech_token_rate", 25.0),
        repetition_penalty=config.gspo.get("repetition_penalty", 1.0),
        compute_dtype=config.model.get("compute_dtype", "bfloat16"),
        speech_silence_tokens=tuple(config.gspo.get(
            "speech_silence_tokens",
            [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323],
        )),
        speech_occupancy_silence_tokens=(
            tuple(config.gspo["speech_occupancy_silence_tokens"])
            if config.gspo.get("speech_occupancy_silence_tokens") else None
        ),
        clip_eps_low=config.gspo.get("clip_eps_low", 0.2),
        clip_eps_high=config.gspo.get("clip_eps_high", 0.2),
        kl_coef=config.gspo.get("kl_coef", 0.0),
        kl_adaptive=config.gspo.get("kl_adaptive", False),
        kl_target=config.gspo.get("kl_target", 0.01),
        kl_horizon=config.gspo.get("kl_horizon", 10000.0),
        kl_coef_min=config.gspo.get("kl_coef_min", 1e-6),
        kl_coef_max=config.gspo.get("kl_coef_max", 1.0),
        reward_type=config.gspo.get("reward_type", "translation_quality"),
        quality_weight=config.gspo.get("quality_weight", 1.0),
        pause_weight=config.gspo.get("pause_weight", 1.0),
        duration_weight=config.gspo.get("duration_weight", 0.5),
        duration_eps_free=config.gspo.get("duration_eps_free", 0.05),
        duration_speech_only=config.gspo.get("duration_speech_only", False),
        duration_speech_ratio_center=config.gspo.get("duration_speech_ratio_center", 1.0),
        metricx_model_dir=config.gspo.get("metricx_model_dir", ""),
        metricx_repo=config.gspo.get("metricx_repo", ""),
        metricx_tokenizer=config.gspo.get("metricx_tokenizer", "google/mt5-xl"),
        metricx_max_input_length=config.gspo.get("metricx_max_input_length", 1536),
        metricx_max_score=config.gspo.get("metricx_max_score", 25.0),
        metricx_batch_size=config.gspo.get("metricx_batch_size", 16),
        token2wav_flow_model=config.gspo.get("token2wav_flow_model", ""),
        token2wav_hift_model=config.gspo.get("token2wav_hift_model", ""),
        token2wav_campplus_model=config.gspo.get("token2wav_campplus_model", ""),
        token2wav_speech_tokenizer_model=config.gspo.get(
            "token2wav_speech_tokenizer_model", "",
        ),
        token2wav_fp16=config.gspo.get("token2wav_fp16", False),
        token2wav_n_timesteps=config.gspo.get("token2wav_n_timesteps", 2),
        token2wav_cfg=config.gspo.get("token2wav_cfg", False),
        pause_vad_sample_rate=config.gspo.get("pause_vad_sample_rate", 16000),
        pause_vad_threshold=config.gspo.get("pause_vad_threshold", 0.8),
        pause_min_speech_ms=config.gspo.get("pause_min_speech_ms", 250),
        pause_min_silence_ms=config.gspo.get("pause_min_silence_ms", 100),
        pause_speech_pad_ms=config.gspo.get("pause_speech_pad_ms", 30),
        pause_backend=config.gspo.get("pause_backend", "token"),
        timeline_w_miss=config.gspo.get("timeline_w_miss", 1.5),
        timeline_w_fa=config.gspo.get("timeline_w_fa", 1.0),
        timeline_w_overhang=config.gspo.get("timeline_w_overhang", 2.0),
        timeline_w_drift=config.gspo.get("timeline_w_drift", 0.0),
        timeline_drift_tol=config.gspo.get("timeline_drift_tol", 3.0),
        pause_strict_min_silence_ms=config.gspo.get(
            "pause_strict_min_silence_ms", 400,
        ),
        pause_dense_min_silence_ms=config.gspo.get(
            "pause_dense_min_silence_ms", 200,
        ),
        pause_dense_weight=config.gspo.get("pause_dense_weight", 0.25),
        pause_collar=config.gspo.get("pause_collar", 0.10),
        pause_added_silence_lambda=config.gspo.get(
            "pause_added_silence_lambda", 0.3,
        ),
        pause_added_silence_cap=config.gspo.get("pause_added_silence_cap", 2.0),
        pause_added_free_fraction=config.gspo.get(
            "pause_added_free_fraction", 0.05,
        ),
        pause_silence_flood_ratio=config.gspo.get(
            "pause_silence_flood_ratio", 3.0,
        ),
        pause_bridge_gap_tokens=config.gspo.get("pause_bridge_gap_tokens", 2),
        pause_no_generated_penalty=config.gspo.get(
            "pause_no_generated_penalty", 0.0,
        ),
        pause_target_core_floor=config.gspo.get("pause_target_core_floor", 0.4),
        pause_duration_gate_tolerance=config.gspo.get(
            "pause_duration_gate_tolerance", 0.25,
        ),
        pause_duration_gate_floor=config.gspo.get(
            "pause_duration_gate_floor", -1.0,
        ),
        pause_drop_edge_silence=config.gspo.get("pause_drop_edge_silence", True),
        pause_no_speech_reward=config.gspo.get("pause_no_speech_reward", 0.0),
        pause_eps=config.gspo.get("pause_eps", 1e-6),
        pause_source_vad_from_audio=config.gspo.get(
            "pause_source_vad_from_audio", True,
        ),
        pause_duration_tolerance=config.gspo.get("pause_duration_tolerance", 0.10),
        quality_use_reference=config.gspo.get("quality_use_reference", True),
        adv_std_floor=config.gspo.get("adv_std_floor", 0.02),
        adv_clip=config.gspo.get("adv_clip", 2.0),
        adv_recenter_after_clip=config.gspo.get(
            "adv_recenter_after_clip", False,
        ),
        extreme_filter=config.gspo.get("extreme_filter", False),
        keep_top_k=config.gspo.get("keep_top_k", 4),
        keep_bottom_k=config.gspo.get("keep_bottom_k", 4),
        mask_mode=config.gspo.get("mask_mode", ""),
        ce_coef=config.gspo.get("ce_coef", 0.0),
        ce_every_n_microbatches=config.gspo.get("ce_every_n_microbatches", 1),
        no_codec_reward=config.gspo.get("no_codec_reward", -2.0),
        eval_every_n_steps=config.gspo.get("eval_every_n_steps", 0),
        eval_num_batches=config.gspo.get("eval_num_batches", 0),
        eval_num_samples=config.gspo.get("eval_num_samples", 8),
        eval_temperature=config.gspo.get("eval_temperature", 0.85),
        eval_top_p=config.gspo.get("eval_top_p", 0.85),
        eval_top_k=config.gspo.get("eval_top_k", 0),
        eval_seed=config.gspo.get("eval_seed", 42),
    )

    valid_reward_types = {
        "translation_quality", "isometric", "pause_only", COMBINED_REWARD_TYPE,
    }
    if cfg.reward_type not in valid_reward_types:
        raise ValueError(
            f"unknown reward_type={cfg.reward_type!r}; expected one of "
            f"{sorted(valid_reward_types)}"
        )
    if cfg.reward_type == COMBINED_REWARD_TYPE:
        component_weights = (
            cfg.quality_weight, cfg.pause_weight, cfg.duration_weight,
        )
        if any(weight < 0 for weight in component_weights):
            raise ValueError("combined component weights must be non-negative")
        if sum(component_weights) <= 0:
            raise ValueError("combined reward needs at least one positive component weight")
        if cfg.pause_duration_tolerance < 0:
            raise ValueError("pause_duration_tolerance must be non-negative")
        if cfg.mask_mode:
            raise ValueError(
                "combined reward builds separate text/speech masks; leave mask_mode empty"
            )
    if not 0.0 <= cfg.duration_eps_free < 0.10:
        raise ValueError("duration_eps_free must be in [0, 0.10)")
    if cfg.token2wav_n_timesteps <= 0:
        raise ValueError("token2wav_n_timesteps must be positive")
    if cfg.pause_backend not in ("token", "audio", "timeline"):
        raise ValueError("pause_backend must be 'token', 'audio' or 'timeline'")
    if min(
        cfg.pause_strict_min_silence_ms,
        cfg.pause_dense_min_silence_ms,
    ) < 0:
        raise ValueError("pause silence thresholds must be non-negative")
    if cfg.pause_dense_min_silence_ms > cfg.pause_strict_min_silence_ms:
        raise ValueError(
            "dense pause threshold cannot exceed the strict threshold"
        )
    if not 0.0 <= cfg.pause_dense_weight <= 1.0:
        raise ValueError("pause_dense_weight must be in [0, 1]")
    if min(
        cfg.pause_added_silence_lambda,
        cfg.pause_added_silence_cap,
        cfg.pause_added_free_fraction,
        cfg.pause_no_generated_penalty,
        cfg.pause_target_core_floor,
        cfg.pause_duration_gate_tolerance,
        cfg.pause_bridge_gap_tokens,
    ) < 0.0:
        raise ValueError("pause reward knobs must be non-negative")
    if cfg.pause_silence_flood_ratio < cfg.pause_added_silence_cap:
        raise ValueError(
            "pause_silence_flood_ratio must be >= pause_added_silence_cap"
        )
    if cfg.pause_backend in ("token", "timeline") and not cfg.speech_silence_tokens:
        raise ValueError("token/timeline pause backends require speech_silence_tokens")
    if cfg.train_micro_batch_size < 0:
        raise ValueError("train_micro_batch_size must be non-negative")

    # --- distributed setup ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Make model sampling and any process-local augmentation reproducible.
    # Training ranks use distinct streams; the per-microbatch generation seed
    # below additionally makes a resumed step reproduce its sampling stream.
    base_seed = int(config.common.get("seed", 42))
    process_seed = (base_seed + rank) % (2**32)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    # Local-mode rewards skip the dedicated reward server.
    # Other reward types still need rank 0 reserved for the heavy reward model.
    local_reward = cfg.reward_type in ("isometric", "pause_only")

    if local_reward:
        train_group = None  # WORLD
        train_world_size = world_size
        train_rank = rank
        is_chief = (rank == 0)
    else:
        if world_size < 2:
            raise RuntimeError(
                f"Need at least 2 ranks (1 for reward server + ≥1 for training), "
                f"got world_size={world_size}"
            )
        train_ranks = list(range(1, world_size))
        train_group = dist.new_group(ranks=train_ranks)
        train_world_size = world_size - 1
        train_rank = rank - 1
        is_chief = (rank == 1)

        # ============================================================
        # Rank 0 — dedicated reward server
        # ============================================================
        if rank == 0:
            reward_device = torch.device("cuda:0")
            torch.cuda.set_device(reward_device)
            print(
                f"[rank 0] Starting reward server "
                f"(reward_type={cfg.reward_type}, reward_device={reward_device})"
            )
            if cfg.reward_type in ("translation_quality", COMBINED_REWARD_TYPE):
                assert cfg.metricx_model_dir, (
                    f"metricx_model_dir must be set for reward_type='{cfg.reward_type}'"
                )
                assert cfg.metricx_repo, (
                    f"metricx_repo must be set for reward_type='{cfg.reward_type}'"
                )
            if cfg.reward_type in ("pause_only", COMBINED_REWARD_TYPE):
                assert cfg.token2wav_flow_model, (
                    f"token2wav_flow_model must be set for reward_type='{cfg.reward_type}'"
                )
                assert cfg.token2wav_hift_model, (
                    f"token2wav_hift_model must be set for reward_type='{cfg.reward_type}'"
                )
                assert cfg.token2wav_campplus_model, (
                    f"token2wav_campplus_model must be set for reward_type='{cfg.reward_type}'"
                )
                assert cfg.token2wav_speech_tokenizer_model, (
                    f"token2wav_speech_tokenizer_model must be set for reward_type='{cfg.reward_type}'"
                )
            server = RewardServer(
                rank=0,
                world_size=world_size,
                reward_type=cfg.reward_type,
                speech_token_rate=cfg.speech_token_rate,
                duration_eps_free=cfg.duration_eps_free,
                duration_speech_only=cfg.duration_speech_only,
                duration_speech_ratio_center=cfg.duration_speech_ratio_center,
                reward_device=str(reward_device),
                metricx_model_dir=cfg.metricx_model_dir,
                metricx_repo=cfg.metricx_repo,
                metricx_tokenizer=cfg.metricx_tokenizer,
                metricx_max_input_length=cfg.metricx_max_input_length,
                metricx_max_score=cfg.metricx_max_score,
                metricx_batch_size=cfg.metricx_batch_size,
                token2wav_flow_model=cfg.token2wav_flow_model,
                token2wav_hift_model=cfg.token2wav_hift_model,
                token2wav_campplus_model=cfg.token2wav_campplus_model,
                token2wav_speech_tokenizer_model=cfg.token2wav_speech_tokenizer_model,
                token2wav_fp16=cfg.token2wav_fp16,
                token2wav_n_timesteps=cfg.token2wav_n_timesteps,
                token2wav_cfg=cfg.token2wav_cfg,
                pause_backend=cfg.pause_backend,
                timeline_w_miss=cfg.timeline_w_miss,
                timeline_w_fa=cfg.timeline_w_fa,
                timeline_w_overhang=cfg.timeline_w_overhang,
                timeline_w_drift=cfg.timeline_w_drift,
                timeline_drift_tol=cfg.timeline_drift_tol,
                pause_silence_codes=cfg.speech_silence_tokens,
                pause_occupancy_silence_codes=cfg.speech_occupancy_silence_tokens,
                pause_vad_sample_rate=cfg.pause_vad_sample_rate,
                pause_vad_threshold=cfg.pause_vad_threshold,
                pause_min_speech_ms=cfg.pause_min_speech_ms,
                pause_min_silence_ms=cfg.pause_min_silence_ms,
                pause_speech_pad_ms=cfg.pause_speech_pad_ms,
                pause_strict_min_silence_ms=cfg.pause_strict_min_silence_ms,
                pause_dense_min_silence_ms=cfg.pause_dense_min_silence_ms,
                pause_drop_edge_silence=cfg.pause_drop_edge_silence,
                pause_no_speech_reward=cfg.pause_no_speech_reward,
                pause_collar=cfg.pause_collar,
                pause_dense_weight=cfg.pause_dense_weight,
                pause_added_silence_lambda=cfg.pause_added_silence_lambda,
                pause_added_silence_cap=cfg.pause_added_silence_cap,
                pause_added_free_fraction=cfg.pause_added_free_fraction,
                pause_silence_flood_ratio=cfg.pause_silence_flood_ratio,
                pause_bridge_gap_tokens=cfg.pause_bridge_gap_tokens,
                pause_no_generated_penalty=cfg.pause_no_generated_penalty,
                pause_target_core_floor=cfg.pause_target_core_floor,
                pause_duration_gate_tolerance=cfg.pause_duration_gate_tolerance,
                pause_duration_gate_floor=cfg.pause_duration_gate_floor,
                pause_eps=cfg.pause_eps,
                pause_source_vad_from_audio=cfg.pause_source_vad_from_audio,
                quality_use_reference=cfg.quality_use_reference,
                extreme_filter=cfg.extreme_filter,
                keep_top_k=cfg.keep_top_k,
                keep_bottom_k=cfg.keep_bottom_k,
            )
            print(f"[rank 0] Reward server ready, serving forever on {device}")
            server.serve_forever()
            print(f"[rank 0] Reward server shut down")
            dist.destroy_process_group()
            return

    # ============================================================
    # Ranks 1..world_size-1 — training
    # ============================================================
    reward_server = RewardServer(
        rank=rank,
        world_size=world_size,
        reward_type=cfg.reward_type,
        speech_token_rate=cfg.speech_token_rate,
        duration_eps_free=cfg.duration_eps_free,
        duration_speech_only=cfg.duration_speech_only,
        duration_speech_ratio_center=cfg.duration_speech_ratio_center,
        reward_device=str(device),
        token2wav_flow_model=cfg.token2wav_flow_model,
        token2wav_hift_model=cfg.token2wav_hift_model,
        token2wav_campplus_model=cfg.token2wav_campplus_model,
        token2wav_speech_tokenizer_model=cfg.token2wav_speech_tokenizer_model,
        token2wav_fp16=cfg.token2wav_fp16,
        token2wav_n_timesteps=cfg.token2wav_n_timesteps,
        token2wav_cfg=cfg.token2wav_cfg,
        pause_backend=cfg.pause_backend,
        timeline_w_miss=cfg.timeline_w_miss,
        timeline_w_fa=cfg.timeline_w_fa,
        timeline_w_overhang=cfg.timeline_w_overhang,
        timeline_w_drift=cfg.timeline_w_drift,
        timeline_drift_tol=cfg.timeline_drift_tol,
        pause_silence_codes=cfg.speech_silence_tokens,
        pause_occupancy_silence_codes=cfg.speech_occupancy_silence_tokens,
        pause_vad_sample_rate=cfg.pause_vad_sample_rate,
        pause_vad_threshold=cfg.pause_vad_threshold,
        pause_min_speech_ms=cfg.pause_min_speech_ms,
        pause_min_silence_ms=cfg.pause_min_silence_ms,
        pause_speech_pad_ms=cfg.pause_speech_pad_ms,
        pause_strict_min_silence_ms=cfg.pause_strict_min_silence_ms,
        pause_dense_min_silence_ms=cfg.pause_dense_min_silence_ms,
        pause_drop_edge_silence=cfg.pause_drop_edge_silence,
        pause_no_speech_reward=cfg.pause_no_speech_reward,
        pause_collar=cfg.pause_collar,
        pause_dense_weight=cfg.pause_dense_weight,
        pause_added_silence_lambda=cfg.pause_added_silence_lambda,
        pause_added_silence_cap=cfg.pause_added_silence_cap,
        pause_added_free_fraction=cfg.pause_added_free_fraction,
        pause_silence_flood_ratio=cfg.pause_silence_flood_ratio,
        pause_bridge_gap_tokens=cfg.pause_bridge_gap_tokens,
        pause_no_generated_penalty=cfg.pause_no_generated_penalty,
        pause_target_core_floor=cfg.pause_target_core_floor,
        pause_duration_gate_tolerance=cfg.pause_duration_gate_tolerance,
        pause_duration_gate_floor=cfg.pause_duration_gate_floor,
        pause_eps=cfg.pause_eps,
        pause_source_vad_from_audio=cfg.pause_source_vad_from_audio,
        extreme_filter=cfg.extreme_filter,
        keep_top_k=cfg.keep_top_k,
        keep_bottom_k=cfg.keep_bottom_k,
    )
    if is_chief and cfg.train_micro_batch_size > 0:
        print(
            "[candidate-microbatch] "
            f"train_micro_batch_size={cfg.train_micro_batch_size}; "
            f"num_samples={cfg.num_samples}, "
            f"keep_top_k={cfg.keep_top_k}, keep_bottom_k={cfg.keep_bottom_k} "
            "remain unchanged"
        )

    # --- wandb ---
    wandb_cfg = config.get("wandb", {})
    use_wandb = is_chief and wandb_cfg.get("enabled", False)
    if use_wandb:
        prior_run_id = read_wandb_run_id(config.common.output_dir)
        init_kwargs = dict(
            project=wandb_cfg.get("project", "TransDubber"),
            name=wandb_cfg.get("name", None),
            group=wandb_cfg.get("group", None),
            tags=list(wandb_cfg.get("tags", [])),
            config=OmegaConf.to_container(config, resolve=True),
        )
        if prior_run_id:
            init_kwargs["id"] = prior_run_id
            init_kwargs["resume"] = "allow"
            print(f"[wandb] resuming run id={prior_run_id}")
        run = wandb.init(**init_kwargs)
        if not prior_run_id and run is not None:
            write_wandb_run_id(config.common.output_dir, run.id)

    # --- frozen reference model (needed when kl_coef > 0 OR kl_adaptive) ---
    needs_ref = cfg.kl_coef > 0 or cfg.kl_adaptive
    ref_model = None
    if needs_ref:
        if is_chief:
            print("Loading reference model (frozen)...")
        ref_model, _ = _build_model(config, device, freeze=True)
        if is_chief:
            print(f"Reference model loaded on {device}")
    else:
        if is_chief:
            print("kl_coef=0 and kl_adaptive=False, skipping reference model")

    # --- adaptive KL controller (chief drives, all ranks track via broadcast) ---
    kl_controller: Optional[AdaptiveKLController] = None
    if cfg.kl_adaptive:
        kl_controller = AdaptiveKLController(
            init_kl_coef=cfg.kl_coef,
            target=cfg.kl_target,
            horizon=cfg.kl_horizon,
            coef_min=cfg.kl_coef_min,
            coef_max=cfg.kl_coef_max,
        )
        if is_chief:
            print(
                f"[adaptive-kl] target={cfg.kl_target} horizon={cfg.kl_horizon} "
                f"init_coef={cfg.kl_coef} bounds=[{cfg.kl_coef_min},{cfg.kl_coef_max}]"
            )

    # --- trainable policy model ---
    if is_chief:
        print("Loading policy model...")
    policy_model, processor = _build_model(config, device, freeze=False)
    tok = processor.tokenizer
    codec_offset = config.model.speech_codec_offset
    codec_size = config.model.speech_codec_size
    if is_chief:
        print(f"Policy model loaded on {device}")

    policy_ddp = DDP(
        policy_model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        process_group=train_group,
    )

    # --- optimizer ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, policy_ddp.parameters()),
        lr=config.optimization.lr,
        betas=(
            config.optimization.optimizer.adam_betas_0,
            config.optimization.optimizer.adam_betas_1,
        ),
        eps=config.optimization.optimizer.adam_eps,
    )

    # --- lr schedule ---
    max_steps = config.optimization.max_steps
    warmup_ratio = config.optimization.warmup_ratio
    if max_steps <= 0:
        max_steps = 999_999
    warmup_steps = int(max_steps * warmup_ratio)
    peak_lr = config.optimization.lr_scheduler.peak_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step / max(warmup_steps, 1)) * peak_lr
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- data ---
    dataset = T2STRLDataset(config.dataset, "train", processor=processor)
    collator = GRPOCollator(processor, sampling_rate=config.dataset.sampling_rate)
    sampler = DistributedSampler(
        dataset, num_replicas=train_world_size, rank=train_rank, shuffle=True,
        seed=base_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    # --- CE/SFT dataloader (independent SFT-format dataset, infinite cycle) ---
    ce_iter = None
    if cfg.ce_coef > 0:
        ce_parquet_dir = list(config.gspo.ce_parquet_dir)
        ce_batch_size = int(config.gspo.get("ce_batch_size", config.dataset.batch_size))
        ce_num_workers = int(config.gspo.get("ce_num_workers", 2))
        ce_dataset_config = OmegaConf.create({
            "train_parquet_dir": ce_parquet_dir,
            "dev_parquet_dir": [],
        })
        ce_dataset = T2STDataset(ce_dataset_config, "train", processor=processor)
        ce_collator = CECollator(
            processor, codec_offset, codec_size,
            sampling_rate=config.dataset.sampling_rate,
        )
        ce_sampler = DistributedSampler(
            ce_dataset, num_replicas=train_world_size, rank=train_rank, shuffle=True,
            seed=base_seed,
        )
        ce_loader = DataLoader(
            ce_dataset,
            batch_size=ce_batch_size,
            sampler=ce_sampler,
            collate_fn=ce_collator,
            num_workers=ce_num_workers,
            pin_memory=True,
        )
        ce_iter = _cycle_ce_loader(ce_loader, ce_sampler)
        if is_chief:
            print(
                f"CE/SFT mixing enabled: ce_coef={cfg.ce_coef} "
                f"parquet={ce_parquet_dir} batch_size={ce_batch_size} "
                f"len(ce_dataset)={len(ce_dataset)}"
            )

    # --- held-out eval dataloader (dataset.dev_parquet_dir) ---
    eval_loader = None
    eval_cfg = None
    dev_parquet_dir = list(config.dataset.get("dev_parquet_dir", []) or [])
    if cfg.eval_every_n_steps > 0 and dev_parquet_dir:
        eval_dataset = T2STRLDataset(config.dataset, "dev", processor=processor)
        eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=train_world_size, rank=train_rank,
            shuffle=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.dataset.batch_size,
            sampler=eval_sampler,
            collate_fn=collator,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
        )
        # Eval reuses the training reward server but with its own sampling knobs;
        # extreme_filter is disabled inside evaluate() so means cover all samples.
        eval_cfg = replace(
            cfg,
            num_samples=cfg.eval_num_samples,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            top_k=cfg.eval_top_k,
            extreme_filter=False,
            drop_zero_signal_groups=False,
        )
        if is_chief:
            print(
                f"Eval enabled: dev={dev_parquet_dir} "
                f"every {cfg.eval_every_n_steps} steps "
                f"(num_batches={cfg.eval_num_batches or 'full'}, "
                f"len(eval_dataset)={len(eval_dataset)}, "
                f"num_samples={cfg.eval_num_samples}, temp={cfg.eval_temperature}, "
                f"top_p={cfg.eval_top_p}, top_k={cfg.eval_top_k}, "
                f"seed={cfg.eval_seed})"
            )
    elif is_chief and cfg.eval_every_n_steps > 0:
        print("Eval requested (eval_every_n_steps>0) but dev_parquet_dir is empty; skipping eval.")

    # --- bookkeeping ---
    log_steps = getattr(config.optimization, "log_steps", 10)
    out_dir = f"{config.common.output_dir}/checkpoints"
    if is_chief:
        os.makedirs(out_dir, exist_ok=True)

    # Save ckpt_last.pt and exit when the runtime budget expires.
    # Default 230 min = 3h50m, leaving ~10min for save+sync on a 4h job.
    # Set to 0 to disable the time-based stop.
    max_runtime_minutes = config.optimization.get("max_runtime_minutes", 230)
    max_runtime_seconds = max_runtime_minutes * 60
    proc_start_time = time.time()

    # --- resume from latest RL checkpoint, if any ---
    global_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    dist.barrier(group=train_group)
    latest_ckpt = find_latest_train_ckpt(out_dir)
    if latest_ckpt is not None:
        if is_chief:
            print(f"[resume] loading {latest_ckpt}")
        start_epoch, global_step, resume_step_in_epoch = load_train_state(
            latest_ckpt, policy_model, optimizer, scheduler, device,
        )
        if is_chief:
            print(
                f"[resume] epoch={start_epoch} step={global_step} "
                f"step_in_epoch={resume_step_in_epoch}"
            )
        # Restore adaptive-KL controller state (all train ranks read the
        # sidecar independently — file is small, no broadcast needed).
        if kl_controller is not None:
            if load_kl_state(out_dir, kl_controller):
                cfg.kl_coef = kl_controller.value
                if is_chief:
                    print(f"[resume] kl_coef restored → {cfg.kl_coef:.6g}")
            elif is_chief:
                print(
                    f"[resume] no {KL_STATE_FNAME} found; "
                    f"keeping init kl_coef={cfg.kl_coef:.6g}"
                )
    dist.barrier(group=train_group)
    time_limit_hit = False
    train_state_dtype_audited = False
    require_fp32_train_state = str(
        config.model.get("train_param_dtype", "bfloat16")
    ).strip().lower() in ("fp32", "float32")

    # --- baseline eval at step 0 (before any optimizer step) ---
    # Only on a fresh start (global_step == 0); skipped on resume so we don't
    # re-eval a partially-trained checkpoint at its resumed step.
    if eval_loader is not None and global_step == 0:
        dist.barrier(group=train_group)
        eval_metrics = evaluate(
            policy_model, eval_loader, eval_cfg,
            codec_offset, codec_size, tok, device,
            reward_server, train_group, train_world_size,
            max_batches=cfg.eval_num_batches,
        )
        if is_chief:
            print(
                f"[eval step {global_step}] "
                f"reward_mean={eval_metrics['reward_mean']:.4f} "
                f"followed_silence={eval_metrics.get('pause_followed_silence_mean', float('nan')):.4f} "
                f"added_silence={eval_metrics.get('pause_added_silence_mean', float('nan')):.4f} "
                f"dur10={eval_metrics.get('duration_within_10_frac_mean', eval_metrics.get('duration_within_10_frac', float('nan'))):.4f} "
                f"dur5={eval_metrics.get('duration_within_5_frac_mean', eval_metrics.get('duration_within_5_frac', float('nan'))):.4f} "
                f"dur2={eval_metrics.get('duration_within_2_frac_mean', eval_metrics.get('duration_within_2_frac', float('nan'))):.4f} "
                f"(n_samples={eval_metrics['n_samples']:.0f}, "
                f"n_batches={eval_metrics['n_batches']:.0f})"
            )
            if use_wandb:
                wandb.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=global_step,
                )

    pbar = tqdm(
        total=max_steps, initial=global_step,
        desc="Training", dynamic_ncols=True,
        disable=not is_chief,
    )

    epoch = start_epoch
    for epoch in range(start_epoch, config.optimization.max_epoch):
        sampler.set_epoch(epoch)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        optimizer.zero_grad()
        done = False

        loader_iter = iter(loader)
        step_in_epoch = resume_step_in_epoch if epoch == start_epoch else 0
        if step_in_epoch > 0:
            skip = step_in_epoch * cfg.num_gen_batches
            if is_chief:
                print(f"[resume] skipping {skip} batches in epoch {epoch}")
            for _ in range(skip):
                try:
                    next(loader_iter)
                except StopIteration:
                    done = True
                    break

        while not done:
            # A completed checkpoint may be submitted once more by an external
            # monitor. Exit without taking an accidental step 301.
            if global_step >= max_steps:
                done = True
                break
            phase_t: Dict[str, float] = {
                "gen": 0.0, "reward_wait": 0.0, "fwd_bwd": 0.0, "optim": 0.0,
            }
            step_t0 = time.perf_counter()

            # --- generate (reward_wait is timed separately inside) ---
            gen_buffer = []
            policy_ddp.eval()
            with _PhaseTimer(device, phase_t, "gen"):
                for generation_batch_idx in range(cfg.num_gen_batches):
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        done = True
                        break
                    generation_seed = (
                        base_seed
                        + rank * 10_000_000
                        + global_step * cfg.num_gen_batches
                        + generation_batch_idx
                    ) % (2**32)
                    torch.manual_seed(generation_seed)
                    torch.cuda.manual_seed_all(generation_seed)
                    gen_data = generate_step(
                        policy_model, batch, cfg,
                        codec_offset, codec_size, tok, device,
                        reward_server=reward_server,
                        timings=phase_t,
                    )
                    if gen_data is not None:
                        # Attach CE only on every Nth surviving micro-batch
                        # (using gen_buffer's pre-append length keeps the
                        # pattern aligned across ranks: index 0, N, 2N, ...).
                        attach_ce = (
                            ce_iter is not None
                            and len(gen_buffer) % cfg.ce_every_n_microbatches == 0
                        )
                        if attach_ce:
                            ce_raw = next(ce_iter)
                            gen_data["ce_batch"] = {
                                "input_ids": ce_raw["input_ids"].to(device),
                                "attention_mask": ce_raw["attention_mask"].to(device),
                                "labels": ce_raw["labels"].to(device),
                                "input_features": ce_raw["input_features"].to(
                                    device=device, dtype=torch.bfloat16,
                                ),
                                "feature_attention_mask": ce_raw["feature_attention_mask"].to(device),
                            }
                        gen_buffer.append(gen_data)

            # synchronise buffer length across training ranks
            local_buf = torch.tensor(len(gen_buffer), device=device)
            dist.all_reduce(local_buf, op=dist.ReduceOp.MIN, group=train_group)
            gen_buffer = gen_buffer[: local_buf.item()]

            if not gen_buffer:
                continue

            # --- train: accumulate micro-batches → one optimizer step ---
            n_micro = len(gen_buffer)
            policy_ddp.train()
            accumulated_metrics: Dict[str, float] = {}
            accumulated_loss = 0.0
            ce_metric_keys = ("ce_loss", "ce_n_seq")
            ce_sums: Dict[str, float] = {k: 0.0 for k in ce_metric_keys}
            ce_fires = 0
            with _PhaseTimer(device, phase_t, "fwd_bwd"):
                for i, gen_data in enumerate(gen_buffer):
                    is_sync_step = (i + 1) == n_micro
                    retained_samples = int(gen_data["rewards"].shape[0])
                    use_candidate_chunks = (
                        cfg.train_micro_batch_size > 0
                        and cfg.train_micro_batch_size < retained_samples
                    )
                    if use_candidate_chunks:
                        loss_value, step_metrics = gspo_train_step_microbatched(
                            policy_ddp,
                            ref_model,
                            gen_data,
                            cfg,
                            device,
                            loss_scale=1.0 / n_micro,
                            sync_gradients=is_sync_step,
                        )
                    else:
                        ctx = nullcontext if is_sync_step else policy_ddp.no_sync
                        with ctx():
                            loss, step_metrics = gspo_train_step(
                                policy_ddp, ref_model, gen_data, cfg, device,
                            )
                            (loss / n_micro).backward()
                        loss_value = loss.item()
                    accumulated_loss += loss_value / n_micro
                    if "ce_loss" in step_metrics:
                        for k in ce_metric_keys:
                            ce_sums[k] += step_metrics.pop(k, 0.0)
                        ce_fires += 1
                    for k, v in step_metrics.items():
                        accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / n_micro
            if ce_fires > 0:
                for k in ce_metric_keys:
                    accumulated_metrics[k] = ce_sums[k] / ce_fires
                accumulated_metrics["ce_n_fires"] = ce_fires

            with _PhaseTimer(device, phase_t, "optim"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_ddp.parameters() if p.requires_grad],
                    max_norm=config.optimization.get("max_grad_norm", 1.0),
                )
                optimizer.step()
                scheduler.step()
                if not train_state_dtype_audited:
                    dtype_summary = _audit_train_state_dtypes(
                        policy_model, optimizer,
                        require_fp32=require_fp32_train_state,
                    )
                    if is_chief:
                        print(f"[train-state-audit] {dtype_summary}")
                    train_state_dtype_audited = True
                optimizer.zero_grad()
            global_step += 1
            step_in_epoch += 1
            phase_t["step"] = time.perf_counter() - step_t0

            cur_lr = scheduler.get_last_lr()[0]
            avg_reward = accumulated_metrics["reward_mean"]
            accumulated_metrics["grad_norm"] = grad_norm.item()

            # Adaptive-KL update: average observed KL across training ranks
            # (so all ranks compute identical kl_coef), drive the controller
            # on the chief, then broadcast the new value back to all ranks.
            if kl_controller is not None:
                local_kl = torch.tensor(
                    accumulated_metrics["kl_mean"], device=device, dtype=torch.float32,
                )
                dist.all_reduce(local_kl, op=dist.ReduceOp.SUM, group=train_group)
                avg_kl = (local_kl / train_world_size).item()
                kl_controller.update(avg_kl, n_steps=1)
                cfg.kl_coef = kl_controller.value
                accumulated_metrics["kl_coef"] = cfg.kl_coef
                accumulated_metrics["kl_avg_train"] = avg_kl
            else:
                accumulated_metrics["kl_coef"] = cfg.kl_coef

            pbar.update(1)
            pbar.set_postfix(step=f"{global_step}", reward=f"{avg_reward:.2f}")

            # Per-phase timing: reduce MAX across train ranks (slowest gates step
            # time) every log step. Keep this off the hot path otherwise.
            timing_keys = ["gen", "reward_wait", "fwd_bwd", "optim", "step"]
            timing_max: Dict[str, float] = {}
            if global_step % log_steps == 0:
                t_tensor = torch.tensor(
                    [phase_t[k] for k in timing_keys],
                    device=device, dtype=torch.float32,
                )
                dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX, group=train_group)
                timing_max = dict(zip(timing_keys, t_tensor.cpu().tolist()))
                timing_max["gen_pure"] = max(
                    timing_max["gen"] - timing_max["reward_wait"], 0.0
                )
                if is_chief:
                    print(
                        f"[step {global_step}] step={timing_max['step']:.2f}s "
                        f"gen={timing_max['gen']:.2f}s "
                        f"(gen_pure={timing_max['gen_pure']:.2f}s, "
                        f"reward_wait={timing_max['reward_wait']:.2f}s) "
                        f"fwd_bwd={timing_max['fwd_bwd']:.2f}s "
                        f"optim={timing_max['optim']:.2f}s"
                    )

            if use_wandb and global_step % log_steps == 0:
                log_payload = {
                    "train/loss": accumulated_loss,
                    "train/policy_loss": accumulated_metrics["policy_loss"],
                    "train/kl_mean": accumulated_metrics["kl_mean"],
                    "train/kl_loss": accumulated_metrics["kl_loss"],
                    "train/kl_coef": accumulated_metrics["kl_coef"],
                    "train/lr": cur_lr,
                    "train/reward_mean": accumulated_metrics["reward_mean"],
                    "train/reward_std": accumulated_metrics["reward_std"],
                    "train/advantage_mean": accumulated_metrics["advantage_mean"],
                    "train/advantage_std": accumulated_metrics["advantage_std"],
                    "train/adv_std_floor": accumulated_metrics["adv_std_floor"],
                    "train/adv_clip": accumulated_metrics["adv_clip"],
                    "train/ratio_mean": accumulated_metrics["ratio_mean"],
                    "train/clip_fraction": accumulated_metrics["clip_fraction"],
                    "train/n_valid": accumulated_metrics["n_valid"],
                    "train/grad_norm": accumulated_metrics["grad_norm"],
                }
                for k, v in timing_max.items():
                    log_payload[f"time/{k}_s"] = v
                if kl_controller is not None:
                    log_payload["train/kl_avg_train"] = accumulated_metrics["kl_avg_train"]
                if cfg.ce_coef > 0:
                    log_payload["train/ce_loss"] = accumulated_metrics.get("ce_loss", 0.0)
                    log_payload["train/ce_coef"] = cfg.ce_coef
                    log_payload["train/ce_n_seq"] = accumulated_metrics.get("ce_n_seq", 0.0)
                    log_payload["train/ce_n_fires"] = accumulated_metrics.get("ce_n_fires", 0.0)
                for k, v in accumulated_metrics.items():
                    if k.startswith(("pause_", "quality_", "duration_")) or k in (
                        "num_src_pauses_mean",
                        "num_gen_pauses_mean",
                        "num_dense_gen_pauses_mean",
                        "num_samples",
                        "num_groups",
                    ):
                        log_payload[f"train/{k}"] = v
                wandb.log(log_payload, step=global_step)

            # --- held-out eval on the dev set ---
            if eval_loader is not None and global_step % cfg.eval_every_n_steps == 0:
                dist.barrier(group=train_group)
                eval_metrics = evaluate(
                    policy_model, eval_loader, eval_cfg,
                    codec_offset, codec_size, tok, device,
                    reward_server, train_group, train_world_size,
                    max_batches=cfg.eval_num_batches,
                )
                if is_chief:
                    print(
                        f"[eval step {global_step}] "
                        f"reward_mean={eval_metrics['reward_mean']:.4f} "
                        f"followed_silence={eval_metrics.get('pause_followed_silence_mean', float('nan')):.4f} "
                        f"added_silence={eval_metrics.get('pause_added_silence_mean', float('nan')):.4f} "
                        f"dur10={eval_metrics.get('duration_within_10_frac_mean', eval_metrics.get('duration_within_10_frac', float('nan'))):.4f} "
                        f"dur5={eval_metrics.get('duration_within_5_frac_mean', eval_metrics.get('duration_within_5_frac', float('nan'))):.4f} "
                        f"dur2={eval_metrics.get('duration_within_2_frac_mean', eval_metrics.get('duration_within_2_frac', float('nan'))):.4f} "
                        f"(n_samples={eval_metrics['n_samples']:.0f}, "
                        f"n_batches={eval_metrics['n_batches']:.0f})"
                    )
                    if use_wandb:
                        wandb.log(
                            {f"eval/{k}": v for k, v in eval_metrics.items()},
                            step=global_step,
                        )

            if global_step % config.checkpoint.save_every_n_steps == 0:
                dist.barrier(group=train_group)
                if is_chief:
                    path = f"{out_dir}/ckpt_step{global_step}.pt"
                    save_train_state(
                        path, policy_model, optimizer, scheduler,
                        epoch, global_step, step_in_epoch,
                    )
                    if kl_controller is not None:
                        save_kl_state(out_dir, kl_controller)
                    print(f"[step {global_step}] checkpoint saved → {path}")

            # Save ckpt_last.pt and exit when the runtime budget expires.
            if max_runtime_seconds > 0:
                elapsed = time.time() - proc_start_time
                stop_flag = torch.tensor(
                    1 if elapsed > max_runtime_seconds else 0,
                    device=device, dtype=torch.int32,
                )
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX, group=train_group)
                if stop_flag.item() > 0:
                    dist.barrier(group=train_group)
                    if is_chief:
                        path = f"{out_dir}/{LAST_CKPT_NAME}"
                        save_train_state(
                            path, policy_model, optimizer, scheduler,
                            epoch, global_step, step_in_epoch,
                            also_write_last=False,
                        )
                        if kl_controller is not None:
                            save_kl_state(out_dir, kl_controller)
                        print(
                            f"[time-limit {elapsed/60:.1f}min] saved {path} "
                            f"at step {global_step}; exiting for resubmit"
                        )
                    time_limit_hit = True
                    done = True
                    break

            if global_step >= max_steps:
                done = True
                break

        if time_limit_hit or global_step >= max_steps:
            break

        dist.barrier(group=train_group)
        if is_chief:
            path = f"{out_dir}/ckpt_e{epoch}.pt"
            save_train_state(
                path, policy_model, optimizer, scheduler,
                epoch, global_step, 0,
            )
            if kl_controller is not None:
                save_kl_state(out_dir, kl_controller)
            print(f"[epoch {epoch}] checkpoint saved → {path}")

    pbar.close()
    dist.barrier(group=train_group)
    if is_chief and not time_limit_hit:
        path = f"{out_dir}/ckpt_final.pt"
        save_train_state(
            path, policy_model, optimizer, scheduler,
            epoch, global_step, 0,
        )
        if kl_controller is not None:
            save_kl_state(out_dir, kl_controller)
        print(f"Training complete. Final checkpoint → {path}")

    if use_wandb:
        wandb.finish()

    reward_server.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
