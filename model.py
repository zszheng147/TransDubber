import gc
import math

import torch
import torch.nn.functional as F
from torch import optim
import lightning as L

from models import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
from transformers import GenerationConfig


def resolve_train_param_dtype(raw_dtype, compute_dtype):
    """Resolve the stored trainable-parameter dtype independently of autocast."""
    name = "auto" if raw_dtype is None else str(raw_dtype).strip().lower()
    if name in {"", "auto"}:
        return compute_dtype
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "torch.float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "torch.float16": torch.float16,
    }
    if name not in aliases:
        raise ValueError(
            "model.train_param_dtype must be auto, float32, bfloat16, or "
            f"float16; got {raw_dtype!r}"
        )
    return aliases[name]


def resolve_lr_schedule_steps(
    actual_total_steps,
    max_epochs,
    warmup_ratio,
    decay_horizon_epochs=None,
    warmup_steps_override=-1,
):
    """Return (decay-total steps, warmup steps) without coupling warmup to horizon."""
    actual_total_steps = int(actual_total_steps)
    max_epochs = int(max_epochs)
    warmup_ratio = float(warmup_ratio)
    if actual_total_steps <= 0:
        raise ValueError("actual_total_steps must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")

    if decay_horizon_epochs is None:
        horizon = float(max_epochs)
    else:
        horizon = float(decay_horizon_epochs)
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("decay_horizon_epochs must be positive")
    decay_total_steps = max(
        1, int(round(actual_total_steps * horizon / float(max_epochs)))
    )

    if warmup_steps_override is None or int(warmup_steps_override) < 0:
        warmup_steps = int(actual_total_steps * warmup_ratio)
    else:
        warmup_steps = int(warmup_steps_override)
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps >= decay_total_steps:
        raise ValueError(
            "warmup_steps must be smaller than the decay schedule length: "
            f"{warmup_steps} vs {decay_total_steps}"
        )
    return decay_total_steps, warmup_steps


def patch_outer_forward(model):
    """Redirect model.forward() to model.thinker.forward() for HF-Trainer-free usage."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(
        self, input_ids=None, attention_mask=None,
        input_features=None, feature_attention_mask=None,
        labels=None, **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ------------------------ Define the LightningModule ------------------------ #
class LitModelModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        self.model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.train_param_dtype = resolve_train_param_dtype(
            getattr(config.model, "train_param_dtype", "auto"),
            self.model_dtype,
        )

        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            config.model.model_path, dtype=self.train_param_dtype, device_map=None,
        )
        self.codec_offset = getattr(config.model, "speech_codec_offset", 151936)
        codec_size = getattr(config.model, "speech_codec_size", 6561)
        self.model.thinker.resize_token_embeddings(self.codec_offset + codec_size)
        
        self.processor = Qwen3ASRProcessor.from_pretrained(
            config.model.model_path, fix_mistral_regex=True
        )

        special_tokens = [t for t in list(getattr(config.model, "special_tokens", [])) if t]
        if special_tokens: # num_added < 200 is ok
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )

        self.text_loss_weight = float(getattr(config.model, "text_loss_weight", 0.4))
        self.sequence_normalized_loss = bool(
            getattr(config.model, "sequence_normalized_loss", False)
        )
        self.pause_token = getattr(config.dataset, "pause_token", None) or None
        self.pause_token_loss_weight = float(
            getattr(config.model, "pause_token_loss_weight", 1.0)
        )
        self.pause_speech_token_loss_weight = float(
            getattr(config.model, "pause_speech_token_loss_weight", 1.0)
        )
        if self.pause_token_loss_weight <= 0.0:
            raise ValueError("pause_token_loss_weight must be > 0")
        if self.pause_speech_token_loss_weight <= 0.0:
            raise ValueError("pause_speech_token_loss_weight must be > 0")
        self.pause_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids(self.pause_token)
            if self.pause_token
            else None
        )
        # Weight EOS supervision explicitly to control termination behavior.
        # The default of 1.0 leaves the token loss unchanged.
        self.eos_token_loss_weight = float(
            getattr(config.model, "eos_token_loss_weight", 1.0)
        )
        if self.eos_token_loss_weight <= 0.0:
            raise ValueError("eos_token_loss_weight must be > 0")
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.speech_lr_mult = float(getattr(config.optimization, "speech_lr_mult", 1.0))

        patch_outer_forward(self.model)
        self.model.generation_config = GenerationConfig.from_model_config(self.model.config)

    def _compute_decomposed_loss(
        self, outputs, labels, sample_loss_weights=None, pause_speech_mask=None,
    ):
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)

        valid_mask = shifted_labels != -100
        speech_mask = valid_mask & (shifted_labels >= self.codec_offset)
        text_mask = valid_mask & ~speech_mask

        token_weights = torch.ones_like(per_token_loss)
        token_weights[text_mask] = self.text_loss_weight
        pause_mask = torch.zeros_like(valid_mask)
        if self.pause_token_id is not None:
            pause_mask = valid_mask & (shifted_labels == self.pause_token_id)
            token_weights[pause_mask] *= self.pause_token_loss_weight
        if self.eos_token_loss_weight != 1.0 and self.eos_token_id is not None:
            eos_mask = valid_mask & (shifted_labels == self.eos_token_id)
            token_weights[eos_mask] *= self.eos_token_loss_weight

        shifted_pause_speech_mask = torch.zeros_like(valid_mask)
        if pause_speech_mask is not None:
            if pause_speech_mask.shape != labels.shape:
                raise ValueError(
                    "pause_speech_mask must have the same shape as labels, got "
                    f"{tuple(pause_speech_mask.shape)} vs {tuple(labels.shape)}"
                )
            shifted_pause_speech_mask = pause_speech_mask[:, 1:].to(
                device=labels.device, dtype=torch.bool,
            )
            invalid_pause_speech = shifted_pause_speech_mask & ~speech_mask
            if invalid_pause_speech.any():
                raise ValueError(
                    "pause_speech_mask selected a padding, text, or structural token"
                )
            # Skip multiplication when the weight is neutral.
            if self.pause_speech_token_loss_weight != 1.0:
                token_weights[shifted_pause_speech_mask] *= (
                    self.pause_speech_token_loss_weight
                )

        if sample_loss_weights is None:
            sample_loss_weights = torch.ones(
                labels.shape[0], device=labels.device, dtype=per_token_loss.dtype,
            )
        else:
            sample_loss_weights = sample_loss_weights.to(
                device=labels.device, dtype=per_token_loss.dtype,
            )

        if self.sequence_normalized_loss:
            # Full S2ST rows contain hundreds of codec labels whereas ASR-only
            # rows contain only text.  Normalising each sequence first prevents
            # the real-pause detector supervision from being drowned out merely
            # because its valid sequence is shorter.
            weighted_valid = token_weights * valid_mask
            seq_den = weighted_valid.sum(dim=1)
            seq_loss = (per_token_loss * weighted_valid).sum(dim=1) / seq_den.clamp(min=1.0)
            has_labels = seq_den > 0
            effective_sample_weights = sample_loss_weights * has_labels
            total_loss = (
                seq_loss * effective_sample_weights
            ).sum() / effective_sample_weights.sum().clamp(min=1.0)
        else:
            weighted_valid = (
                token_weights
                * valid_mask
                * sample_loss_weights.unsqueeze(1)
            )
            total_loss = (
                per_token_loss * weighted_valid
            ).sum() / weighted_valid.sum().clamp(min=1.0)
        text_loss = per_token_loss[text_mask].mean() if text_mask.any() else None
        speech_loss = per_token_loss[speech_mask].mean() if speech_mask.any() else None
        pause_loss = per_token_loss[pause_mask].mean() if pause_mask.any() else None
        pause_speech_loss = (
            per_token_loss[shifted_pause_speech_mask].mean()
            if shifted_pause_speech_mask.any() else None
        )
        return total_loss, text_loss, speech_loss, pause_loss, pause_speech_loss

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_features=batch.get("input_features"),
            feature_attention_mask=batch.get("feature_attention_mask"),
            labels=batch["labels"],
        )
        total_loss, text_loss, speech_loss, pause_loss, pause_speech_loss = self._compute_decomposed_loss(
            outputs, batch["labels"], batch.get("sample_loss_weights"),
            batch.get("pause_speech_mask"),
        )

        self.log("train_loss", total_loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
        # Component losses are data-dependent: an ASR-only batch has no speech
        # labels, and a no-pause batch has no pause labels.  Different DDP
        # ranks can therefore take different branches here.  Never use a
        # distributed log reduction inside these conditional branches; ranks
        # that skip the branch would not enter the matching collective and the
        # whole job would deadlock.  The rank-zero/local values are sufficient
        # for these diagnostic metrics; the unconditional total loss above is
        # still reduced across all ranks.
        if text_loss is not None:
            self.log("train_text_loss", text_loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=False)
        if speech_loss is not None:
            self.log("train_speech_loss", speech_loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=False)
        if pause_loss is not None:
            self.log("train_pause_loss", pause_loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=False)
        if pause_speech_loss is not None:
            self.log("train_pause_speech_loss", pause_speech_loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_features=batch.get("input_features"),
            feature_attention_mask=batch.get("feature_attention_mask"),
            labels=batch["labels"],
        )
        total_loss, text_loss, speech_loss, pause_loss, pause_speech_loss = self._compute_decomposed_loss(
            outputs, batch["labels"], batch.get("sample_loss_weights"),
            batch.get("pause_speech_mask"),
        )

        self.log("val_loss", total_loss.item(), on_step=False, prog_bar=True, logger=True, sync_dist=True)
        if text_loss is not None:
            self.log("val_text_loss", text_loss.item(), on_step=False, logger=True, sync_dist=False)
        if speech_loss is not None:
            self.log("val_speech_loss", speech_loss.item(), on_step=False, logger=True, sync_dist=False)
        if pause_loss is not None:
            self.log("val_pause_loss", pause_loss.item(), on_step=False, logger=True, sync_dist=False)
        if pause_speech_loss is not None:
            self.log("val_pause_speech_loss", pause_speech_loss.item(), on_step=False, logger=True, sync_dist=False)

    def on_train_epoch_start(self):
        if hasattr(self.trainer, "train_dataloader") and self.trainer.train_dataloader is not None:
            sampler = getattr(self.trainer.train_dataloader, "batch_sampler", None)
            if sampler and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.trainer.current_epoch)

    def on_validation_epoch_end(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _iter_speech_weight_tensors(self):
        """Return the list of unique embedding/output-head weight Parameters that
        contain speech codec rows. Handles tied weights by deduplicating on id().
        """
        modules = [
            self.model.thinker.get_input_embeddings(),
            self.model.thinker.get_output_embeddings(),
        ]
        weights, seen = [], set()
        for m in modules:
            w = getattr(m, "weight", None) if m is not None else None
            if w is None or id(w) in seen:
                continue
            seen.add(id(w))
            weights.append(w)
        return weights

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.optimization.lr,
            betas=(
                self.config.optimization.optimizer.adam_betas_0,
                self.config.optimization.optimizer.adam_betas_1,
            ),
            eps=self.config.optimization.optimizer.adam_eps,
            weight_decay=getattr(self.config.optimization.optimizer, "weight_decay", 0.0),
        )

        if self.speech_lr_mult != 1.0:
            speech_weights = self._iter_speech_weight_tensors()
            codec_offset = self.codec_offset
            extra = self.speech_lr_mult - 1.0
            snapshots: list = []

            def _pre_step_hook(opt, args, kwargs):
                snapshots.clear()
                for w in speech_weights:
                    snapshots.append((w, w.data[codec_offset:].detach().clone()))

            def _post_step_hook(opt, args, kwargs):
                with torch.no_grad():
                    for w, before in snapshots:
                        current = w.data[codec_offset:]
                        delta = current - before
                        current.add_(delta, alpha=extra)
                snapshots.clear()

            optimizer.register_step_pre_hook(_pre_step_hook)
            optimizer.register_step_post_hook(_post_step_hook)

        # Check parameter and optimizer-state dtypes after the first Adam step.
        dtype_audited = False

        def _post_step_dtype_audit(opt, args, kwargs):
            nonlocal dtype_audited
            if dtype_audited:
                return
            expected = self.train_param_dtype
            bad_params = [
                name for name, param in self.named_parameters()
                if param.requires_grad and param.dtype != expected
            ]
            bad_grads = [
                name for name, param in self.named_parameters()
                if param.requires_grad and param.grad is not None
                and param.grad.dtype != expected
            ]
            bad_moments = []
            moment_count = 0
            for state in opt.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    value = state.get(key)
                    if value is None:
                        continue
                    moment_count += 1
                    if value.dtype != expected:
                        bad_moments.append((key, str(value.dtype)))
            if bad_params or bad_grads or bad_moments or moment_count == 0:
                raise RuntimeError(
                    "Training-state dtype audit failed: "
                    f"expected={expected}, bad_params={bad_params[:3]}, "
                    f"bad_grads={bad_grads[:3]}, "
                    f"bad_moments={bad_moments[:3]}, moments={moment_count}"
                )
            if self.global_rank == 0:
                print(
                    "[train-state-audit] "
                    f"params/grads/Adam moments={expected}; "
                    f"moment_tensors={moment_count}",
                    flush=True,
                )
            dtype_audited = True

        optimizer.register_step_post_hook(_post_step_dtype_audit)

        actual_total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler_config = self.config.optimization.lr_scheduler
        decay_total_steps, warmup_steps = resolve_lr_schedule_steps(
            actual_total_steps=actual_total_steps,
            max_epochs=self.config.optimization.max_epoch,
            warmup_ratio=self.config.optimization.warmup_ratio,
            decay_horizon_epochs=getattr(
                scheduler_config, "decay_horizon_epochs", None,
            ),
            warmup_steps_override=getattr(
                scheduler_config, "warmup_steps", -1,
            ),
        )
        if self.global_rank == 0:
            print(
                "[lr-schedule] "
                f"actual_train_steps={actual_total_steps}; "
                f"decay_total_steps={decay_total_steps}; "
                f"warmup_steps={warmup_steps}",
                flush=True,
            )

        def lr_lambda(step):
            peak_lr = scheduler_config.peak_lr
            if step < warmup_steps:
                return (step / max(warmup_steps, 1)) * peak_lr
            decay_steps = decay_total_steps - warmup_steps
            if decay_steps <= 0:
                return peak_lr
            return peak_lr * max(0.0, 1 - (step - warmup_steps) / decay_steps)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
