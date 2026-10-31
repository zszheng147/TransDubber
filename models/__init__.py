from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


# ASR-only callers do not need the optional CosyVoice/librosa/ONNX token2wav
# stack.  Import it only when one of its public helpers is requested; this also
# keeps ``from models import Qwen3ASRForConditionalGeneration`` usable in
# lightweight/offline data-preparation environments.
_TOKEN2WAV_EXPORTS = {"build_model", "prepare_prompt_conditioning", "token2wav"}


def __getattr__(name):
    if name in _TOKEN2WAV_EXPORTS:
        from .token2wav.helper import build_model, prepare_prompt_conditioning, token2wav

        exports = {
            "build_model": build_model,
            "prepare_prompt_conditioning": prepare_prompt_conditioning,
            "token2wav": token2wav,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRProcessor",
    "build_model",
    "prepare_prompt_conditioning",
    "token2wav",
]
