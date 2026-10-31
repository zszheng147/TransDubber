#!/usr/bin/env bash
# Translate English speech file(s) into isometric, pause-aligned Chinese speech.
#
#   bash inference/infer.sh <checkpoint> <out_dir> <audio> [more audio ...]
#
# Required environment:
#   base_model      Qwen3-ASR-1.7B directory
#   token2wav_root  Fun-CosyVoice3-0.5B directory (flow.pt, hift.pt, campplus.onnx, speech_tokenizer_v3.onnx)
# Optional: PY (python interpreter), DEVICE (cuda:0), TOP_P / TEMPERATURE / TOP_K / SEED
#           (decoding protocol, defaults 0.85 / 0.85 / 0 / 42), attn_impl (sdpa if no flash-attn)
#
# Writes <out_dir>/audio/<name>.wav, <out_dir>/predictions.jsonl, <out_dir>/translated_texts.txt.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: bash inference/infer.sh <checkpoint> <out_dir> <audio> [more audio ...]" >&2
    exit 2
fi
CKPT=$1
OUT=$2
shift 2

: "${base_model:?set base_model=/path/to/Qwen3-ASR-1.7B}"
: "${token2wav_root:?set token2wav_root=/path/to/Fun-CosyVoice3-0.5B}"

REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

exec "${PY:-python}" "$REPO/inference/infer_single.py" \
    --ckpt_path "$CKPT" --out_dir "$OUT" --device "${DEVICE:-cuda:0}" \
    --top_p "${TOP_P:-0.85}" --temperature "${TEMPERATURE:-0.85}" --top_k "${TOP_K:-0}" \
    --seed "${SEED:-42}" --audio "$@"
