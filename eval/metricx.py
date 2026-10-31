import os
import sys

import torch
import transformers
from tqdm import tqdm

# MetricX-24-Hybrid-XXL weights (HF: google/metricx-24-hybrid-xxl-v2p6-bfloat16)
# and a checkout of https://github.com/google-research/metricx (provides
# metricx24.models.MT5ForRegression).  By default the repo is expected at
# <metricx_root>/metricx.
METRICX_DIR = os.environ.get("metricx_root")
if not METRICX_DIR:
    sys.exit("metricx.py: set metricx_root=<dir with the MetricX-24 weights>")
METRICX_REPO = os.environ.get("metricx_repo", f"{METRICX_DIR}/metricx")
sys.path.insert(0, METRICX_REPO)
from metricx24 import models  # noqa: E402

TOKENIZER = "google/mt5-xl"
MAX_INPUT_LENGTH = 1536
BATCH_SIZE = 4

# Manifest with references: name \t source_text \t reference_text
gt_path = os.environ.get("GT_PATH")
if not gt_path:
    sys.exit("metricx.py: set GT_PATH=<manifest.tsv> (name\tsource\treference)")
generated_path = sys.argv[1]

gt_data = {}
with open(gt_path) as f1:
    for line in f1:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or not parts[2].strip():
            continue  # no reference available for this row
        fname, src, ref = parts[0], parts[1], parts[2]
        gt_data[fname] = {"src": src, "ref": ref}

data = []
with open(generated_path) as f2:
    for line in f2:
        parts = line.rstrip("\n").split("\t")
        fname = parts[0]
        hyp = parts[1] if len(parts) > 1 else ""
        if fname not in gt_data: continue
        data.append({"name": fname, "src": gt_data[fname]["src"], "ref": gt_data[fname]["ref"], "mt": hyp})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
model = models.MT5ForRegression.from_pretrained(METRICX_DIR, torch_dtype="auto").to(device)
model.eval()


def make_input(ex, qe: bool):
    if qe:
        return f"source: {ex['src']} candidate: {ex['mt']}"
    return f"source: {ex['src']} candidate: {ex['mt']} reference: {ex['ref']}"


def score(qe: bool):
    preds = []
    for start in tqdm(range(0, len(data), BATCH_SIZE), desc="QE" if qe else "Ref"):
        batch = [make_input(ex, qe) for ex in data[start : start + BATCH_SIZE]]
        enc = tokenizer(
            batch,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][:, :-1].to(device)
        attention_mask = enc["attention_mask"][:, :-1].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        preds.extend(out.predictions.float().cpu().tolist())
    return preds


ref_scores = score(qe=False)
qe_scores = score(qe=True)

output_file = os.path.join(os.path.dirname(generated_path), "metricx.txt")
with open(output_file, "w") as wf:
    print(f"MetricX-24 reference-based (lower is better): {sum(ref_scores) / len(ref_scores):.4f}", file=wf)
    print(f"MetricX-24 QE (lower is better):              {sum(qe_scores) / len(qe_scores):.4f}", file=wf)
per_sample_file = os.path.join(os.path.dirname(generated_path), "metricx_per_sample.tsv")
with open(per_sample_file, "w") as wf:
    for ex, r, q in zip(data, ref_scores, qe_scores):
        print(f"{ex['name']}\t{r:.4f}\t{q:.4f}", file=wf)
# print(f"MetricX-24 reference-based (lower is better): {sum(ref_scores) / len(ref_scores):.4f}")
# print(f"MetricX-24 QE (lower is better):              {sum(qe_scores) / len(qe_scores):.4f}")