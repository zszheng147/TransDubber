#!/usr/bin/env python
"""COMET (wmt22-comet-da) scoring for TransDubber test system outputs.

For each system dir (containing translated_texts.txt: name\\tmt), builds
(src, mt, ref) triplets from the manifest, scores with COMET, writes
comet_per_sample.tsv + comet.txt into the dir, prints all/cv/gs means.

Usage:
  python eval/comet_score.py --manifest .../manifest.tsv --model-dir <ckpt.pt or auto> \\
      <dir1> <dir2> ...
Rows with empty reference or empty hypothesis are skipped (matches the
MetricX terminated-only convention).
"""
import argparse
import os

from comet import load_from_checkpoint


def load_manifest(path):
    d = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[1].strip() and parts[2].strip():
            d[parts[0]] = (parts[1], parts[2])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model-ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"manifest rows with src+ref: {len(manifest)}", flush=True)
    model = load_from_checkpoint(args.model_ckpt)

    for d in args.dirs:
        tt = os.path.join(d, "translated_texts.txt")
        if not os.path.exists(tt):
            print(f"[skip] no translated_texts.txt in {d}", flush=True)
            continue
        names, data = [], []
        for line in open(tt):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            name = parts[0]
            if name not in manifest:
                continue
            src, ref = manifest[name]
            names.append(name)
            data.append({"src": src, "mt": parts[1], "ref": ref})
        out = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
        scores = out["scores"]
        with open(os.path.join(d, "comet_per_sample.tsv"), "w") as f:
            for n, s in zip(names, scores):
                f.write(f"{n}\t{s:.6f}\n")

        def mean(pred):
            sel = [s for n, s in zip(names, scores) if pred(n)]
            return (sum(sel) / len(sel), len(sel)) if sel else (float("nan"), 0)

        a, na = mean(lambda n: True)
        c, nc = mean(lambda n: n.startswith("common_voice"))
        g, ng = mean(lambda n: not n.startswith("common_voice"))
        line = (f"COMET wmt22-comet-da (higher better): all {a:.4f} (n={na}) | "
                f"cv {c:.4f} (n={nc}) | gs {g:.4f} (n={ng})")
        with open(os.path.join(d, "comet.txt"), "w") as f:
            f.write(line + "\n")
        print(f"{d}\n  {line}", flush=True)


if __name__ == "__main__":
    main()
