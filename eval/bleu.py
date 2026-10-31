import unicodedata
import sacrebleu

def get_all_unicode_punctuation():
    punctuation_chars = set()
    for codepoint in range(0x110000):  # Unicode goes up to U+10FFFF
        char = chr(codepoint)
        # Check if the Unicode category starts with 'P' (e.g., 'Po', 'Pd', etc.)
        if unicodedata.category(char).startswith('P'):
            punctuation_chars.add(char)
    return ''.join(sorted(punctuation_chars))

punctuation_all = get_all_unicode_punctuation()

import os
import sys

# Manifest with reference translations: name \t source_text \t reference_text
GT_PATH = os.environ.get("GT_PATH")
if not GT_PATH:
    sys.exit("bleu.py: set GT_PATH=<manifest.tsv> (name\tsource\treference)")
d = {}
with open(GT_PATH) as f:
    for line in f.readlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or not parts[2].strip():
            continue  # no reference available for this row
        fname, src, ref = parts[0], parts[1], parts[2]
        ref = ref.translate(str.maketrans('', '', punctuation_all))
        d[fname] = [ref]

bleu_scores = []
with open(sys.argv[1]) as f:
    for line in f.readlines()[:]:
        parts = line.rstrip("\n").split("\t")
        fname = parts[0]
        hyp = parts[1] if len(parts) > 1 else ""
        hyp = hyp.translate(str.maketrans('', '', punctuation_all))
        if fname in d:
            references = [d[fname]]
            candidates = [hyp]
            bleu = sacrebleu.corpus_bleu(candidates, references, tokenize='zh')
            bleu_scores.append((fname, bleu.score, hyp))

print(sum(score for _, score, _ in bleu_scores) / len(bleu_scores))

per_sample_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "bleu_per_sample.tsv")
with open(per_sample_path, "w") as wf:
    for fname, score, _ in bleu_scores:
        print(f"{fname}\t{score:.4f}", file=wf)