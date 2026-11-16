"""Build the 100-question binary-choice set from TruthfulQA (Lin et al., 2022; Apache-2.0).

Each item pairs the dataset's "Best Answer" with its "Best Incorrect Answer".
Option order is randomised with a fixed seed so the correct letter is balanced.
"""
import csv, json, random, sys
from pathlib import Path

SEED, N = 20261116, 100
root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "data" / "TruthfulQA.csv", encoding="utf-8")))
rows = [r for r in rows if r["Best Answer"].strip() and r["Best Incorrect Answer"].strip()
        and r["Best Answer"].strip() != r["Best Incorrect Answer"].strip()]
rng = random.Random(SEED)
sample = rng.sample(rows, N)
out = []
for i, r in enumerate(sample):
    correct_first = rng.random() < 0.5
    a, b = (r["Best Answer"], r["Best Incorrect Answer"]) if correct_first else (r["Best Incorrect Answer"], r["Best Answer"])
    out.append({"qid": i, "category": r["Category"], "question": r["Question"].strip(),
                "A": a.strip(), "B": b.strip(), "correct": "A" if correct_first else "B"})
with open(root / "data" / "questions.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} questions; correct=A in {sum(o['correct']=='A' for o in out)}", file=sys.stderr)
