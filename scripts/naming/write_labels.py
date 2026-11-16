"""Write a labels file (configs/labels/<map>[_fine].tsv) from names in "id<TAB>name" form.

Adds each topic's current top five terms, which tst uses to check that the names belong to this model
(tst.topics.model.read_labels). Every topic id of the model must be named exactly once.

    python scripts/naming/write_labels.py runs/law names.tsv configs/labels/law.tsv [--fine]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("names", type=Path, help="TSV: id<TAB>name (a header line starting with 'id' is skipped)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--fine", action="store_true")
    args = ap.parse_args()
    topics = json.loads((args.run / ("fine" if args.fine else "") / "model_topics.json").read_text())
    names: dict[int, str] = {}
    for line in args.names.read_text().splitlines():
        if not line.strip() or line.lower().startswith("id\t"):
            continue
        i, name = line.split("\t")[:2]
        i, name = int(i), name.strip()
        if i in names:
            raise SystemExit(f"topic {i} is named twice")
        if not name or "\t" in name:
            raise SystemExit(f"topic {i}: empty name")
        names[i] = name
    if sorted(names) != list(range(len(topics))):
        missing = sorted(set(range(len(topics))) - set(names))
        extra = sorted(set(names) - set(range(len(topics))))
        raise SystemExit(f"{len(topics)} topics: missing {missing}, unknown {extra}")
    rows = ["id\tname\tterms"] + [f"{t['id']}\t{names[t['id']]}\t" + ", ".join(x["term"] for x in t["top_terms"][:5])
                                  for t in topics]
    args.out.write_text("\n".join(rows) + "\n")
    print(f"{args.out}: {len(topics)} names")


if __name__ == "__main__":
    main()
