"""Candidate academic stop words from finished field runs (how src/tst/topics/academic_stopwords.yaml was built).

For every word group (variants merged as in the pipeline) and every run: the share of documents containing it
(recovered from the run's idf), its spread over the topics of the main NMF model (normalised entropy of its
loadings, 1 = equal in all topics) and how often it is among a topic's 10 top words. Groups are ranked by
their document share in the second-least frequent field, i.e. words that are frequent everywhere come first.
The ranked table is meant for review by hand; words that are a subject in any field stay off the list.

    python scripts/stopword_candidates.py runs candidates.tsv [--top 1500]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tst.topics.curation import build_variants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="folder with one finished run per field (state.json, vectorizer/, model/)")
    ap.add_argument("out", type=Path, help="output table (TSV)")
    ap.add_argument("--top", type=int, default=1500)
    a = ap.parse_args()

    fields = sorted(p.parent.name for p in a.runs.glob("*/state.json"))
    df_by_field, n_docs, topic_stats = {}, {}, {}
    total: collections.Counter = collections.Counter()
    for f in fields:
        run = a.runs / f
        n = json.loads((run / "state.json").read_text())["vectorize"]["info"]["documents"]
        vocab = json.loads((run / "vectorizer" / "vocabulary.json").read_text())
        idf = np.load(run / "vectorizer" / "idf.npy")
        df = np.round((1 + n) / np.exp(idf - 1) - 1)  # inverse of the smooth idf
        df_by_field[f], n_docs[f] = dict(zip(vocab, df.tolist())), n
        total.update(df_by_field[f])
        H = np.load(run / "model" / "components.npy")
        model_vocab = json.loads((run / "model" / "vocabulary.json").read_text())
        P = H / H.sum(0, keepdims=True).clip(1e-12)
        spread = -(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(0) / np.log(H.shape[0])
        in_top = np.zeros(H.shape[1], int)
        for row in np.argsort(-H, axis=1)[:, :10]:
            in_top[row] += 1
        topic_stats[f] = ({w: i for i, w in enumerate(model_vocab)}, spread, in_top)

    variants = build_variants(total, ["plural", "spelling"])
    groups = collections.defaultdict(set)
    for w in total:
        groups[variants.get(w, w)].add(w)
    rows = []
    for name, members in groups.items():
        share, spread, top10 = [], [], 0
        for f in fields:
            share.append(min(1.0, sum(df_by_field[f].get(m, 0) for m in members) / n_docs[f]))
            index, sp, in_top = topic_stats[f]
            cols = [index[m] for m in members if m in index]
            if cols:
                spread.append(max(sp[c] for c in cols))
                top10 += int(sum(in_top[c] for c in cols))
        share = np.sort(share)
        rows.append((name, np.median(share), share[1], np.median(spread) if spread else 0.0, top10, sorted(members)))
    rows.sort(key=lambda r: -r[2])
    with a.out.open("w") as out:
        out.write("word\tmedian_doc_share\tsecond_lowest_doc_share\ttopic_spread\ttop10_count\tmembers\n")
        for r in rows[: a.top]:
            out.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.2f}\t{r[4]}\t{','.join(r[5])}\n")
    print(f"{len(rows)} word groups in {len(fields)} runs; top {a.top} written to {a.out}")


if __name__ == "__main__":
    main()
