"""Naming worksheet for one map: per topic its top terms, typical and cited papers, venues and authors, and the
closest named topic of an earlier model as a candidate name.

Reads the exported bundle (topics.json, topic_pages.json). With --old, a folder with main_topics.json /
fine_topics.json of an earlier model (see backup.sh), every topic gets the earlier topic with the most similar
term weights (cosine over the top 20 terms) and its name. A carried name is only a candidate: retraining splits
and merges topics, so a reviewer checks every one (GUIDE.md).

    python scripts/naming/worksheet.py runs/law [--fine] [--old runs/_naming_backup/law] > law.md
"""

import argparse
import json
import math
from pathlib import Path


def _vec(topic: dict) -> dict[str, float]:
    return {t["term"]: t["weight"] for t in topic["top_terms"][:20]}


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na, nb = math.sqrt(sum(w * w for w in a.values())), math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--old", type=Path, help="folder with the earlier model's main_topics.json / fine_topics.json")
    ap.add_argument("--papers", type=int, default=10, help="typical papers per topic")
    args = ap.parse_args()
    res = "fine" if args.fine else "main"
    bundle = args.run / ("fine" if args.fine else "") / "bundle"
    topics = json.loads((bundle / "topics.json").read_text())
    pages = {p["id"]: p for p in json.loads((bundle / "topic_pages.json").read_text())}
    old = []
    if args.old and (args.old / f"{res}_topics.json").exists():
        old = [t for t in json.loads((args.old / f"{res}_topics.json").read_text()) if t.get("name")]
    parents = {}
    if args.fine:
        parents = {t["id"]: t.get("name") or t["terms_label"]
                   for t in json.loads((args.run / "bundle" / "topics.json").read_text())}

    print(f"# {args.run.name} ({res}): {len(topics)} topics\n")
    for t in topics:
        p = pages.get(t["id"], {})
        print(f"## Topic {t['id']}  ({p.get('papers', '?')} papers, {100 * t['prevalence']:.1f}% of the map)")
        print("Terms:", ", ".join(f"{x['term']} ({x['weight']:.3f})" for x in t["top_terms"][:15]))
        if args.fine and t.get("parent"):
            print(f"Parent topic: {t['parent']['id']} {parents.get(t['parent']['id'], '')}")
        if old:
            v = _vec(t)
            best = sorted(((_cos(v, _vec(o)), o) for o in old), key=lambda x: -x[0])[:2]
            print("Earlier names:", "; ".join(f"\"{o['name']}\" (cosine {c:.2f}; {o['terms_label']})" for c, o in best))
        lc = p.get("life_cycle") or {}
        if lc.get("peak_year"):
            print(f"Life cycle: {lc.get('status')}, peak {lc['peak_year']}")
        venues = ", ".join(f"{v['name']} ({v['papers']})" for v in p.get("venues", [])[:5])
        if venues:
            print("Venues:", venues)
        authors = ", ".join(f"{a['name']} ({a['papers']})" for a in p.get("authors", [])[:4])
        if authors:
            print("Authors:", authors)
        print("Typical papers:")
        for d in p.get("typical_papers", [])[: args.papers]:
            print(f"- {d['title'][:160]} ({d['year']}, {d.get('venue') or 'no venue'})")
        cited = p.get("cited_papers", [])[:3]
        if cited:
            print("Most cited:")
            for d in cited:
                print(f"- {d['title'][:160]} ({d['year']}, {d.get('venue') or 'no venue'}, {d['citations']} citations)")
        print()


if __name__ == "__main__":
    main()
