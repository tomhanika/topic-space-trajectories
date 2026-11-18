"""End-to-end run on a synthetic publication list (CSV, as a user would bring it) with two well separated
topics."""

import json

import numpy as np
import polars as pl
import yaml

from tst.cli import main

SPORTS = "football match goal player team league coach stadium season tournament".split()
MUSIC = "guitar melody concert album singer orchestra rhythm piano symphony band".split()
FILLER = "the of and in with this that for which are".split()


def _corpus(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        topic = SPORTS if i % 2 == 0 else MUSIC
        words = list(rng.choice(topic, 30)) + list(rng.choice(FILLER, 10))
        rng.shuffle(words)
        authors = rng.choice(range(1, 30) if i % 2 == 0 else range(30, 60), 3, replace=False)
        rows.append({
            "year": 2000 + i % 10, "venue": "Sports Journal" if i % 2 == 0 else "Music Review",
            "title": f"Paper {i}", "abstract": " ".join(words),
            "authors": "; ".join(f"Author {a}" for a in authors),
            "author_ids": "; ".join(f"orcid-{a}" for a in authors), "citations": i % 7,
        })
    pl.DataFrame(rows).write_csv(path)


def test_full_pipeline(tmp_path):
    _corpus(tmp_path / "corpus.csv")
    cfg = {
        "name": "synthetic", "workdir": "run", "n_jobs": 4,
        "input": {"path": "corpus.csv", "format": "csv"},
        "vectorize": {"min_df": 2, "max_df": 0.9},
        "topics": {"k_grid": [2, 3], "runs": 2},
        "trajectories": {"entities": {"venue": {"min_papers": 2, "window": 2},
                                      "author": {"min_papers": 1, "window": 3}},
                         "pairwise": {"min_overlap": 2}},
        "projection": {"method": "pca"},
        "network": {"min_year": None, "min_author_papers": 2},
        "novelty": {"min_window_papers": 10},
        "terms": {"min_bigram_papers": 5},
        "export": {"density_grid": 32, "notes": {"vectorize.keep": "A test note.", "foo.bar": "No parameter."},
                   "labels_file": "labels.tsv", "label": "Synthetic corpus", "group": "Tests"},
        "fine": {"enabled": True},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    main(["run", str(tmp_path / "cfg.yaml")])

    run = tmp_path / "run"
    state = json.loads((run / "state.json").read_text())
    assert state["train"]["info"]["k"] == 2
    assert state["ingest"]["info"]["documents"] == 400

    # The two venues end up in different topics.
    traj = pl.read_parquet(run / "bundle" / "trajectories.parquet").filter(pl.col("entity_type") == "venue")
    main_by_venue = traj.group_by("entity_id").agg(pl.col("main_topic").mode().first())
    assert main_by_venue["main_topic"].n_unique() == 2

    manifest = json.loads((run / "bundle" / "manifest.json").read_text())
    for f in manifest["files"].values():
        assert (run / "bundle" / f).exists()
    flows = pl.read_parquet(run / "bundle" / "network" / "flows.parquet")
    assert flows.height > 0
    # The settings travel with the bundle: the resolved config and a description for readers.
    assert manifest["run"]["config"]["topics"]["k_grid"] == [2, 3] and "workdir" not in manifest["run"]["config"]
    params = {g["group"]: {p["label"]: p for p in g["parameters"]} for g in manifest["run"]["parameters"]}
    assert params["Topic model"]["Topics"]["value"] == "2"
    assert params["Trajectories"]["Authors"]["value"].startswith("at least 1 paper in a year")
    # Every parameter is a decision record; the literature it cites and the lists it points to are in the bundle.
    assert manifest["schema_version"] == "1.5"
    refs = {r["id"]: r for r in manifest["run"]["references"]}
    for g in manifest["run"]["parameters"]:
        for p in g["parameters"]:
            assert {"rationale", "references", "note", "details"} <= p.keys() and set(p["references"]) <= refs.keys()
    assert refs["greene2014"]["citation"] == "Greene et al. 2014"
    assert params["Vocabulary"]["Academic stop words"]["note"] == "A test note."
    assert params["Notes"]["foo.bar"]["description"] == "No parameter."
    methods = json.loads((run / "bundle" / "methods.json").read_text())
    for p in (p for g in params.values() for p in g.values() if p["details"]):
        node = methods
        for part in p["details"].split("."):
            node = node[part]
    assert methods["vocabulary"]["stopwords"]["categories"][0]["words"]
    assert all(d["group"] for d in methods["cleaning"]["drop_patterns"])
    assert methods["cleaning"]["papers"]["kept"] == manifest["run"]["stages"]["preprocess"]["ok"]
    vocab = json.loads((run / "bundle" / "vocabulary.json").read_text())
    assert len(vocab["terms"]) == len(vocab["document_frequency"]) == manifest["run"]["stages"]["vectorize"]["features"]
    assert vocab["document_frequency"] == sorted(vocab["document_frequency"], reverse=True)
    assert 0 < vocab["document_frequency"][-1] <= vocab["document_frequency"][0] <= vocab["papers"]
    assert set(vocab["variants"].values()) <= set(vocab["terms"])

    # Derived measures: topic pages, field indicators, novelty and citation percentiles per paper.
    bundle = run / "bundle"
    topics = json.loads((bundle / "topics.json").read_text())
    assert all(t["name"] is None and t["label"] == t["terms_label"] for t in topics)  # no labels.tsv yet
    pages = json.loads((bundle / "topic_pages.json").read_text())
    assert [p["id"] for p in pages] == [0, 1] and all(p["typical_papers"] and p["venues"] for p in pages)
    assert {p["life_cycle"]["status"] for p in pages} <= {"emerging", "growing", "stable", "declining", None}
    field = json.loads((bundle / "field_indicators.json").read_text())
    assert [r["year"] for r in field["years"]] == list(range(2000, 2010))
    # Two clean topics: every paper has one topic only, so none has a novelty score.
    assert all(r["team_mean"] == 3 for r in field["years"]) and field["novelty_impact"]["single_topic"]["papers"] > 0
    docs = pl.read_parquet(bundle / "documents.parquet")
    assert {"novelty", "citation_pct", "unusual"} <= set(docs.columns) and docs["citation_pct"].is_between(0, 100).all()
    assert {"Paper novelty", "Citation impact", "Field indicators", "Topic names"} <= params["Derived measures"].keys()
    assert "Average shortest path" in params["Collaboration network"]
    sna = pl.read_parquet(bundle / "network" / "sna.parquet")
    assert sna.filter(pl.col("topic") == -1)["avg_shortest_path"].drop_nulls().drop_nans().min() >= 1
    terms = pl.read_parquet(run / "terms" / "terms.parquet")
    assert {"football", "guitar"} <= set(terms["term"]) and terms["term"].str.contains(" ").any()

    # Topic names: a labels file with the top terms they were written for; re-export only.
    tsv = ["id\tname\tterms"] + [f"{t['id']}\tTopic {t['id']}\t" + ", ".join(x["term"] for x in t["top_terms"][:5])
                                  for t in topics]
    (tmp_path / "labels.tsv").write_text("\n".join(tsv) + "\n")
    main(["run", str(tmp_path / "cfg.yaml"), "--stages", "export", "--force", "--resolution", "main"])
    named = json.loads((bundle / "topics.json").read_text())
    assert [t["label"] for t in named] == ["Topic 0", "Topic 1"] and named[0]["terms_label"] == topics[0]["terms_label"]

    # The fine resolution reuses the main run's documents and sweep and has its own bundle.
    fine = json.loads((run / "fine" / "state.json").read_text())
    assert "ingest" not in fine and fine["train"]["info"]["k"] == 3
    manifest = json.loads((run / "fine" / "bundle" / "manifest.json").read_text())
    assert manifest["k"] == 3 and manifest["resolution"] == "fine"
    assert manifest["run"]["stages"]["ingest"]["documents"] == 400
    assert not (run / "fine" / "documents.parquet").exists()
    params = {p["label"]: p for g in manifest["run"]["parameters"] for p in g["parameters"]}
    assert params["Topics"]["description"].startswith("Fine resolution") and "Scores" in params
    fine_topics = json.loads((run / "fine" / "bundle" / "topics.json").read_text())
    assert all(t["parent"]["id"] in (0, 1) for t in fine_topics)
    assert not (run / "fine" / "terms").exists()

    # `tst collect` gathers both bundles for a web app.
    main(["collect", str(tmp_path / "web"), str(tmp_path / "cfg.yaml")])
    index = json.loads((tmp_path / "web" / "index.json").read_text())
    assert index["global"] == {"terms": "_terms/index.json", "mobility": "_global/mobility.json"}
    explorer = json.loads((tmp_path / "web" / "_terms" / "index.json").read_text())
    assert explorer["fields"] == ["synthetic"] and "fo" in explorer["files"]
    fo = pl.read_parquet(tmp_path / "web" / "_terms" / explorer["files"]["fo"])
    assert 180 < fo.filter(pl.col("term") == "football")["papers"].sum() <= 200  # of the 200 sports papers
    mobility = json.loads((tmp_path / "web" / "_global" / "mobility.json").read_text())
    assert mobility["fields"] == ["synthetic"] and mobility["periods"][-1]["end"] == 2009
    assert index["fields"] == [{"name": "synthetic", "label": "Synthetic corpus", "group": "Tests", "resolutions": {
        "main": {"path": "synthetic", "k": 2, "created": index["fields"][0]["resolutions"]["main"]["created"]},
        "fine": {"path": "synthetic_fine", "k": 3,
                 "created": index["fields"][0]["resolutions"]["fine"]["created"]}}}]
    assert (tmp_path / "web" / "synthetic_fine" / "topics.json").exists()

    # A second run is a no-op, and so is a dry run.
    before = (run / "state.json").read_text()
    before_fine = (run / "fine" / "state.json").read_text()
    main(["run", str(tmp_path / "cfg.yaml")])
    assert (run / "state.json").read_text() == before
    assert (run / "fine" / "state.json").read_text() == before_fine
    from tst.config import load_config
    from tst.run import RunContext
    from tst.stages import run as run_stages

    assert run_stages(RunContext(load_config(tmp_path / "cfg.yaml"), "main"), dry_run=True) == []
    # Options at their default value are not in the fingerprints: writing them out changes nothing.
    cfg["preprocess"] = {"extra_drop_patterns": {}}
    cfg["export"]["publish"] = True
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    assert run_stages(RunContext(load_config(tmp_path / "cfg.yaml"), "main"), dry_run=True) == []
    # A corpus's own drop pattern re-runs everything from preprocess on, and nothing runs in a dry run.
    cfg["preprocess"] = {"extra_drop_patterns": {"Guitar spam": [r"\bguitar\b"]}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    outdated = run_stages(RunContext(load_config(tmp_path / "cfg.yaml"), "main"), dry_run=True)
    assert outdated[0] == "preprocess" and "ingest" not in outdated and "export" in outdated
    assert (run / "state.json").read_text() == before
    # An unpublished corpus is removed from a collected folder.
    cfg["preprocess"] = {}
    cfg["export"]["publish"] = False
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    main(["collect", str(tmp_path / "web"), str(tmp_path / "cfg.yaml"), "--no-global"])
    assert json.loads((tmp_path / "web" / "index.json").read_text())["fields"] == []
    assert not (tmp_path / "web" / "synthetic").exists() and not (tmp_path / "web" / "synthetic_fine").exists()
    cfg["export"]["publish"] = True
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))

    # The model bundle embeds new text into the right topic.
    from tst.topics.model import TopicModel

    model = TopicModel.load(str(run / "model"))  # as in the manual: a plain string path
    t_sport = model.transform_texts(["football goal player team league season"])[0]
    t_music = model.transform_texts(["guitar concert album singer piano"])[0]
    assert t_sport.argmax() != t_music.argmax()
