import numpy as np
import polars as pl
import pytest

from tst import io, network, trajectories
from tst.analysis import intertopic_flow, main_topics
from tst.config import EntityTrajectoryConfig, NetworkConfig
from tst.text import Tokenizer
from tst.topics import nmf
from tst.topics.model import TopicModel
from tst.topics.vectorize import Vectorizer


# --- reading ---------------------------------------------------------------
def test_csv_scopus_style(tmp_path):
    from tst.config import InputConfig

    (tmp_path / "s.csv").write_text(
        'Authors,Author(s) ID,Title,Year,Source title,Cited by,Abstract,EID\n'
        '"Doe J.; Roe R.","123; 456","On graphs",2019,"J. Graphs",4,"We study graphs, a lot.",2-s2.0-1\n'
        '"Poe P.",,"Untitled",,"J. Graphs",,"no year",2-s2.0-2\n'
        '"Moe M.; Doe J.","ORCID-9; 123","More graphs",2021-05-01,,,"",2-s2.0-3\n')
    cfg = InputConfig(path=tmp_path / "s.csv", format="csv", csv={"columns": {
        "authors": "Authors", "author_ids": "Author(s) ID", "title": "Title", "year": "Year",
        "venue": "Source title", "citations": "Cited by", "abstract": "Abstract", "source_id": "EID"}})
    info = io.ingest(cfg, tmp_path / "d.parquet")
    assert info["documents"] == 2 and info["rows_without_year"] == 1
    df = pl.read_parquet(tmp_path / "d.parquet")
    a, b = df.to_dicts()
    assert a["year"] == 2019 and a["venue"] == "J. Graphs" and a["citations"] == 4
    assert a["abstract"] == "We study graphs, a lot." and a["source_id"] == "2-s2.0-1"
    assert a["authors"] == [{"id": 123, "name": "Doe J."}, {"id": 456, "name": "Roe R."}]
    assert b["year"] == 2021 and b["venue"] is None and b["abstract"] is None
    assert b["authors"][0]["name"] == "Moe M." and b["authors"][0]["id"] < 0 and b["authors"][1]["id"] == 123


def test_csv_tab_separated_with_canonical_names(tmp_path):
    from tst.config import InputConfig

    (tmp_path / "w.tsv").write_text("year\ttitle\tabstract\tauthors\n2020\tT\tAn abstract\tA, B; C, D\n")
    cfg = InputConfig(path=tmp_path / "w.tsv", format="csv", csv={"separator": "\t"})
    io.ingest(cfg, tmp_path / "d.parquet")
    row = pl.read_parquet(tmp_path / "d.parquet").row(0, named=True)
    assert [x["name"] for x in row["authors"]] == ["A, B", "C, D"] and row["authors"][0]["id"] is None


def test_csv_folder_of_exports_with_bom(tmp_path):
    from tst.config import InputConfig

    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "a.txt").write_bytes(b"\xef\xbb\xbfPY\tTI\tAB\n2020\tA\tFirst\n")
    (tmp_path / "exports" / "b.txt").write_bytes(b"\xef\xbb\xbfPY\tTI\tAB\n2021\tB\tSecond\n")
    cfg = InputConfig(path=tmp_path / "exports", format="csv",
                      csv={"separator": "\t", "columns": {"year": "PY", "title": "TI", "abstract": "AB"}})
    io.ingest(cfg, tmp_path / "d.parquet")
    assert pl.read_parquet(tmp_path / "d.parquet")["title"].to_list() == ["A", "B"]


def test_csv_reports_missing_columns(tmp_path):
    from tst.config import InputConfig

    (tmp_path / "x.csv").write_text("PY,TI\n2020,T\n")
    with pytest.raises(ValueError, match="input.csv.columns"):
        io.ingest(InputConfig(path=tmp_path / "x.csv", format="csv"), tmp_path / "d.parquet")


def test_tokenizer_strips_markup_and_stopwords():
    tok = Tokenizer()
    tokens, share, _, foreign, other = tok.analyze("We study $x^2$ the <b>Neural</b> networks at http://x.org with 3 "
                                                   "layers, x < 5 and y > 3 &lt;sup&gt;2&lt;/sup&gt; &amp; maps")
    assert tokens == ["study", "neural", "networks", "layers", "maps"]
    assert 0 < share < 1 and foreign == 0 and other == 0


def test_tokenizer_apostrophes_accents_and_scripts():
    tok = Tokenizer(join_apostrophes=True, fold_accents=True, latin_only=True)
    assert tok("Islamic preaching (da’wah) in the Qur'an, Kant's critique, students' views, we don't know, "
               "O’Brien in Hawaiʻi") == ["islamic", "preaching", "dawah", "quran", "kant", "critique", "students",
                                         "views", "know", "obrien", "hawaii"]
    assert tok("Schrödinger, Poincaré, Straße, Łódź, İstanbul, non\u2010linear, inter\u00adnational") == \
        ["schrodinger", "poincare", "strasse", "lodz", "istanbul", "non", "linear", "international"]
    tokens, *_, other = tok.analyze("Ignition delay of bipropellants 加圧状態の空気または酸素雰囲気のなかにおいて,燃料希薄の混合条件")
    assert tokens == ["ignition", "delay", "bipropellants"] and other > 0.5
    tokens, *_, other = tok.analyze("Свойства материалов and the properties of materials")
    assert tokens == ["properties", "materials"] and 0 < other < 0.5
    # the old behaviour, as in model bundles written before these options existed
    assert Tokenizer()("Kant's Schrödinger Свойства") == ["kant", "schrödinger", "свойства"]


def test_tokenizer_removes_old_database_formatting():
    import re

    from tst.config import PreprocessConfig
    tok = Tokenizer()
    text = ("Scattering on He/sup 4/ and O/sup 16/ (>100 /spl mu/m, E/sub p/=24 GeV): f{sub L}, k{sup 2}, "
            "e{sup +}e{sup {minus}} {yields} {pi}{sup 0} {gamma}, YBa/sub 2/Cu/sub 3/O/sub 7-/spl delta// films, "
            "/sup 4/He/sup +/ /times/ e/sup +/e/sup /minus//, {ital K}{sup +}{r arrow}")
    assert tok(text) == ["scattering", "gev", "yba", "films"]
    assert tok("Masses in {GeV} and input/output, plus/minus") == ["masses", "gev", "input", "output", "plus", "minus"]
    strip = re.compile("|".join(f"(?:{p})" for p in PreprocessConfig().strip_patterns), re.I)
    assert tok(strip.sub(" ", "Cross sections were measured. (tr-auth) (orig.)")) == ["cross", "sections", "measured"]


def test_language_vote_rejects_bilingual_with_foreign_main_text():
    tok = Tokenizer()
    _, share, en, de, _ = tok.analyze("Die Arbeit untersucht, wie sich die Theorie der Wahrnehmung von Kant und Hegel "
                            "auf das Problem der Freiheit auswirkt und ob dies mit der Ethik vereinbar ist. "
                            "Summary: this paper asks whether the theory of perception of Kant has an effect on the problem "
                            "of freedom.")
    assert share > 0.08 and de > en  # passes the old threshold, fails the vote


def test_english_part_drops_foreign_half_of_bilingual_abstract():
    tok = Tokenizer()
    text = ("Penelitian ini bertujuan untuk mengetahui dampak urbanisasi pada desa dan masyarakat yang ada. "
            "This study examines the effect of urbanization on villages and their communities. "
            "Key words: urbanization, village.")
    assert tok.english_part(text) == ("This study examines the effect of urbanization on villages and their "
                                      "communities. Key words: urbanization, village.")


def test_foreign_stopwords_keep_ambiguous_words():
    tok = Tokenizer(foreign_stopwords=True)
    assert tok("Los conceptos del nous und die Seele") == ["conceptos", "nous", "die", "seele"]
    assert tok("tambem nao") == []  # accent-stripped spellings are covered


def test_corpus_filters(tmp_path):
    from tst.config import PreprocessConfig
    from tst.text import corpus_filters

    abstracts = ["Thank you for reading defying hitler. People have search hundreds times for it.",
                 "Same text.", "same   TEXT.", "<jats:p />", "<jats:p />", "<jats:p />", "A real abstract.",
                 "Table S1. Primers used in this study.", "Additional file 1 lists the primers."]
    titles = ["T"] * 7 + ["Additional file 1 of Gene regulation in yeast", "Additional file 1 of a study"]
    titles[6] = None
    pl.DataFrame({"doc_id": range(len(abstracts)), "title": titles, "abstract": abstracts}).write_parquet(
        tmp_path / "d.parquet")
    skip = corpus_filters(tmp_path / "d.parquet", PreprocessConfig(boilerplate_min_count=3))
    assert skip == {0: "spam", 2: "duplicate", 3: "boilerplate", 4: "boilerplate", 5: "boilerplate", 7: "spam",
                    8: "spam"}
    # "^" anchors at the title, so an abstract starting like a supplement title is kept.
    pl.DataFrame({"doc_id": [0], "title": ["Primer design"], "abstract": ["Additional file 1 of the study ..."]}
                 ).write_parquet(tmp_path / "e.parquet")
    assert corpus_filters(tmp_path / "e.parquet", PreprocessConfig()) == {}
    # A corpus's own patterns are added to the defaults and labelled with their group.
    pl.DataFrame({"doc_id": [0, 1, 2], "title": ["PQ-27291-19 - Jan O'Sullivan", "Hospital beds", "x"],
                  "abstract": ["163. Deputy Jan O'Sullivan asked the Minister for Health the status of ...",
                               "Bed numbers in Irish hospitals.", "Thank you for reading this book."]}
                 ).write_parquet(tmp_path / "f.parquet")
    cfg = PreprocessConfig(extra_drop_patterns={"Parliamentary questions": [r"^pq-\d+-\d+\b"]})
    assert corpus_filters(tmp_path / "f.parquet", cfg) == {0: "spam", 2: "spam"}
    assert cfg.drop_pattern_labels()[r"^pq-\d+-\d+\b"] == "Parliamentary questions"
    assert cfg.all_drop_patterns()[: len(cfg.drop_patterns)] == cfg.drop_patterns
    assert "extra_drop_patterns" not in PreprocessConfig().fingerprint_dump()
    assert cfg.fingerprint_dump()["extra_drop_patterns"] == cfg.extra_drop_patterns


def test_preprocess_rejects_abstracts_in_other_scripts(tmp_path):
    from tst.config import PreprocessConfig
    from tst.text import preprocess

    english = ("This study examines how the spectral properties of thin films depend on the substrate temperature and "
               "the deposition rate in magnetron sputtering.")
    japanese = "加圧状態の空気または酸素雰囲気のなかにおいて,燃料希薄の混合条件で白色発煙硝酸を酸化剤とする着火遅れについて研究した。"
    pl.DataFrame({"doc_id": [0, 1], "title": ["Thin films", "Effect of some parameters on the ignition delay of "
                                              "hypergolic bipropellants under pressurized atmosphere"],
                  "abstract": [english, japanese]}).write_parquet(tmp_path / "d.parquet")
    cfg = PreprocessConfig(min_tokens=3)
    preprocess(tmp_path / "d.parquet", tmp_path / "t.parquet", cfg, Tokenizer.from_config(cfg), 1)
    assert pl.read_parquet(tmp_path / "t.parquet").sort("doc_id")["reason"].to_list() == ["ok", "language"]


def test_preprocess_leaves_out_titles_in_other_languages(tmp_path):
    from tst.config import LanguageFilterConfig, PreprocessConfig
    from tst.text import preprocess

    english = ("This study examines the effect of service quality and price on customer satisfaction and loyalty "
               "in online retail, using a survey of customers and structural equation modelling.")
    titles = ["Pengaruh Kualitas Pelayanan terhadap Kepuasan Pelanggan",      # Indonesian
              "Calidad del servicio y satisfacción de los clientes",          # Spanish
              "Van der Waals forces in customer satisfaction models",         # English with a name particle
              "Service quality and customer satisfaction"]                    # English
    pl.DataFrame({"doc_id": [0, 1, 2, 3], "title": titles, "abstract": [english] * 4}
                 ).write_parquet(tmp_path / "d.parquet")
    cfg = PreprocessConfig(min_tokens=3, dedupe=False)
    tok = Tokenizer.from_config(cfg)
    assert tok.skip_foreign_titles
    assert [tok.foreign_title(t) for t in titles] == [True, True, False, False]
    info = preprocess(tmp_path / "d.parquet", tmp_path / "t.parquet", cfg, tok, 1)
    t = pl.read_parquet(tmp_path / "t.parquet").sort("doc_id")
    assert t["reason"].to_list() == ["ok"] * 4 and info["titles_ignored"] == 2
    assert t["title_ignored"].to_list() == [True, True, False, False]
    assert not {"pengaruh", "kualitas", "pelanggan", "calidad", "clientes"} & set(t["tokens"].explode().to_list())
    assert "waals" in t["tokens"][2].to_list()
    # Off: the title stays.
    off = PreprocessConfig(min_tokens=3, language_filter=LanguageFilterConfig(ignore_foreign_titles=False))
    assert not Tokenizer.from_config(off).ignores_title(titles[0], english, ["title", "abstract"])
    # Without an abstract the title is all there is, and it decides the language.
    assert not tok.ignores_title(titles[0], None, ["title", "abstract"])
    assert "sciencefiction" in tok("Science fiction and science-fiction film")


# --- trajectories ----------------------------------------------------------
def _docs(rows):
    """rows: (year, venue, theta)"""
    theta = np.array([r[2] for r in rows], dtype=np.float32)
    docs = pl.DataFrame({
        "row": np.arange(len(rows), dtype=np.uint32), "year": [r[0] for r in rows],
        "venue": [r[1] for r in rows], "authors": [[] for _ in rows],
    }, schema_overrides={"authors": pl.List(pl.Struct({"id": pl.Int64, "name": pl.String}))})
    return docs, theta


ROWS = [(2000, "V", [1, 0]), (2000, "V", [0, 1]), (2001, "V", [1, 0]), (2003, "V", [0, 1]),
        (2001, "W", [0, 1])]


def test_yearly_centroids():
    docs, theta = _docs(ROWS)
    traj, C = trajectories.build(docs, theta, "venue", EntityTrajectoryConfig(min_papers=1, window=1), {})
    v = traj.filter(pl.col("entity_id") == "V")
    assert v["year"].to_list() == [2000, 2001, 2003]
    np.testing.assert_allclose(C[:3], [[.5, .5], [1, 0], [0, 1]])
    assert v["n_papers"].to_list() == [2, 1, 1]


def test_no_entity_above_threshold():
    docs, theta = _docs(ROWS)
    traj, C = trajectories.build(docs, theta, "venue", EntityTrajectoryConfig(min_papers=10), {})
    assert traj.height == 0 and C.shape == (0, 2) and "step" in traj.columns


@pytest.mark.parametrize("mode,expected", [
    # window 3 over the years present: 2000 -> {2000}, 2001 -> {2000, 2001}, 2003 -> {2001, 2003}
    ("centroid_mean", [[.5, .5], [.75, .25], [.5, .5]]),
    ("pooled", [[.5, .5], [2 / 3, 1 / 3], [.5, .5]]),
    ("cumulative", [[.5, .5], [2 / 3, 1 / 3], [.5, .5]]),
])
def test_smoothing(mode, expected):
    docs, theta = _docs(ROWS)
    cfg = EntityTrajectoryConfig(min_papers=1, window=3, smoothing=mode)
    traj, C = trajectories.build(docs, theta, "venue", cfg, {})
    np.testing.assert_allclose(C[:3], expected, atol=1e-6)
    if mode == "pooled":
        assert traj["n_papers"].to_list()[:3] == [2, 3, 2]


def test_point_metrics_and_summary():
    docs, theta = _docs(ROWS)
    traj, C = trajectories.build(docs, theta, "venue", EntityTrajectoryConfig(min_papers=1, window=1), {})
    v = traj.filter(pl.col("entity_id") == "V")
    assert v["step"][0] is None
    np.testing.assert_allclose(v["step"][1], trajectories.hellinger(np.array([.5, .5]), np.array([1., 0.])))
    np.testing.assert_allclose(v["step"][2], 1.0)  # disjoint distributions
    assert v["speed"][2] == pytest.approx(0.5)  # two-year gap
    s = trajectories.entity_summary(traj, C).filter(pl.col("entity_id") == "V")
    assert s["path_length"][0] == pytest.approx(v["step"][1] + 1.0)
    assert s["displacement"][0] == pytest.approx(v["step"][1])


def test_min_papers_filter():
    docs, theta = _docs(ROWS)
    traj, _ = trajectories.build(docs, theta, "venue", EntityTrajectoryConfig(min_papers=2, window=1), {})
    assert traj["year"].to_list() == [2000]


def test_lead_lag():
    # b repeats a's path one year later, so a leads b.
    path = [[1 - t / 5, t / 5, 0] for t in range(6)]  # steady drift from topic 0 to topic 1
    years = list(range(2000, 2006))
    traj = pl.DataFrame({"entity_type": "venue", "entity_id": ["a"] * 6 + ["b"] * 5,
                         "entity_name": ["a"] * 6 + ["b"] * 5, "year": years + years[1:], "n_papers": 1})
    C = np.array(path + path[:5], dtype=np.float32)
    from tst.config import PairwiseConfig

    pw = trajectories.pairwise(traj, C, PairwiseConfig(max_lag=2, min_overlap=2))
    row = pw.row(0, named=True)
    sign = 1 if row["a"] == "a" else -1
    assert sign * row["lead_score"] > 0 and row["best_lag"] == 1


# --- topic model -----------------------------------------------------------
def test_nnls_recovers_mixture():
    rng = np.random.default_rng(0)
    H = rng.random((3, 40)).astype(np.float32)
    W = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5]], dtype=np.float32)
    import scipy.sparse as sp

    X = sp.csr_matrix(W @ H)
    vec = Vectorizer([f"t{i}" for i in range(40)], np.ones(40))
    model = TopicModel(Tokenizer(), ["title", "abstract"], vec, H)
    theta, strength = model.transform_matrix(X)
    np.testing.assert_allclose(theta, W / W.sum(axis=1, keepdims=True), atol=1e-3)
    assert (strength > 0).all()


def test_empty_document_gets_zero_row():
    import scipy.sparse as sp

    model = TopicModel(Tokenizer(), ["abstract"], Vectorizer(["a", "b"], np.ones(2)), np.eye(2, dtype=np.float32))
    theta, strength = model.transform_matrix(sp.csr_matrix((1, 2), dtype=np.float32))
    assert strength[0] == 0 and not np.isnan(theta).any()


def test_stability_metrics():
    tops = np.array([[1, 2, 3], [4, 5, 6]])
    assert nmf.ranking_agreement(tops, tops[::-1]) == pytest.approx(1.0)
    assert nmf.ranking_agreement(tops, np.array([[7, 8, 9], [10, 11, 12]])) == 0.0
    assert nmf.average_jaccard(np.array([1, 2]), np.array([2, 1])) == pytest.approx(0.5)


def test_vectorizer_weights_are_l2_normalised():
    vec = Vectorizer(["a", "b", "c"], np.array([1.0, 2.0, 3.0]), tf="raw")
    X = vec.transform([["a", "a", "b"], ["zzz"]]).toarray()
    np.testing.assert_allclose(np.linalg.norm(X[0]), 1.0)
    np.testing.assert_allclose(X[0, :2] / X[0, 1], [1.0, 1.0])  # 2*1 vs 1*2
    assert X[1].sum() == 0


def test_variants_merge_plurals_and_spellings_that_occur():
    from tst.topics.curation import build_variants

    df = {"model": 50, "models": 80, "genetic": 30, "genetics": 10, "studies": 40, "study": 60, "process": 9,
          "processes": 5, "behaviour": 4, "behaviours": 2, "behavior": 20, "analyses": 3, "analysis": 30,
          "news": 7, "new": 90, "gas": 5, "gases": 2, "lens": 3, "colours": 2, "color": 5, "physics": 9}
    v = build_variants(df, ["plural", "spelling"])
    assert v["model"] == "models" and v["genetics"] == "genetic" and v["studies"] == "study"
    assert v["processes"] == "process" and v["analyses"] == "analysis" and v["gases"] == "gas"
    assert v["behaviour"] == v["behaviours"] == "behavior" and v["colours"] == "color"
    assert "news" not in v and "new" not in v and "lens" not in v and "physics" not in v  # no "physic"
    assert build_variants(df, ["plural"]).get("behaviour") is None
    assert build_variants(df, []) == {}
    # -es is stripped only after s/x/z/ch/sh/o, a plural is never stripped to another plural, and singulars in
    # -s that look like plurals stay apart (false merges seen in the philosophy vocabulary).
    df = {"license": 90, "licensees": 5, "aid": 50, "aides": 4, "lenses": 6, "lens": 9, "len": 2, "man": 80,
          "men": 40, "mens": 3, "menses": 2, "bias": 30, "bia": 2, "biases": 8, "amine": 3, "amines": 4, "amin": 6}
    v = build_variants(df, ["plural", "spelling"])
    assert "licensees" not in v and "aides" not in v and "mens" not in v and "menses" not in v
    assert v["men"] == "man" and v["lenses"] == "lens" and v["biases"] == "bias" and v["amine"] == "amines"
    assert "lens" not in v and "len" not in v and "bias" not in v and "amin" not in v


def test_fit_vectorizer_merges_variants_and_drops_stopword_groups(tmp_path):
    from tst.config import VectorizeConfig
    from tst.topics.vectorize import build_matrix, fit_vectorizer

    docs = [["models", "genetics", "study"], ["model", "genetic", "studies"], ["models", "neuron", "results"],
            ["model", "neuron", "genetic"]]
    path = tmp_path / "tokens.parquet"
    pl.DataFrame({"doc_id": range(4), "tokens": docs, "reason": ["ok"] * 4}).write_parquet(path)
    cfg = VectorizeConfig(min_df=1, max_df=1.0, keep=["results"])
    vec, info = fit_vectorizer(path, cfg, n_jobs=2)
    assert sorted(vec.vocabulary) == ["genetic", "model", "neuron", "results"]  # "results" kept by cfg.keep
    assert info["stopwords_removed"] == 1  # study/studies
    i = vec.index["model"]
    assert vec.idf[i] == pytest.approx(np.log(5 / 5) + 1)  # in all 4 documents, not 2 + 2
    X, ids = build_matrix(path, vec, n_jobs=2)
    assert (X[:, i].toarray() > 0).all()
    vec.save(tmp_path / "vec")
    assert Vectorizer.load(tmp_path / "vec").variants["models"] == "model"
    plain, _ = fit_vectorizer(path, VectorizeConfig(min_df=1, max_df=1.0, conflate=[], academic_stopwords=False), 2)
    assert {"models", "studies", "study"} <= set(plain.vocabulary)


def test_tokenizer_joins_hyphenated_prefixes():
    tok = Tokenizer(join_prefixes=True)
    assert tok("Non-linear e-learning, X-ray co-operation, D-glucose and well-being, a state-of-the-art "
               "method") == ["nonlinear", "elearning", "xray", "cooperation", "glucose", "wellbeing",
                             "stateoftheart", "method"]
    assert Tokenizer()("non-linear") == ["non", "linear"]


# --- network ---------------------------------------------------------------
def _write_docs(tmp_path, papers):
    """papers: (doc_id, year, [author ids])"""
    df = pl.DataFrame({
        "doc_id": [p[0] for p in papers], "year": [p[1] for p in papers],
        "authors": [[{"id": a, "name": f"n{a}"} for a in p[2]] for p in papers],
    })
    path = tmp_path / "documents.parquet"
    df.write_parquet(path)
    return path


def test_network_orientation_and_window(tmp_path):
    # Topic 0 everywhere. Author 1 has two papers, author 2 one: 1 is the expert, edge 1 -> 2.
    docs = _write_docs(tmp_path, [(0, 2000, [1, 2]), (1, 2000, [1]), (2, 2002, [3, 4])])
    theta = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    cfg = NetworkConfig(min_year=None, min_author_papers=1, top_k_topics=1, window=2)
    info = network.build(docs, np.array([0, 1, 2]), theta, cfg, tmp_path / "net")
    edges = pl.read_parquet(tmp_path / "net" / "edges.parquet").sort("year", "src")
    e2000 = edges.filter(pl.col("year") == 2000)
    assert e2000.select("src", "dst", "topic", "weight").rows() == [(1, 2, 0, 1.0)]
    # The window of 2 years carries the 2000 edge into 2001.
    assert edges.filter(pl.col("year") == 2001).height == 1
    # Equal expertise (authors 3 and 4 only share one paper): both directions.
    assert sorted(edges.filter(pl.col("year") == 2002).select("src", "dst").rows()) == [(3, 4), (4, 3)]
    ex = pl.read_parquet(tmp_path / "net" / "expertise.parquet")
    assert ex.filter((pl.col("author") == 1) & (pl.col("year") == 2000))["weight"][0] == pytest.approx(2.0)
    assert info["collaborations"] == 2


def test_network_top_k_and_caps(tmp_path):
    docs = _write_docs(tmp_path, [(0, 2000, [1, 2]), (1, 2000, [1, 2, 3, 4])])
    theta = np.array([[.6, .3, .1], [.5, .5, 0]], dtype=np.float32)
    cfg = NetworkConfig(min_year=None, min_author_papers=1, top_k_topics=2, window=1, max_authors_per_paper=3)
    network.build(docs, np.array([0, 1]), theta, cfg, tmp_path / "net")
    edges = pl.read_parquet(tmp_path / "net" / "edges.parquet")
    assert set(edges["topic"].to_list()) == {0, 1}  # topic 2 dropped by top_k
    assert set(edges["src"].to_list()) | set(edges["dst"].to_list()) == {1, 2}  # paper 1 exceeds the cap


def test_intertopic_flow():
    ex = pl.DataFrame({"author": [1, 1, 2], "year": 2000, "topic": [0, 1, 1], "weight": [3.0, 1.0, 2.0]})
    mt = main_topics(ex)
    assert dict(mt.rows()) == {1: 0, 2: 1}
    edges = pl.DataFrame({"src": [1, 1], "dst": [2, 2], "year": 2000, "topic": [0, 1], "weight": [0.5, 0.7]})
    flow = intertopic_flow(edges, mt, 2)
    # Only the edge in the source's main topic counts: 0 -> 1 with weight 0.5.
    np.testing.assert_allclose(flow, [[0, 0.5], [0, 0]])


def test_default_n_jobs_uses_all_cores(tmp_path):
    import os

    from tst.config import load_config

    (tmp_path / "c.yaml").write_text("name: x\nworkdir: w\ninput: {path: in.csv, format: csv}\n")
    assert load_config(tmp_path / "c.yaml").n_jobs == os.cpu_count()


def test_last_complete_year_drops_partial_trailing_years():
    counts = {2015: 100, 2016: 110, 2017: 120, 2018: 120, 2019: 130, 2020: 125, 2021: 40, 2022: 1}
    years = np.repeat(list(counts), list(counts.values()))
    assert trajectories.last_complete_year(years) == 2020
    # A complete last year is kept.
    assert trajectories.last_complete_year(years[years <= 2020]) == 2020
    assert trajectories.last_complete_year(np.array([], dtype=np.int64)) is None
    # Stray future years with a sparse run of years before them.
    counts = {2023: 900, 2024: 1000, 2025: 1000, 2026: 700, 2027: 1, 2029: 2, 2030: 2}
    years = np.repeat(list(counts), list(counts.values()))
    assert trajectories.last_complete_year(years) == 2026


def test_select_fine_k():
    import pandas as pd

    t = pd.DataFrame({"k": [10, 20, 30, 40], "stability": [0.9, 0.8, 0.6, 0.75], "npmi": [0.2, 0.3, 0.1, 0.1]})
    assert nmf.select_fine_k(t, 20) == 40          # largest stable k above the main k
    assert nmf.select_fine_k(t, 20, min_stability=0.95) == 30   # none stable: the next larger k
    with pytest.raises(ValueError):
        nmf.select_fine_k(t, 40)


def test_author_memberships_skip_large_collaborations():
    docs = pl.DataFrame({"row": [0, 1], "year": [2020, 2020], "venue": ["V", "V"], "authors": [
        [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        [{"id": i, "name": str(i)} for i in range(60)]]})
    assert trajectories.memberships(docs, "author", False, max_authors=50)["row"].unique().to_list() == [0]
    assert trajectories.memberships(docs, "author", False)["row"].n_unique() == 2
    assert trajectories.memberships(docs, "venue", False, max_authors=50).height == 2


def test_method_documentation_is_complete():
    from tst.config import DEFAULT_DROP_PATTERNS, DEFAULT_STRIP_PATTERNS, PATTERN_LABELS
    from tst.references import REFERENCES, bibliography, short_citation
    from tst.topics.curation import academic_stopword_list, academic_stopwords

    # Every default cleaning pattern has a group name for the bundle.
    assert all(p in PATTERN_LABELS for p in DEFAULT_DROP_PATTERNS + DEFAULT_STRIP_PATTERNS)
    assert short_citation("schofield2016") == "Schofield & Mimno 2016"
    assert short_citation("greene2014") == "Greene et al. 2014" and short_citation("harman1991") == "Harman 1991"
    assert [r["id"] for r in bibliography(["lau2014", "dieng2020", "lau2014"])] == ["lau2014", "dieng2020"]
    assert all({"authors", "year", "title", "venue", "url"} <= r.keys() for r in REFERENCES.values())
    # The stop-word list: categories, references, and the candidates left off on purpose are not on it.
    lst = academic_stopword_list()
    assert set(lst["references"]) <= REFERENCES.keys() and lst["date"]
    left_off = {w for e in lst["not_stopwords"] for w in e["words"]}
    assert {"model", "data", "current"} <= left_off and not left_off & academic_stopwords()
    assert {"study", "research", "review", "does"} <= academic_stopwords()


# --- derived measures ------------------------------------------------------
def test_mean_shortest_path_exact_and_sampled():
    import igraph as ig

    from tst.analysis import mean_shortest_path

    path = ig.Graph.Ring(10, circular=False)          # mean distance of a path with n nodes: (n + 1) / 3
    assert mean_shortest_path(path, 100, 0) == pytest.approx(11 / 3)
    g = ig.Graph.Lattice([30, 30], circular=True)       # sampled estimate close to the exact mean
    exact = g.average_path_length(directed=False)
    assert mean_shortest_path(g, 300, 1) == pytest.approx(exact, rel=0.02)
    two = ig.Graph(n=5, edges=[(0, 1), (1, 2), (3, 4)])  # only the largest component counts
    assert mean_shortest_path(two, 100, 0) == pytest.approx(4 / 3)


def test_novelty_scores_rare_topic_pairs(tmp_path):
    from tst.config import NoveltyConfig
    from tst.novelty import compute

    rng = np.random.default_rng(0)
    rows = []
    # Years 2000-2002: topics 0+1 and 2+3 are combined, never 0+2.
    for y in (2000, 2001, 2002):
        for _ in range(400):
            a = rng.integers(2)
            rows.append((y, [0.5, 0.5, 0, 0] if a else [0, 0, 0.5, 0.5]))
    rows += [(2003, [0.5, 0, 0.5, 0]), (2003, [0.5, 0.5, 0, 0]), (2003, [1.0, 0, 0, 0])]
    n = len(rows)
    pl.DataFrame({"doc_id": np.arange(n), "theta": [r[1] for r in rows]},
                 schema={"doc_id": pl.Int64, "theta": pl.Array(pl.Float32, 4)}).write_parquet(tmp_path / "dt.parquet")
    pl.DataFrame({"doc_id": np.arange(n), "year": [r[0] for r in rows],
                  "citations": np.arange(n) % 7}).write_parquet(tmp_path / "docs.parquet")
    info = compute(tmp_path / "dt.parquet", tmp_path / "docs.parquet", NoveltyConfig(min_window_papers=500),
                   tmp_path / "nov.parquet")
    nov = pl.read_parquet(tmp_path / "nov.parquet").filter(pl.col("year") == 2003).sort("doc_id")
    rare, common, single = nov["novelty"].to_list()
    assert rare > 2 and common < 0 and single is None      # 0+2 never seen; 0+1 twice as often as expected
    assert nov["unusual"].to_list() == [True, False, False]
    assert info["years"] == [2002, 2003]                    # 2001 has only 400 papers in its window
    assert pl.read_parquet(tmp_path / "nov.parquet")["citation_pct"].is_between(0, 100).all()


def test_life_cycle_status():
    from tst.config import IndicatorsConfig
    from tst.indicators import life_cycle

    years = list(range(2000, 2020))
    cfg = IndicatorsConfig()
    rising = life_cycle(years, np.r_[np.full(12, 0.01), np.linspace(0.02, 0.1, 8)], cfg)
    assert rising["status"] == "emerging" and rising["peak_year"] >= 2017
    falling = life_cycle(years, np.r_[np.full(15, 0.1), np.full(5, 0.05)], cfg)
    assert falling["status"] == "declining" and falling["recent_change"] == pytest.approx(-0.5)
    assert life_cycle(years, np.full(20, 0.05), cfg)["status"] == "stable"


def test_topic_names_from_tsv(tmp_path):
    from tst.topics.model import read_labels

    topics = [{"id": i, "top_terms": [{"term": t} for t in terms]}
              for i, terms in enumerate([["god", "theology", "christian", "divine", "church"],
                                         ["kant", "critique", "reason", "transcendental", "pure"]])]
    f = tmp_path / "labels.tsv"
    f.write_text("id\tname\tterms\n0\tTheology\tgod, theology, christian, divine, church\n"
                 "1\tKant\tkant, critique, reason, transcendental, pure\n")
    assert read_labels(f, 2, topics) == ["Theology", "Kant"]
    # A retrained model with other topics behind the ids: the names are not used.
    assert read_labels(f, 2, list(reversed(topics))) == []
    assert read_labels(tmp_path / "missing.tsv", 2, topics) == []
    assert read_labels(f, 3, topics + [topics[0]]) == []             # names for 2 topics, model has 3
    (tmp_path / "plain.txt").write_text("A\nB\n")
    assert read_labels(tmp_path / "plain.txt", 2) == ["A", "B"]


def test_scraped_publisher_pages_are_dropped(tmp_path):
    from tst.config import PreprocessConfig
    from tst.text import corpus_filters

    real = ("We studied the effect of early mobilisation after knee surgery in 120 patients and found shorter stays "
            "and fewer complications; users can add bookmarks to favourites in the app.")
    pages = [
        "Journal of Urology, VOL. 85 Advertisement Copyright & PermissionsMetricsAuthor Information Expand All "
        "Advertisement PDF DownloadLoading ...",
        "https://doi.org/10.1137/1002061PDFBibTexSections ToolsAdd to favoritesExport CitationTrack CitationsEmail",
        "SectionsAboutPDF/EPUB ToolsAdd to FavouritesDownload CitationsTrack CitationsPermissions Share",
        "Article Tools OPTIONS & TOOLS Export Citation Track Citation Add To Favorites Rights & Permissions",
        "https://doi.org/10.1044/leader.FTR1.06202001.4 SectionsAbout ToolsAdd to favorites ShareFacebookTwitter",
    ]
    pl.DataFrame({"doc_id": range(len(pages) + 1), "title": ["t"] * (len(pages) + 1),
                  "abstract": [real] + pages}).write_parquet(tmp_path / "d.parquet")
    skip = corpus_filters(tmp_path / "d.parquet", PreprocessConfig())
    assert skip == {i: "spam" for i in range(1, len(pages) + 1)}
