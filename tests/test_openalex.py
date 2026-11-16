import json

import pyarrow as pa
import pyarrow.parquet as pq

from tst import io, openalex
from tst.config import InputConfig


def test_abstract_text_rebuilds_word_order():
    inv = json.dumps({"topic": [1, 4], "Space": [0], "trajectories": [2], "of": [3]})
    assert openalex.abstract_text(inv) == "Space topic trajectories of topic"
    assert openalex.abstract_text(None) is None
    assert openalex.abstract_text("{}") is None
    assert openalex.abstract_text("not json") is None


def test_tail_id():
    assert openalex._tail_id("https://openalex.org/W2006176091") == 2006176091
    assert openalex._tail_id("https://openalex.org/A5036823227") == 5036823227
    assert openalex._tail_id("https://openalex.org/subfields/1211") == 1211
    assert openalex._tail_id(None) is None


def _snapshot_rows():
    def work(wid, year, subfield, abstract, typ="article", venue="Mind"):
        return {
            "id": f"https://openalex.org/W{wid}", "title": f"Paper {wid}", "publication_year": year,
            "language": "en", "type": typ, "cited_by_count": 3,
            "authorships": [{"author": {"id": "https://openalex.org/A7", "display_name": "Ada"}}],
            "primary_topic": {"subfield": {"id": f"https://openalex.org/subfields/{subfield}"},
                              "field": {"id": f"https://openalex.org/fields/{subfield // 100}"}},
            "primary_location": {"source": {"display_name": venue}},
            "abstract_inverted_index": json.dumps(abstract) if abstract else None,
        }
    return [work(1, 2001, 1211, {"being": [0]}), work(2, 2002, 1211, None),
            work(3, 2003, 2731, {"eye": [0]}), work(4, 2004, 1211, {"mind": [0]}, typ="erratum"),
            work(5, 2005, 1211, {"self": [0]}, venue="Zenodo"), work(6, 2006, 1211, {"good": [0]}, typ="book")]


def test_store_and_ingest(tmp_path):
    store = tmp_path / "store"
    (store / "works").mkdir(parents=True)
    tab = openalex._convert(pa.Table.from_pylist(_snapshot_rows()), part=0)
    assert tab.num_rows == 5  # work 2 has no abstract
    pq.write_table(tab, store / "works" / "a.parquet")
    # A newer copy of work 1 in a later partition wins.
    newer = openalex._convert(pa.Table.from_pylist(_snapshot_rows()[:1]), part=1)
    pq.write_table(newer.set_column(3, "title", pa.array(["Paper 1 v2"])), store / "works" / "b.parquet")
    pq.write_table(pa.table({"id": ["S1", "S2"], "display_name": ["Mind", "Zenodo"],
                             "type": ["journal", "repository"]}), store / "sources.parquet")

    cfg = InputConfig(path=store, format="openalex", openalex={"subfields": [1211]})
    out = tmp_path / "documents.parquet"
    info = io.ingest(cfg, out)
    import polars as pl

    df = pl.read_parquet(out)
    assert info["documents"] == 2  # the erratum, the book and the other subfield are left out
    row = df.row(0, named=True)
    assert row["title"] == "Paper 1 v2" and row["abstract"] == "being" and row["source_id"] == "W1"
    assert row["authors"] == [{"id": 7, "name": "Ada"}] and row["venue"] == "Mind"
    assert df.row(1, named=True)["venue"] is None  # a repository is no venue


def test_openalex_junk_patterns():
    import re

    from tst.config import DEFAULT_DROP_PATTERNS, DEFAULT_STRIP_PATTERNS

    strip = re.compile("|".join(f"(?:{p})" for p in DEFAULT_STRIP_PATTERNS), re.I)
    text = ("Kant's ethics. These keywords were added by machine and not by the authors. This process is "
            "experimental and the keywords may be updated as the learning algorithm improves.")
    assert strip.sub(" ", text).strip() == "Kant's ethics."
    assert " ".join(strip.sub(" ", "Let [Formula: see text] be a ring.").split()) == "Let be a ring."
    assert strip.sub(" ", "Summary form only given. We present InGaAs photodiodes.").strip() == "We present InGaAs photodiodes."
    clean = lambda t: " ".join(strip.sub(" ", t).split())  # noqa: E731
    assert clean("Abstract Background: Salamanders call. Methods: We recorded. (C) 2010 Elsevier Ltd. All rights "
                 "reserved.") == "Salamanders call. We recorded."
    assert clean("Copyright: 2013 Islam et al. This is an open-access article distributed under the terms of the "
                 "Creative Commons Attribution License, which permits unrestricted use, provided the original "
                 "author and source are credited. Cells grow.") == "Cells grow."
    assert clean("Abstract algebra studies groups (c) and rings.") == "Abstract algebra studies groups (c) and rings."
    drop = re.compile("|".join(f"(?:{p})" for p in DEFAULT_DROP_PATTERNS), re.I)
    assert drop.search("Wazifa for love marriage Call/WhatsApp on +91")
    assert drop.search("Search for other works by this author on: This Site PubMed Google Scholar")
    assert drop.search("ADVERTISEMENT RETURN TO ISSUEPREVArticleNEXTCorresponding States")
    assert drop.search("Get PDF Email Share Share with Facebook Tweet This Post on reddit")
    assert drop.search("A letter report issued by the General Accounting Office with an abstract")
    assert drop.search("This is a PSYNDEX Tests Review of BMP - BREMER MOSAIKPROBE.")
    assert drop.search("Li3V5O12 crystallizes in the monoclinic C2 space group. The structure is three-dimensional.")
    assert drop.search("Q8-CLUSTER-162: E8 Term: covid — E8 Intelligence Research\nQ8 Compression Breakthrough")
    assert drop.search("S. Kidman and Co Ltd - The Guide to Australian Business Records is a biographical, "
                       "bibliographical and archival database of Australian business entities")
    assert drop.search("Advertisement Search Menu Menu Navbar Search Filter This issueAll Social Work Research")
    assert drop.search("Poland\nIan F. D. Morrow; Poland: Eagle in the East, International Affairs, Volume 46, "
                       "Issue 1, 1 January 1970, Pages 170–171, https://doi.org/10.2307/2613874")
    assert not drop.search("Kinetic model\nWe present a model [1]. See Nature, Volume 12, Issue 3, 1 March 2001, "
                           "Pages 1-20. " + "We discuss the coupling at the inner boundary of the magnetosphere. " * 6)
    assert drop.search("Auditor of State David A. Vaudt today released an audit report on the City of Elkhart, Iowa.")
    assert not drop.search("We discuss Kant's critique of pure reason and its reception.")


def test_snapshot_year(tmp_path):
    assert openalex.snapshot_year(tmp_path) is None
    (tmp_path / "manifest.json").write_text(json.dumps({"date": "2026-10-14", "files": []}))
    assert openalex.snapshot_year(tmp_path) == 2026
