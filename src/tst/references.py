"""Literature behind the method choices. Parameters in the bundle cite these by id (manifest.json:
run.parameters[].parameters[].references), and the bundle lists the cited ones in run.references."""

from __future__ import annotations

import re

# id -> authors, year, title, venue, url (None where no stable link exists). Checked against Crossref, doi.org, the
# ACL Anthology, arXiv and library catalogues on 2026-11-05; page1999 only against mirrors (Stanford's server was
# down that day). lee2015 to eppstein2004 checked against Crossref on 2026-11-07.
REFERENCES: dict[str, dict] = {
    "openalex": {
        "authors": "Priem, J., Piwowar, H. & Orr, R.", "year": 2022,
        "title": "OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts",
        "venue": "arXiv preprint arXiv:2205.01833", "url": "https://doi.org/10.48550/arXiv.2205.01833"},
    "harman1991": {
        "authors": "Harman, D.", "year": 1991, "title": "How effective is suffixing?",
        "venue": "Journal of the American Society for Information Science 42(1), 7–15",
        "url": "https://doi.org/10.1002/(SICI)1097-4571(199101)42:1%3C7::AID-ASI2%3E3.0.CO;2-P"},
    "schofield2016": {
        "authors": "Schofield, A. & Mimno, D.", "year": 2016,
        "title": "Comparing Apples to Apple: The Effects of Stemmers on Topic Models",
        "venue": "Transactions of the Association for Computational Linguistics 4, 287–300",
        "url": "https://doi.org/10.1162/tacl_a_00099"},
    "schofield2017": {
        "authors": "Schofield, A., Magnusson, M. & Mimno, D.", "year": 2017,
        "title": "Pulling Out the Stops: Rethinking Stopword Removal for Topic Models",
        "venue": "Proceedings of the 15th Conference of the European Chapter of the Association for Computational "
                 "Linguistics (EACL 2017), Volume 2: Short Papers, 432–436",
        "url": "https://doi.org/10.18653/v1/E17-2069"},
    "gerlach2019": {
        "authors": "Gerlach, M., Shi, H. & Amaral, L. A. N.", "year": 2019,
        "title": "A universal information theoretic approach to the identification of stopwords",
        "venue": "Nature Machine Intelligence 1(12), 606–612", "url": "https://doi.org/10.1038/s42256-019-0112-6"},
    "sarica2021": {
        "authors": "Sarica, S. & Luo, J.", "year": 2021, "title": "Stopwords in technical language processing",
        "venue": "PLOS ONE 16(8), e0254937", "url": "https://doi.org/10.1371/journal.pone.0254937"},
    "salton1988": {
        "authors": "Salton, G. & Buckley, C.", "year": 1988,
        "title": "Term-weighting approaches in automatic text retrieval",
        "venue": "Information Processing & Management 24(5), 513–523",
        "url": "https://doi.org/10.1016/0306-4573(88)90021-0"},
    "lee1999": {
        "authors": "Lee, D. D. & Seung, H. S.", "year": 1999,
        "title": "Learning the parts of objects by non-negative matrix factorization",
        "venue": "Nature 401(6755), 788–791", "url": "https://doi.org/10.1038/44565"},
    "boutsidis2008": {
        "authors": "Boutsidis, C. & Gallopoulos, E.", "year": 2008,
        "title": "SVD based initialization: A head start for nonnegative matrix factorization",
        "venue": "Pattern Recognition 41(4), 1350–1362", "url": "https://doi.org/10.1016/j.patcog.2007.09.010"},
    "ocallaghan2015": {
        "authors": "O'Callaghan, D., Greene, D., Carthy, J. & Cunningham, P.", "year": 2015,
        "title": "An analysis of the coherence of descriptors in topic modeling",
        "venue": "Expert Systems with Applications 42(13), 5645–5657",
        "url": "https://doi.org/10.1016/j.eswa.2015.02.055"},
    "greene2014": {
        "authors": "Greene, D., O'Callaghan, D. & Cunningham, P.", "year": 2014,
        "title": "How Many Topics? Stability Analysis for Topic Models",
        "venue": "ECML PKDD 2014, Lecture Notes in Computer Science 8724, 498–513",
        "url": "https://doi.org/10.1007/978-3-662-44848-9_32"},
    "lau2014": {
        "authors": "Lau, J. H., Newman, D. & Baldwin, T.", "year": 2014,
        "title": "Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality",
        "venue": "Proceedings of the 14th Conference of the European Chapter of the Association for Computational "
                 "Linguistics (EACL 2014), 530–539",
        "url": "https://doi.org/10.3115/v1/E14-1056"},
    "bouma2009": {
        "authors": "Bouma, G.", "year": 2009,
        "title": "Normalized (Pointwise) Mutual Information in Collocation Extraction",
        "venue": "From Form to Meaning: Processing Texts Automatically. Proceedings of the Biennial GSCL Conference "
                 "2009, Narr, Tübingen, 31–40", "url": None},
    "dieng2020": {
        "authors": "Dieng, A. B., Ruiz, F. J. R. & Blei, D. M.", "year": 2020,
        "title": "Topic Modeling in Embedding Spaces",
        "venue": "Transactions of the Association for Computational Linguistics 8, 439–453",
        "url": "https://doi.org/10.1162/tacl_a_00325"},
    "lawson1974": {
        "authors": "Lawson, C. L. & Hanson, R. J.", "year": 1974, "title": "Solving Least Squares Problems",
        "venue": "Prentice-Hall, Englewood Cliffs, NJ", "url": None},
    "schaefermeier2021": {
        "authors": "Schaefermeier, B., Stumme, G. & Hanika, T.", "year": 2021,
        "title": "Topic space trajectories: A case study on machine learning literature",
        "venue": "Scientometrics 126(7), 5759–5795", "url": "https://doi.org/10.1007/s11192-021-03931-0"},
    "dangelo2015": {
        "authors": "D'Angelo, C. A. & Abramo, G.", "year": 2015,
        "title": "Publication Rates in 192 Research Fields of the Hard Sciences",
        "venue": "Proceedings of ISSI 2015, 15th International Society of Scientometrics and Informetrics "
                 "Conference, Istanbul, 909–919",
        "url": "https://www.issi-society.org/proceedings/issi_2015/0909.pdf"},
    "piro2013": {
        "authors": "Piro, F. N., Aksnes, D. W. & Rørstad, K.", "year": 2013,
        "title": "A macro analysis of productivity differences across fields: Challenges in the measurement of "
                 "scientific publishing",
        "venue": "Journal of the American Society for Information Science and Technology 64(2), 307–320",
        "url": "https://doi.org/10.1002/asi.22746"},
    "mcinnes2018": {
        "authors": "McInnes, L., Healy, J. & Melville, J.", "year": 2018,
        "title": "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction",
        "venue": "arXiv preprint arXiv:1802.03426", "url": "https://doi.org/10.48550/arXiv.1802.03426"},
    "traag2019": {
        "authors": "Traag, V. A., Waltman, L. & van Eck, N. J.", "year": 2019,
        "title": "From Louvain to Leiden: guaranteeing well-connected communities",
        "venue": "Scientific Reports 9, article 5233", "url": "https://doi.org/10.1038/s41598-019-41695-z"},
    "lee2015": {
        "authors": "Lee, Y.-N., Walsh, J. P. & Wang, J.", "year": 2015,
        "title": "Creativity in scientific teams: Unpacking novelty and impact",
        "venue": "Research Policy 44(3), 684–697", "url": "https://doi.org/10.1016/j.respol.2014.10.007"},
    "uzzi2013": {
        "authors": "Uzzi, B., Mukherjee, S., Stringer, M. & Jones, B.", "year": 2013,
        "title": "Atypical Combinations and Scientific Impact",
        "venue": "Science 342(6157), 468–472", "url": "https://doi.org/10.1126/science.1240474"},
    "wang2017": {
        "authors": "Wang, J., Veugelers, R. & Stephan, P.", "year": 2017,
        "title": "Bias against novelty in science: A cautionary tale for users of bibliometric indicators",
        "venue": "Research Policy 46(8), 1416–1436", "url": "https://doi.org/10.1016/j.respol.2017.06.006"},
    "waltman2016": {
        "authors": "Waltman, L.", "year": 2016, "title": "A review of the literature on citation impact indicators",
        "venue": "Journal of Informetrics 10(2), 365–391", "url": "https://doi.org/10.1016/j.joi.2016.02.007"},
    "hill1973": {
        "authors": "Hill, M. O.", "year": 1973, "title": "Diversity and Evenness: A Unifying Notation and Its Consequences",
        "venue": "Ecology 54(2), 427–432", "url": "https://doi.org/10.2307/1934352"},
    "jost2006": {
        "authors": "Jost, L.", "year": 2006, "title": "Entropy and diversity",
        "venue": "Oikos 113(2), 363–375", "url": "https://doi.org/10.1111/j.2006.0030-1299.14714.x"},
    "eppstein2004": {
        "authors": "Eppstein, D. & Wang, J.", "year": 2004, "title": "Fast Approximation of Centrality",
        "venue": "Journal of Graph Algorithms and Applications 8(1), 39–45", "url": "https://doi.org/10.7155/jgaa.00081"},
    "page1999": {
        "authors": "Page, L., Brin, S., Motwani, R. & Winograd, T.", "year": 1999,
        "title": "The PageRank Citation Ranking: Bringing Order to the Web",
        "venue": "Technical Report 1999-66, Stanford InfoLab", "url": "http://ilpubs.stanford.edu:8090/422/"},
}


def short_citation(ref_id: str) -> str:
    """"Schofield & Mimno 2016", "Greene et al. 2014"."""
    r = REFERENCES[ref_id]
    names = [a.split(",")[0].strip() for a in re.split(r"\.,\s+|\s+&\s+", r["authors"])]
    lead =names[0] if len(names) == 1 else f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} et al."
    return f"{lead} {r['year']}"


def bibliography(ids) -> list[dict]:
    """The cited references in order of first citation: [{id, citation, authors, year, title, venue, url}]."""
    out, seen = [], set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append({"id": i, "citation": short_citation(i), **REFERENCES[i]})
    return out
