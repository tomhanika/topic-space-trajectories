"""Vocabulary curation before TF-IDF: merging variants of a word and removing the generic words of academic prose.

Variants. Topic word lists showed singular and plural side by side ("model, models", "genetic, genetics") and
British next to American spelling ("behaviour, behavior"). Merging is deliberately light: plural and -s forms
(an S-stemmer after Harman 1991, plus common Latin and Greek plurals) and British/American spellings, and only
when both forms occur in the corpus. Aggressive stemmers such as Porter's conflate unrelated words
("university", "universe") and do not improve topic models (Schofield & Mimno, TACL 2016). Each group is named
by its most frequent member, so topic words stay real words.

Stop words. Words such as "study", "results" or "propose" are frequent in every field and appear in many topics
without describing any of them. academic_stopwords.yaml lists them by category, with how the list was built and
the candidates left off on purpose; the bundle shows both (methods.json). A stop word removes its whole variant
group ("study" also removes "studies").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import cache
from importlib.resources import files

import yaml

# Latin and Greek plurals that the suffix rules miss or would map to a wrong word ("analyses" -> "analyse").
# Pairs whose meanings drifted apart are left out on purpose: media/medium, data/datum, bases (base or basis).
IRREGULAR_PLURALS = {
    "analyses": "analysis", "hypotheses": "hypothesis", "theses": "thesis", "crises": "crisis",
    "diagnoses": "diagnosis", "syntheses": "synthesis", "prognoses": "prognosis", "metastases": "metastasis",
    "emphases": "emphasis", "parentheses": "parenthesis", "psychoses": "psychosis", "neuroses": "neurosis",
    "stenoses": "stenosis", "thromboses": "thrombosis", "axes": "axis", "ellipses": "ellipsis",
    "criteria": "criterion", "phenomena": "phenomenon",
    "spectra": "spectrum", "bacteria": "bacterium", "curricula": "curriculum", "strata": "stratum",
    "maxima": "maximum", "minima": "minimum", "optima": "optimum", "momenta": "momentum",
    "equilibria": "equilibrium", "consortia": "consortium", "symposia": "symposium", "quanta": "quantum",
    "nuclei": "nucleus", "stimuli": "stimulus", "fungi": "fungus", "loci": "locus", "radii": "radius",
    "foci": "focus", "bacilli": "bacillus", "alumni": "alumnus", "calculi": "calculus",
    "formulae": "formula", "larvae": "larva", "algae": "alga", "vertebrae": "vertebra", "antennae": "antenna",
    "nebulae": "nebula", "matrices": "matrix", "indices": "index", "vertices": "vertex", "apices": "apex",
    "cortices": "cortex", "appendices": "appendix", "genera": "genus", "corpora": "corpus",
    "women": "woman", "men": "man", "children": "child", "feet": "foot", "teeth": "tooth", "mice": "mouse",
}
# Words ending in -s that are not the plural of the word without it ("news" is not "new", "aids" not "aid").
# Singular nouns ending in -as/-os/-ns are listed too where the word without -s exists ("bias" is not "bia").
NO_PLURAL = frozenset(
    "news means goods arms customs odds aids humanities logistics chaos caries species series logos mars rights "
    "leads menses mens bias alias atlas canvas lens pancreas ethos pathos cosmos kudos thermos".split())
_ES_STEM = ("s", "x", "z", "ch", "sh", "o")  # -es is a plural ending only after these: processes, boxes, heroes


def plural_candidates(w: str) -> list[str]:
    """Possible singulars of w; the most frequent one that occurs in the corpus is used."""
    if w in IRREGULAR_PLURALS:
        return [IRREGULAR_PLURALS[w]]
    if len(w) < 4 or not w.endswith("s") or w.endswith(("ss", "us", "is")) or w in NO_PLURAL:
        return []
    out = [w[:-1]]  # models -> model, genetics -> genetic, phases -> phase
    if w.endswith("ies") and w[-4] not in "ae":
        out.append(w[:-3] + "y")  # studies -> study
    if w.endswith("es") and w[:-2].endswith(_ES_STEM):
        out.append(w[:-2])  # processes -> process, approaches -> approach, gases -> gas, heroes -> hero
    if w.endswith("ves"):
        out += [w[:-3] + "f", w[:-3] + "fe"]  # leaves -> leaf, lives -> life
    # Never a form that is itself a plural: "mens" is not the plural of "men" (that chain ends at "man").
    return [c for c in out if c not in IRREGULAR_PLURALS]


# British -> American spelling. Suffix rules; the result is only used if it occurs in the corpus.
_SPELLING_RULES = [
    (re.compile(r"(?<=[a-z]{2})is(e|ed|es|er|ers|ing|ation|ations|ational)$"), r"iz\1"),  # organise, organisation
    (re.compile(r"(?<=[a-z])ys(e|ed|es|er|ers|ing)$"), r"yz\1"),  # analyse, catalysed
    (re.compile(r"(?<=[a-z]{2})our(s|ed|ing|er|ers|able|ably|ite|ites|al|ally|ism|ist|ists|less|ful|hood|hoods)?$"),
     r"or\1"),  # behaviour(al), colour, tumour, neighbourhood
    (re.compile(r"(?<=[a-z]{2})tr(e|es|ed|ing)$"), r"ter\1"),  # centre(s), fibre, metre, centred -> centered
    (re.compile(r"(?<=[a-z]{2})ogue(s)?$"), r"og\1"),  # catalogue, dialogue, analogue
    (re.compile(r"(?<=[a-z])(el|al)l(ing|ings|ed|er|ers|or|ors)$"), r"\1\2"),  # modelling, signalling, travelled
]
# British infixes of medical and scientific words (haemoglobin, anaemia, paediatric, oestrogen, palaeo-).
_SPELLING_INFIX = [("haem", "hem"), ("aemi", "emi"), ("paed", "ped"), ("aetiol", "etiol"), ("anaesth", "anesth"),
                   ("gynaec", "gynec"), ("faec", "fec"), ("oedem", "edem"), ("oestr", "estr"), ("foet", "fet"),
                   ("oesoph", "esoph"), ("diarrhoe", "diarrhe"), ("coeliac", "celiac"), ("amoeb", "ameb"),
                   ("dyspnoe", "dyspne"), ("apnoe", "apne"), ("homoeo", "homeo"), ("palaeo", "paleo"),
                   ("archaeo", "archeo"), ("mediaev", "mediev"), ("caesarean", "cesarean"), ("sulph", "sulf")]
_SPELLING_WORDS = {
    "aluminium": "aluminum", "programme": "program", "ageing": "aging", "judgement": "judgment",
    "acknowledgement": "acknowledgment", "grey": "gray", "plough": "plow", "mould": "mold", "moulding": "molding",
    "moulded": "molded", "enrolment": "enrollment", "fulfil": "fulfill", "fulfilment": "fulfillment",
    "skilful": "skillful", "wilful": "willful", "instalment": "installment", "manoeuvre": "maneuver",
    "manoeuvring": "maneuvering", "manoeuvrability": "maneuverability", "aeroplane": "airplane",
    "artefact": "artifact", "sceptic": "skeptic", "sceptical": "skeptical", "scepticism": "skepticism",
    "defence": "defense", "licence": "license", "offence": "offense", "pretence": "pretense",
    "practise": "practice", "practised": "practiced", "practising": "practicing", "jewellery": "jewelry",
}


def spelling_candidates(w: str) -> list[str]:
    out = [_SPELLING_WORDS[w]] if w in _SPELLING_WORDS else []
    out += [r.sub(s, w) for r, s in _SPELLING_RULES if r.search(w)]
    out += [w.replace(a, b) for a, b in _SPELLING_INFIX if a in w]
    return [c for c in out if c != w]


def _best_known(cands: list[str], known: Mapping[str, int]) -> str | None:
    found = [c for c in cands if c in known]
    return max(found, key=lambda c: known[c]) if found else None


def variant_key(w: str, known: Mapping[str, int], methods: Iterable[str]) -> str:
    """The singular, American form of w, as far as these forms occur in `known` (word -> document frequency)."""
    methods = set(methods)
    for _ in range(4):  # until stable; "colours" -> "colour" -> "color"
        prev = w
        if "plural" in methods:
            w = _best_known(plural_candidates(w), known) or w
        if "spelling" in methods:
            cands = spelling_candidates(w)
            if "plural" in methods:  # "colours" -> "color" when neither "colour" nor "colors" occurs
                cands += [p for c in cands for p in plural_candidates(c)]
                cands += [s for c in plural_candidates(w) for s in spelling_candidates(c)]
            w = _best_known(cands, known) or w
        if w == prev:
            break
    return w


def build_variants(df: Mapping[str, int], methods: Iterable[str]) -> dict[str, str]:
    """Map each word to the most frequent member of its variant group; words that stay unchanged are left out."""
    methods = list(methods)
    if not methods:
        return {}
    groups: dict[str, list[str]] = {}
    for w in df:
        groups.setdefault(variant_key(w, df, methods), []).append(w)
    out = {}
    for key, members in groups.items():
        if len(members) == 1 and members[0] == key:
            continue
        members = sorted(set(members) | {key})
        name = max(members, key=lambda m: (df.get(m, 0), -len(m), m))
        out.update((m, name) for m in members if m != name)
    return out


@cache
def academic_stopword_list() -> dict:
    """academic_stopwords.yaml: {date, method, references, categories: [{name, words}], not_stopwords: [{words,
    reason}]}, with the word strings split into lists."""
    data = yaml.safe_load(files("tst.topics").joinpath("academic_stopwords.yaml").read_text(encoding="utf-8"))
    for entry in data["categories"] + data["not_stopwords"]:
        entry["words"] = entry["words"].split()
    data["date"] = str(data["date"])
    return data


def academic_stopwords() -> frozenset[str]:
    return frozenset(w for c in academic_stopword_list()["categories"] for w in c["words"])


def stopword_names(stopwords: Iterable[str], variants: Mapping[str, str]) -> set[str]:
    """Group names (vocabulary terms) removed by these stop words."""
    return {variants.get(w, w) for w in stopwords}
