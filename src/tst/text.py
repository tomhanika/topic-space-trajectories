"""Tokenisation. The same Tokenizer is stored in the model bundle and used for training and embedding."""

from __future__ import annotations

import html
import logging
import multiprocessing as mp
import re
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import PreprocessConfig
from .run import limited_threads

log = logging.getLogger("tst.text")

_LATEX_MATH = re.compile(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.S)
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(\{[^}]*\})?")
# Tags only (<sup>, </jats:p>, <mml:math ...>); "a < b and c > d" is no tag.
_HTML = re.compile(r"</?[a-z][a-z0-9:_-]{0,30}(?:\s[^<>]{0,300})?/?>")
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
# Formatting codes of records converted from old databases (OSTI/INIS energy and physics abstracts, IEEE Xplore
# until about 2006), in titles and abstracts: "He/sup 4/", "E/sub p/", "/spl mu/m", "/sub 7-/spl delta//",
# "k{sup 2}", "e{sup {minus}}", "p{yields}n", "/times/". 3 % of physics abstracts; they formed a topic
# "sup, minus, yields, spl". Applied after lower-casing.
_DB_SYMBOLS = ("yields|minus|plus|plus_minus|minus_plus|times|approx|similar|center_dot|open_quotes|close_quotes|"
               "degree|deg|infinity|prime|double_prime|sim|equivalent|proportional|dagger|bar|tilde|hat|vert_bar|"
               "radical|integral|partial|nabla|sqrt|r arrow|l arrow|lr arrow|alpha|beta|gamma|delta|epsilon|zeta|"
               "eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega")
_DB_CODES = re.compile(
    rf"/(?:sup|sub|spl|ital|bar)\s(?:[^/\n]|/(?:spl\s[a-z]+|{_DB_SYMBOLS})/){{0,30}}/"
    rf"|/(?:{_DB_SYMBOLS})/"
    rf"|\{{(?:sup|sub|ital|bold|bar)\s(?:[^{{}}\n]|\{{[^{{}}\n]{{0,25}}\}}){{0,40}}\}}"
    rf"|\{{(?:{_DB_SYMBOLS})\}}")
_WORD = re.compile(r"[^\W\d_]+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\s*\n\s*")
# Prefixes that are no words of their own; "non-linear" and "nonlinear" become one token. "self", "well" and
# chemical locants (d-, l-, n-, p-) stay apart, since joining them would split "glucose" from "d-glucose".
HYPHEN_PREFIXES = ("non anti multi semi pre post sub inter intra micro nano macro mono poly hyper hypo ultra "
                   "pseudo quasi co re bi tri neo trans meta counter over under e x").split()
_PREFIX_HYPHEN = re.compile(r"\b(" + "|".join(HYPHEN_PREFIXES) + r")-(?=[^\W\d_]{2})")
# Phrases written as one word, so that they are not lost as stop words ("well-being") or leave a misleading word
# behind ("state-of-the-art" -> "art"; "stateoftheart" is an academic stop word) or merge two subjects ("science
# fiction" with "science").
JOINED_PHRASES = {"wellbeing": r"well[- ]?being", "stateoftheart": r"state[- ]of[- ]the[- ]art",
                  "sciencefiction": r"science[- ]fiction"}
# (a word the text must contain, pattern, joined word); the check skips the regex for most texts.
_PHRASES = [(re.findall(r"[a-z]+", p)[-1], re.compile(rf"\b{p}\b"), w) for w, p in JOINED_PHRASES.items()]
# Apostrophes: English endings are removed ("Kant's" -> "kant", "don't" -> "do"), other apostrophes inside a
# word are dropped ("da'wah" -> "dawah", "Qur'an" -> "quran", "O'Brien" -> "obrien"), so that no fragments
# ("wah", "qur") remain. U+02BB/U+02BC are letters to the regex engine, so they are listed explicitly.
_APOS = "'\u2019\u02bc\u02bb`\u00b4"
# The patterns start with the apostrophe (fast to find) and look back for the letter before it.
_ENDING = re.compile(rf"[{_APOS}](?<=[^\W\d_][{_APOS}])(?:s|re|ve|ll)(?![^\W\d_])|"
                     rf"n(?<=[^\W\d_]n)[{_APOS}]t(?![^\W\d_])")
_INNER_APOS = re.compile(rf"[{_APOS}](?<=[^\W\d_][{_APOS}])(?=[^\W\d_])")
_ANY_APOS = re.compile(rf"[{_APOS}]")
# Letters of the Latin alphabet with diacritics (Latin-1, Extended-A/B, IPA, Extended Additional).
_LATIN = re.compile(r"[a-z\u00df-\u00f6\u00f8-\u024f\u0250-\u02af\u1e00-\u1eff]+")
# Letters that Unicode does not decompose into a base letter plus an accent.
_FOLD = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th",
                       "ı": "i", "ħ": "h", "ŋ": "n"})
_COMBINING = re.compile(r"[\u0300-\u036f]+")


def fold_diacritics(text: str) -> str:
    """"Schrödinger" -> "Schrodinger", "façade" -> "facade", "straße" -> "strasse"; other scripts keep their
    letters (without accents)."""
    if text.isascii():
        return text
    return _COMBINING.sub("", unicodedata.normalize("NFKD", text.translate(_FOLD)))


def _fold_words(words: list[str]) -> list[str]:
    return [w if w.isascii() else fold_diacritics(w) for w in words]


# Frequent function words (>= 3 letters) of languages common in bibliographic databases, for the language vote.
FOREIGN_STOPWORDS = {
    "de": "der die und das den dem des ein eine einer eines ist nicht sich mit auf für von auch als wird "
          "werden sind wie bei aus nach über durch oder dass zur zum einem",
    "fr": "les des une est dans pour que qui sur par pas avec ont sont aux ces cette plus mais leur elle "
          "nous vous être été entre comme",
    "es": "los las del una por para con que como más pero sus este esta entre sobre también fue son ser "
          "han muy desde",
    "pt": "dos das uma para com não por mais como sobre são pelo pela entre também seu sua foi ser este "
          "esta nos",
    "it": "della delle degli che non per una con sono come anche alla dalla nel nella questo questa gli "
          "dei del",
    "nl": "het een van zijn niet met voor ook als dat aan bij door wordt worden deze maar naar",
    "id": "yang dan dari dalam untuk dengan ini itu pada adalah tidak oleh akan juga sebagai atau terhadap "
          "secara melalui antara serta bagi kepada tentang tersebut dapat menggunakan",
}


# Some databases strip accents ("nao", "tambem"), so both spellings are listed.
_FOREIGN = {lang: frozenset(ws.split()) | frozenset(fold_diacritics(w) for w in ws.split())
            for lang, ws in FOREIGN_STOPWORDS.items()}
# English counterpart of similar size for a fair vote (the full English list has ~300 words).
ENGLISH_VOTE_WORDS = frozenset(
    "the and that for with this are from was were which have has not but its their these been also into "
    "than such can more other between".split())
# Foreign function words removed as stop words, so that foreign titles cannot form "language topics".
# Words that are also English or philosophical terms stay.
_AMBIGUOUS = frozenset("die son con nous per art fue han dan nos este".split())
FOREIGN_TOKEN_STOPWORDS = frozenset().union(*_FOREIGN.values()) - _AMBIGUOUS
# Not counted in the title vote: name particles and words of English titles ("van der Waals", "von Neumann",
# "Yang-Mills", "Los Alamos", "non-", "plus", "MIT", "DoS", "OOK", "UMa", ".com", "Les Houches").
TITLE_AMBIGUOUS = frozenset("van von der den yang non plus mit door ook dos los uma com par les della".split())
_TITLE_FOREIGN = {lang: ws - _AMBIGUOUS - TITLE_AMBIGUOUS for lang, ws in _FOREIGN.items()}


@dataclass
class Tokenizer:
    kind: str = "default"
    min_len: int = 3
    max_len: int = 30
    extra_stopwords: list[str] = field(default_factory=list)
    foreign_stopwords: bool = False
    join_prefixes: bool = False
    # The defaults of the following keep the behaviour of model bundles written before they existed.
    join_apostrophes: bool = False
    fold_accents: bool = False
    latin_only: bool = False
    skip_foreign_titles: bool = False

    def __post_init__(self):
        self.stopwords = frozenset(ENGLISH_STOP_WORDS) | frozenset(w.lower() for w in self.extra_stopwords)
        if self.foreign_stopwords:
            self.stopwords |= FOREIGN_TOKEN_STOPWORDS
        self._english = frozenset(ENGLISH_STOP_WORDS)

    @classmethod
    def from_config(cls, cfg: PreprocessConfig) -> Tokenizer:
        lang = cfg.language_filter
        return cls("default", cfg.min_token_len, cfg.max_token_len, list(cfg.extra_stopwords),
                   cfg.foreign_stopwords, cfg.join_prefixes, cfg.join_apostrophes, cfg.fold_accents, cfg.latin_only,
                   lang.method == "stopword_vote" and lang.ignore_foreign_titles)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "min_len": self.min_len, "max_len": self.max_len,
                "extra_stopwords": self.extra_stopwords, "foreign_stopwords": self.foreign_stopwords,
                "join_prefixes": self.join_prefixes, "join_apostrophes": self.join_apostrophes,
                "fold_accents": self.fold_accents, "latin_only": self.latin_only,
                "skip_foreign_titles": self.skip_foreign_titles}

    def _scan(self, text: str) -> tuple[list[str], float]:
        """All word tokens including stop words, and the share of letters in words of other scripts than Latin."""
        if "&" in text:
            text = html.unescape(text)  # "&lt;sup&gt;" is a tag, "&amp;" no word
        text = unicodedata.normalize("NFKC", text).lower()
        if not text.isascii():
            # Hyphens that NFKC leaves alone; the soft hyphen marks a possible line break inside a word; "İ".lower()
            # is "i" plus a combining dot, which would split the word.
            text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u00ad", "").replace("\u0307", "")
        if "/" in text or "{" in text:
            text = _DB_CODES.sub(" ", text)
        text = _URL.sub(" ", _HTML.sub(" ", _LATEX_CMD.sub(" ", _LATEX_MATH.sub(" ", text))))
        if self.join_prefixes:
            text = _PREFIX_HYPHEN.sub(r"\1", text)
            for check, pattern, word in _PHRASES:
                if check in text:
                    text = pattern.sub(word, text)
        if self.join_apostrophes and _ANY_APOS.search(text):
            text = _INNER_APOS.sub("", _ENDING.sub("", text))
        words = _WORD.findall(text)
        if self.fold_accents:
            words = _fold_words(words)
        other = 0.0
        if words:
            latin = [w for w in words if w.isascii() or _LATIN.fullmatch(w)]
            if len(latin) < len(words):
                other = 1.0 - sum(map(len, latin)) / sum(map(len, words))
                if self.latin_only:
                    words = latin
        return [w for w in words if self.min_len <= len(w) <= self.max_len], other

    def words(self, text: str) -> list[str]:
        """All word tokens including stop words (needed for the language heuristic)."""
        return self._scan(text)[0]

    def other_script(self, text: str) -> float:
        """The share of letters in words of other scripts than Latin."""
        return self._scan(text)[1]

    def __call__(self, text: str) -> list[str]:
        return [w for w in self.words(text) if w not in self.stopwords]

    def analyze(self, text: str) -> tuple[list[str], float, float, float, float]:
        """Tokens without stop words, the share of English stop words among all words, for the language vote
        the shares of frequent English and of the most frequent other language's function words (comparable
        lists, see FOREIGN_STOPWORDS), and the share of letters in other scripts than Latin."""
        words, other = self._scan(text)
        if not words:
            return [], 0.0, 0.0, 0.0, other
        n = len(words)
        n_stop = sum(w in self._english for w in words)
        en_vote = sum(w in ENGLISH_VOTE_WORDS for w in words)
        foreign = max(sum(w in sw for w in words) for sw in _FOREIGN.values())
        return [w for w in words if w not in self.stopwords], n_stop / n, en_vote / n, foreign / n, other

    def foreign_title(self, title: str) -> bool:
        """True if the title has more function words of one other language than English ones (TITLE_AMBIGUOUS
        not counted). Titles are short, so already one function word decides when no English one is there."""
        words = self.words(title)
        en = sum(w in ENGLISH_VOTE_WORDS for w in words)
        return max(sum(w in sw for w in words) for sw in _TITLE_FOREIGN.values()) > en

    def ignores_title(self, title: str | None, abstract: str | None, fields) -> bool:
        """With skip_foreign_titles, a title in another language over an abstract is left out: an Indonesian or
        Spanish title over an English abstract would put its words into the topics."""
        return bool(self.skip_foreign_titles and title and abstract and "title" in fields and "abstract" in fields
                    and self.foreign_title(title))

    def english_part(self, text: str) -> str:
        """The sentences of `text` without more foreign than English function words: drops the
        non-English half of a bilingual abstract."""
        keep = []
        for sentence in _SENTENCE.split(text):
            words = self.words(sentence)
            en = sum(w in ENGLISH_VOTE_WORDS for w in words)
            if max(sum(w in sw for w in words) for sw in _FOREIGN.values()) <= en:
                keep.append(sentence)
        return " ".join(keep)


def document_text(title: str | None, abstract: str | None, fields: list[str]) -> str:
    parts = {"title": title or "", "abstract": abstract or ""}
    return " ".join(parts[f] for f in fields).strip()


# Spawned workers read their own slices of documents.parquet; the parent only writes results.
_TOK: Tokenizer | None = None
_CFG: PreprocessConfig | None = None
_SKIP: dict[int, str] = {}
_STRIP: re.Pattern | None = None


def _init_worker(tok: dict, cfg: dict, skip_path: Path) -> None:
    # Small initargs only: large ones make process start-up serial and slow on many-core machines.
    global _TOK, _CFG, _SKIP, _STRIP
    _TOK, _CFG = Tokenizer(**tok), PreprocessConfig.model_validate(cfg)
    _STRIP = re.compile("|".join(f"(?:{p})" for p in _CFG.strip_patterns), re.I) if _CFG.strip_patterns else None
    skip = pl.read_parquet(skip_path)
    _SKIP = dict(zip(skip["doc_id"].to_list(), skip["reason"].to_list()))


def _work(task: tuple[Path, int, int, int]) -> pa.Table:
    path, group, part, n_parts = task
    t = pq.ParquetFile(path).read_row_group(group, columns=["doc_id", "title", "abstract"])
    step = -(-t.num_rows // n_parts)
    t = t.slice(part * step, step)
    tok, cfg = _TOK, _CFG
    ids, titles, abstracts = (t.column(c).to_pylist() for c in ("doc_id", "title", "abstract"))
    lang = cfg.language_filter
    out_tokens, n_tokens, shares, reasons, ignored = [], [], [], [], []
    for doc_id, title, abstract in zip(ids, titles, abstracts):
        skip = _SKIP.get(doc_id) or ("no_abstract" if cfg.require_abstract and not abstract else None)
        if skip:
            out_tokens.append([]), n_tokens.append(0), shares.append(0.0), reasons.append(skip), ignored.append(False)
            continue
        if _STRIP is not None:
            abstract = _STRIP.sub(" ", abstract)
        title_ignored = tok.ignores_title(title, abstract, cfg.fields)
        text = document_text(None if title_ignored else title, abstract, cfg.fields)
        tokens, share, en_vote, foreign, other = tok.analyze(text)
        if other and abstract and "abstract" in cfg.fields:
            # The abstract decides: a Japanese abstract with an English title is no English paper, and Japanese
            # needs fewer letters than English for the same content.
            other = tok.other_script(abstract)
        if lang.method == "stopword_vote" and lang.sentence_filter and foreign >= lang.sentence_filter:
            text = tok.english_part(text)
            tokens, share, en_vote, foreign, _ = tok.analyze(text) if text else ([], 0.0, 0.0, 1.0, 0.0)
        reason = "ok"
        if not text or (lang.method != "none" and other > lang.max_other_script):
            reason = "language"
        elif len(tokens) < cfg.min_tokens:
            reason = "too_short"
        elif lang.method != "none" and share <= lang.threshold:
            reason = "language"
        elif lang.method == "stopword_vote" and foreign > en_vote:
            reason = "language"
        out_tokens.append(tokens), n_tokens.append(len(tokens)), shares.append(share), reasons.append(reason)
        ignored.append(title_ignored)
    return pa.table({"doc_id": ids, "tokens": out_tokens, "n_tokens": n_tokens, "stopword_share": shares,
                     "reason": reasons, "title_ignored": ignored}, schema=_TOKENS_SCHEMA)


_TOKENS_SCHEMA = pa.schema([("doc_id", pa.int64()), ("tokens", pa.list_(pa.string())),
                            ("n_tokens", pa.int32()), ("stopword_share", pa.float32()), ("reason", pa.string()),
                            ("title_ignored", pa.bool_())])


def corpus_filters(documents: Path, cfg: PreprocessConfig) -> dict[int, str]:
    """Documents to drop before tokenising: spam patterns, boilerplate and duplicate abstracts."""
    df = pl.read_parquet(documents, columns=["doc_id", "title", "abstract"]).filter(pl.col("abstract").is_not_null())
    skip: dict[int, str] = {}
    if drop := cfg.all_drop_patterns():
        # Patterns see the title and the abstract; "^" anchors at the start of the title.
        pattern = "(?i)" + "|".join(f"(?:{p})" for p in drop)
        text = pl.concat_str([pl.col("title").fill_null(""), pl.col("abstract")], separator="\n")
        skip.update((i, "spam") for i in df.filter(text.str.contains(pattern))["doc_id"].to_list())
    df = df.drop("title")
    norm = df.with_columns(key=pl.col("abstract").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars())
    norm = norm.with_columns(n=pl.len().over("key"), first=pl.col("doc_id").min().over("key"))
    if cfg.boilerplate_min_count:
        ids = norm.filter(pl.col("n") >= cfg.boilerplate_min_count)["doc_id"].to_list()
        skip.update((i, "boilerplate") for i in ids if i not in skip)
    if cfg.dedupe:
        ids = norm.filter((pl.col("n") > 1) & (pl.col("doc_id") != pl.col("first")))["doc_id"].to_list()
        skip.update((i, "duplicate") for i in ids if i not in skip)
    return skip


def preprocess(documents: Path, out: Path, cfg: PreprocessConfig, tokenizer: Tokenizer, n_jobs: int) -> dict:
    skip = corpus_filters(documents, cfg)
    groups = pq.ParquetFile(documents).metadata.num_row_groups
    # Split row groups so that there are a few tasks per worker.
    n_parts = max(1, -(-4 * n_jobs // max(1, groups)))
    tasks = [(documents, g, part, n_parts) for g in range(groups) for part in range(n_parts)]
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        skip_path = Path(tmp) / "skip.parquet"
        pl.DataFrame({"doc_id": list(skip), "reason": list(skip.values())},
                     schema={"doc_id": pl.Int64, "reason": pl.String}).write_parquet(skip_path)
        with (pq.ParquetWriter(out, _TOKENS_SCHEMA) as writer, limited_threads(1),
              ProcessPoolExecutor(max(1, min(n_jobs, len(tasks))), mp_context=mp.get_context("spawn"),
                                  initializer=_init_worker,
                                  initargs=(tokenizer.to_dict(), cfg.model_dump(), skip_path)) as ex):
            for table in ex.map(_work, tasks):
                if table.num_rows:
                    writer.write_table(table)
    t = pl.read_parquet(out, columns=["reason", "title_ignored"])
    info = {r: int(c) for r, c in t["reason"].value_counts().iter_rows()}
    # Not a removal reason: kept papers whose title was left out as not English.
    info["titles_ignored"] = int((t["title_ignored"] & (t["reason"] == "ok")).sum())
    log.info("preprocess: %s", info)
    return info
