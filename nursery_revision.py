"""Stage 2, Part 1 - "Exam Time".

We hand back passages, not answers: the android reads them and writes its own
answer, and a judge checks whether that answer carries the required fact. So
the only thing that matters is that the fact is somewhere inside a 900-token
ceiling.

The retrieval problem here is harder than it looks, because the questions
paraphrase away from the source. The brief's own example asks when the "sensor
grid" was "brought back into alignment"; the answer sentence says the
"Kesterline array was recalibrated on 14 March". Neither "sensor" nor "grid"
occurs anywhere in the material, and "alignment" occurs only in a *different*
document. Rank by word overlap alone and you confidently return the wrong
document.

What survives that is answer-type gating. A "when" question can only be
answered by a sentence carrying a date, and the whole corpus holds few enough
dates to shortlist them all; the same trick narrows "who" to sentences naming
a person. Lexical similarity then orders the shortlist rather than choosing it.

The budget is additive and measured per passage with o200k_base, so the
bookkeeping is exact: each unit is counted once, adding one costs exactly its
own count, and order is irrelevant. Boundaries are paid for - splitting costs
about a token per split - which is cheap next to the recall won by returning
many small candidates instead of a few long ones.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import tiktoken

ENCODING = tiktoken.get_encoding("o200k_base")

# The wrapper enforces 900. Stop under it: the count is exact, but a response
# that lands on the line leaves no room for an off-by-one anywhere.
TOKEN_BUDGET = int(os.getenv("REVISION_TOKEN_BUDGET", "880"))

MATERIALS_DIR = Path(
    os.getenv("STUDY_MATERIALS_DIR", Path(__file__).parent / "study_materials")
)

_STOPWORDS = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "before",
    "but", "by", "can", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "many", "much",
    "of", "on", "or", "our", "over", "so", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "under", "was",
    "were", "what", "when", "where", "which", "who", "whose", "why", "will",
    "with", "you", "your",
}

_WORD = re.compile(r"[a-z0-9]+")

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october"
    "|november|december"
)
# "14 March", "March 14", and the spelled-out "the fifth of January" all appear.
_DATE_RE = re.compile(
    rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b"
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}}\b"
    rf"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    rf"|eleventh|twelfth|thirteenth|fourteenth|fifteenth|twentieth|twenty-first)"
    rf"\s+of\s+(?:{_MONTHS})\b",
    re.I,
)

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    "|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty"
    "|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
)
_NUMBER_RE = re.compile(rf"\b\d[\d,.]*\b|\b(?:{_NUMBER_WORDS})(?:-(?:{_NUMBER_WORDS}))?\b", re.I)

# "Dr. Ansel Kovrith", "Cordelia Vance", "Perrin Ashwicke".
_PERSON_RE = re.compile(r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")

DATE, NUMBER, PERSON, NAME, GENERAL = "date", "number", "person", "name", "general"

# Questions paraphrase; the material does not. These map the phrasings a
# question tends to reach for onto the vocabulary the documents actually use,
# so "who is in charge" can find "has served as station director". Deliberately
# generic - families of paraphrase, not answers to specific questions.
_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"in charge|leads?|runs?|heads?|oversees|responsible|manages?|charge of|presides",
     ("director", "chair", "chief", "lead", "architect", "investigator", "officer",
      "served", "authority", "oversees", "presided", "primary")),
    (r"how often|frequency|interval|cadence|regularly|schedule|every",
     ("every", "days", "hours", "cycle", "schedule", "rotates", "runs", "docks")),
    (r"identifier|call sign|callsign|tag|code|designation|designated|named|called|name",
     ("call", "sign", "tag", "code", "designated", "named", "informally", "internal",
      "officially", "stream")),
    (r"maximum|limit|cap|ceiling|threshold|budget|longest|most|allowed|permitted|may use",
     ("maximum", "limit", "capped", "restricted", "exceeds", "ceiling", "budget",
      "threshold", "allowance", "per")),
    (r"minimum|least|fewest|never fell|floor",
     ("minimum", "never", "below", "floor", "least")),
    (r"deep|depth|below the surface",
     ("depth", "meters", "below", "surface")),
    (r"broke|broken|fail|failure|fault|went wrong|outage|incident|struck",
     ("failure", "fault", "incident", "corroded", "worn", "traced")),
    (r"start|started|begin|began|commence",
     ("began", "start", "adopted", "shipped", "first")),
    (r"how many people|staff|employ|workforce|crew|members|headcount|roster",
     ("population", "resident", "employs", "certified", "maintains", "counts",
      "households", "engineers", "staff", "crew")),
    (r"stay|stays|watched|observation|duration|how long",
     ("minutes", "hours", "seconds", "period", "window", "time")),
    (r"tight|tightly|fasten|torque|bolt",
     ("torque", "torqued", "newton", "meters", "gasket")),
)


def expand_query(question: str) -> list[str]:
    """Extra vocabulary implied by how the question is phrased."""
    lowered = question.lower()
    extra: list[str] = []
    for pattern, words in _EXPANSIONS:
        if re.search(pattern, lowered):
            extra.extend(words)
    return extra


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def _char_ngrams(text: str, n: int = 4) -> Counter:
    squashed = re.sub(r"\s+", " ", text.lower())
    return Counter(squashed[i : i + n] for i in range(max(0, len(squashed) - n + 1)))


def count_tokens(text: str) -> int:
    """Cost of one passage, counted the way the budget counts it."""
    return len(ENCODING.encode(text))


def classify_question(question: str) -> str:
    """What kind of thing would answer this question?

    Used to shortlist candidates by what they *contain* rather than by what
    words they share with the question, which is what makes paraphrased
    questions survivable.
    """
    q = question.lower().strip()
    if re.search(r"\bwho\b|\bwhose\b|\bwhom\b", q):
        return PERSON
    if re.search(r"\bwhen\b|\bwhat date\b|\bwhich day\b|\bwhat day\b|\bhow long ago\b", q):
        return DATE
    if re.search(
        r"\bhow many\b|\bhow much\b|\bhow long\b|\bhow deep\b|\bhow often\b|\bhow far\b"
        r"|\blimit\b|\bthreshold\b|\bcap\b|\bbudget\b|\bmaximum\b|\bminimum\b|\bceiling\b"
        r"|\bcapacity\b|\bdepth\b|\bdose\b|\btorque\b|\binterval\b|\bfrequency\b|\bsize\b",
        q,
    ):
        return NUMBER
    if re.search(
        r"\bwhat is .*(?:called|named)\b|\bwhat name\b|\bwhich .*(?:called|named)\b"
        r"|\bcall sign\b|\bdesignat|\btag\b|\bcode\b|\btitle\b",
        q,
    ):
        return NAME
    return GENERAL


@dataclass
class Unit:
    doc: str
    section: str
    index: int
    body: str
    text: str = ""
    tokens: int = 0
    words: list[str] = field(default_factory=list)
    ngrams: Counter = field(default_factory=Counter)
    has_date: bool = False
    has_number: bool = False
    has_person: bool = False
    has_proper: bool = False

    def finalise(self, label: str) -> None:
        # A short provenance tag costs a handful of tokens and buys the android
        # the context to tell near-identical facts apart - several documents
        # carry a "Halberd"/"Fantail Two"/"Driftglass Two" style decoy.
        self.text = f"[{label} | {self.section}] {self.body}" if self.section else f"[{label}] {self.body}"
        self.tokens = count_tokens(self.text)
        self.words = _tokens(self.body)
        self.ngrams = _char_ngrams(self.body)
        self.has_date = bool(_DATE_RE.search(self.body))
        self.has_number = bool(_NUMBER_RE.search(self.body))
        self.has_person = bool(_PERSON_RE.search(self.body))
        self.has_proper = bool(_PROPER_RE.search(self.body))

    def matches(self, kind: str) -> bool:
        return {
            DATE: self.has_date,
            NUMBER: self.has_number,
            PERSON: self.has_person,
            NAME: self.has_proper,
            GENERAL: True,
        }[kind]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_units(doc_label: str, text: str) -> list[Unit]:
    """One unit per sentence, tagged with the section heading above it.

    Sentences are the smallest safe unit: a fact and the date or figure that
    answers it live in the same sentence, so splitting finer would separate
    them, and merging coarser would spend budget on neighbours that answer
    nothing.
    """
    units: list[Unit] = []
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading.startswith("NOTE:"):
                continue
            section = heading
            continue
        for sentence in _split_sentences(line):
            if len(sentence) < 25:
                continue
            units.append(Unit(doc=doc_label, section=section, index=len(units), body=sentence))
    for unit in units:
        unit.finalise(doc_label)
    return units


class RevisionIndex:
    """BM25 over sentences, gated by what kind of answer each one could give."""

    K1 = 1.4
    B = 0.7

    def __init__(self, units: Sequence[Unit]):
        self.units = list(units)
        self.doc_freq: Counter = Counter()
        for unit in self.units:
            for word in set(unit.words):
                self.doc_freq[word] += 1
        self.total = len(self.units) or 1
        self.avg_len = (sum(len(u.words) for u in self.units) / self.total) or 1.0

        # Vocabulary of each document as a whole. A question often names its
        # subject ("the trench outpost", "the growers cooperative") even when
        # it paraphrases the fact itself, and that is a much stronger signal
        # than any single sentence carries - especially for "how many"
        # questions, where every document offers competing numbers.
        self.doc_words: dict[str, Counter] = {}
        for unit in self.units:
            self.doc_words.setdefault(unit.doc, Counter()).update(unit.words)

    def _bm25(self, query_words: Sequence[str], unit: Unit) -> float:
        if not unit.words:
            return 0.0
        counts = Counter(unit.words)
        length = len(unit.words)
        score = 0.0
        for word in query_words:
            frequency = counts.get(word)
            if not frequency:
                continue
            idf = math.log(1 + (self.total - self.doc_freq[word] + 0.5) / (self.doc_freq[word] + 0.5))
            score += idf * (frequency * (self.K1 + 1)) / (
                frequency + self.K1 * (1 - self.B + self.B * length / self.avg_len)
            )
        return score

    @staticmethod
    def _fuzzy(query_ngrams: Counter, unit: Unit) -> float:
        if not query_ngrams or not unit.ngrams:
            return 0.0
        shared = sum(min(c, unit.ngrams.get(g, 0)) for g, c in query_ngrams.items())
        return shared / (sum(query_ngrams.values()) or 1)

    def _document_prior(self, query_words: Sequence[str]) -> dict[str, float]:
        """How well each document as a whole answers to this question."""
        scores: dict[str, float] = {}
        for doc, counts in self.doc_words.items():
            length = sum(counts.values()) or 1
            score = 0.0
            for word in set(query_words):
                frequency = counts.get(word)
                if frequency:
                    # Rare words carry the subject; common ones are noise.
                    idf = math.log(1 + len(self.doc_words) / (1 + sum(
                        1 for c in self.doc_words.values() if word in c
                    )))
                    score += idf * math.log(1 + frequency) / math.log(1 + length)
            scores[doc] = score
        top = max(scores.values(), default=0.0) or 1.0
        return {doc: score / top for doc, score in scores.items()}

    def rank(self, question: str) -> list[tuple[float, Unit]]:
        query_words = _tokens(question)
        query_ngrams = _char_ngrams(question)
        # Expansion terms rank at a discount: they broaden reach without
        # letting a generic synonym outvote a word the asker actually used.
        expanded = [w for w in expand_query(question) if w not in query_words]
        lexical = [
            self._bm25(query_words, u) + 0.35 * self._bm25(expanded, u)
            for u in self.units
        ]
        fuzzy = [self._fuzzy(query_ngrams, u) for u in self.units]
        prior = self._document_prior(query_words)
        top_lex = max(lexical, default=0.0) or 1.0
        top_fuz = max(fuzzy, default=0.0) or 1.0
        scored = [
            (
                lexical[i] / top_lex
                + 0.4 * (fuzzy[i] / top_fuz)
                + 0.6 * prior.get(unit.doc, 0.0),
                unit,
            )
            for i, unit in enumerate(self.units)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def select(self, question: str, budget: int = TOKEN_BUDGET) -> list[str]:
        """Passages for one question, filling the budget.

        Candidates that could actually answer the question go first, in
        similarity order; whatever budget is left is then spent on the best of
        the rest, so a misclassified question degrades to plain retrieval
        instead of returning nothing useful.
        """
        if not self.units:
            raise RuntimeError(
                f"no study material indexed (looked in {MATERIALS_DIR}); "
                "add the documents so retrieval has something to search"
            )

        kind = classify_question(question)
        ranked = self.rank(question)

        typed = [(s, u) for s, u in ranked if u.matches(kind)]
        rest = [(s, u) for s, u in ranked if not u.matches(kind)]

        chosen: list[Unit] = []
        spent = 0
        seen: set[str] = set()

        for _, unit in typed + rest:
            if unit.body in seen or spent + unit.tokens > budget:
                continue
            chosen.append(unit)
            seen.add(unit.body)
            spent += unit.tokens

        if not chosen:
            cheapest = min(self.units, key=lambda u: u.tokens)
            if cheapest.tokens <= budget:
                chosen = [cheapest]

        # Measure what is actually going out. Going over is scored zero, and
        # the assembled list is what gets measured, so trim - least relevant
        # first - until it genuinely fits.
        while chosen:
            passages = [u.text for u in chosen]
            if budget_used(passages) <= budget:
                return passages
            chosen.pop()
        return []


_index: RevisionIndex | None = None
_index_lock = threading.Lock()


def load_units(directory: Path | None = None) -> list[Unit]:
    directory = Path(directory or MATERIALS_DIR)
    units: list[Unit] = []
    if not directory.is_dir():
        return units
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".text"} or not path.is_file():
            continue
        label = re.sub(r"^\d+[_\-]", "", path.stem).replace("_", " ").title()
        units.extend(build_units(label, path.read_text(encoding="utf-8", errors="ignore")))
    return units


def get_index(refresh: bool = False) -> RevisionIndex:
    """Build the index once per process; every question reuses it."""
    global _index
    with _index_lock:
        if _index is None or refresh:
            _index = RevisionIndex(load_units())
        return _index


def find_passages(question: str, budget: int = TOKEN_BUDGET) -> list[str]:
    """The passages for one revision question."""
    return get_index().select(question or "", budget=budget)


def budget_used(passages: Iterable[str]) -> int:
    """What the wrapper will measure for these passages."""
    return sum(count_tokens(p) for p in passages)
