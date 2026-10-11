import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


LEXICON = tuple(sorted({"普通设备", "定制设备", "设备", "支持", "退货",
                        "保修", "期限", "个月", "适用", "规则"},
                       key=lambda x: (-len(x), x)))


def spans(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[^\W_]+", text)


def words(text):
    result = []
    for part in spans(text):
        i = 0
        while i < len(part):
            token = next((w for w in LEXICON if part.startswith(w, i)), part[i])
            result.append(token)
            i += len(token)
    return result


def bigrams(text):
    return [part[i:i + 2] for part in spans(text)
            for i in range(len(part) - 1)]


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    readers: frozenset[str]


def rank(docs, query, user, analyzer, method, k=2):
    if type(k) is not int or k < 1:
        raise ValueError("positive integer k required")
    if method not in {"overlap", "bm25"}:
        raise ValueError("unknown method")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("duplicate document id")
    visible = sorted((d for d in docs if user in d.readers), key=lambda d: d.id)
    if not visible:
        return ()
    counts = [Counter(analyzer(d.text)) for d in visible]
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / len(visible)
    df = Counter(t for c in counts for t in c)
    terms = set(analyzer(query))
    ranked = []
    for d, c, length in zip(visible, counts, lengths):
        if method == "overlap":
            score = len(terms & c.keys())
        else:
            score = 0.0
            for term in sorted(terms & c.keys()):
                tf = c[term]
                norm = 1.2 * (1 - 0.75 + 0.75 * length / avg)
                score += math.log(len(visible) / df[term]) * tf * 2.2 / (tf + norm)
        if score > 0:
            ranked.append((score, d.id))
    return tuple({"id": id_, "score": round(score, 12)}
                 for score, id_ in sorted(ranked, key=lambda x: (-x[0], x[1]))[:k])


def search(docs, query, user, analyzer, method, k=2):
    return tuple(hit["id"] for hit in rank(docs, query, user, analyzer, method, k))
