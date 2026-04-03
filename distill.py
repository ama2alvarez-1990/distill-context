"""distill — Information-theoretic context compression for LLMs.

Uses Shannon entropy + TF-IDF novelty to identify and keep the most
information-dense content, discarding low-entropy filler.

Author: Amado Alvarez Sueiras
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from typing import Literal


# ---------------------------------------------------------------------------
# Tokenisation (stdlib-only, no tiktoken dependency)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=[.!?])$", re.MULTILINE)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving non-empty ones."""
    raw = _SENT_RE.split(text.strip())
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences or [text.strip()]


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def compute_token_entropy(text: str) -> float:
    """Compute Shannon entropy of the token (word) distribution in *text*.

    Args:
        text: Any string.

    Returns:
        Shannon entropy in bits.  Returns 0.0 for empty or single-token text.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _sentence_entropy(sentence: str) -> float:
    """Entropy of a single sentence (character-level unigram for short text)."""
    tokens = _tokenize(sentence)
    if len(tokens) < 3:
        # Very short sentences: use character-level entropy
        chars = [c.lower() for c in sentence if c.strip()]
        if not chars:
            return 0.0
        counts = Counter(chars)
        total = len(chars)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ---------------------------------------------------------------------------
# TF-IDF novelty
# ---------------------------------------------------------------------------

def _compute_tfidf_scores(sentences: list[str]) -> list[float]:
    """Return a TF-IDF-based novelty score per sentence.

    Each sentence is treated as a 'document'.  A sentence whose terms appear
    rarely across the full corpus scores higher (more novel/unique).

    Args:
        sentences: List of sentence strings.

    Returns:
        List of novelty scores, one per sentence (higher = more novel).
    """
    n = len(sentences)
    if n == 0:
        return []

    # Tokenise each sentence
    doc_tokens: list[list[str]] = [_tokenize(s) for s in sentences]

    # Document frequency for each term
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    scores: list[float] = []
    for tokens in doc_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        score = 0.0
        for term, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = math.log((n + 1) / (df[term] + 1)) + 1.0
            score += tf_val * idf_val
        scores.append(score / len(tf))
    return scores


# ---------------------------------------------------------------------------
# Position weights
# ---------------------------------------------------------------------------

def _position_weights(n: int) -> list[float]:
    """Generate position weights: recent sentences weighted higher.

    Uses a mild exponential decay so the *last* sentence has weight 1.0
    and the first has weight ~0.3 (for a 20-sentence window).

    Args:
        n: Number of sentences.

    Returns:
        List of floats, length *n*.
    """
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    decay = 0.96  # per-step decay going backwards from end
    weights = [decay ** (n - 1 - i) for i in range(n)]
    # Normalise to [0, 1]
    w_max = max(weights)
    return [w / w_max for w in weights]


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_sentence_importance(
    sentences: list[str],
    weights: tuple[float, float, float] = (0.45, 0.45, 0.1),
) -> list[float]:
    """Compute an importance score for each sentence.

    Score is a weighted combination of:
      - Entropy (information density of the sentence itself)
      - TF-IDF novelty (how unique vs the rest of the context)
      - Position weight (recent = more important)

    Args:
        sentences: List of sentence strings.
        weights: (entropy_weight, tfidf_weight, position_weight).  Must sum to 1.

    Returns:
        List of floats in [0, 1], one per sentence.
    """
    if not sentences:
        return []

    w_ent, w_tfidf, w_pos = weights

    # Raw entropy per sentence
    entropies = [_sentence_entropy(s) for s in sentences]

    # TF-IDF novelty
    tfidf_scores = _compute_tfidf_scores(sentences)

    # Position weights
    pos_weights = _position_weights(len(sentences))

    def _normalise(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    norm_ent = _normalise(entropies)
    norm_tfidf = _normalise(tfidf_scores)
    norm_pos = pos_weights  # already in [0,1]

    scores = [
        w_ent * e + w_tfidf * t + w_pos * p
        for e, t, p in zip(norm_ent, norm_tfidf, norm_pos)
    ]
    return scores


# ---------------------------------------------------------------------------
# Approximate token counting
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _approx_tokens(text: str) -> int:
    """Approximate token count: ~0.75 words per token (GPT-style)."""
    words = len(_WS_RE.split(text.strip()))
    return max(1, int(words / 0.75))


# ---------------------------------------------------------------------------
# distill() — main compression function
# ---------------------------------------------------------------------------

def distill(
    text: str,
    target_ratio: float = 0.3,
    method: Literal["entropy", "tfidf", "combined"] = "entropy",
) -> str:
    """Compress *text* to approximately *target_ratio* of its original length.

    Keeps the highest-importance sentences according to *method*:
      - ``"entropy"``  — pure Shannon entropy per sentence
      - ``"tfidf"``    — TF-IDF novelty score
      - ``"combined"`` — weighted combination of entropy + TF-IDF + position

    Args:
        text: Input text to compress.
        target_ratio: Fraction of sentences to retain (0 < ratio ≤ 1).
        method: Scoring method.

    Returns:
        Compressed text preserving sentence order.
    """
    if not text.strip():
        return text

    target_ratio = max(0.05, min(1.0, target_ratio))

    sentences = _split_sentences(text)
    n = len(sentences)

    if n <= 1:
        return text

    # Compute per-sentence scores
    if method == "entropy":
        raw = [_sentence_entropy(s) for s in sentences]
        # Normalise
        lo, hi = min(raw), max(raw)
        if hi == lo:
            scores = [1.0] * n
        else:
            scores = [(v - lo) / (hi - lo) for v in raw]
    elif method == "tfidf":
        raw = _compute_tfidf_scores(sentences)
        lo, hi = min(raw), max(raw)
        if hi == lo:
            scores = [1.0] * n
        else:
            scores = [(v - lo) / (hi - lo) for v in raw]
    else:  # combined
        scores = compute_sentence_importance(sentences)

    # How many sentences to keep
    keep_count = max(1, round(n * target_ratio))

    # Pick indices of top-scoring sentences (preserve original order)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    keep_indices = set(idx for idx, _ in indexed[:keep_count])
    kept = [s for i, s in enumerate(sentences) if i in keep_indices]

    return " ".join(kept)


# ---------------------------------------------------------------------------
# distill_conversation()
# ---------------------------------------------------------------------------

def distill_conversation(
    messages: list[dict],
    target_tokens: int,
    method: Literal["entropy", "tfidf", "combined"] = "combined",
) -> list[dict]:
    """Compress a conversation to fit within *target_tokens*.

    Strategy:
      1. Estimate current total tokens.
      2. If already within budget, return as-is.
      3. Otherwise, score all sentences across all messages, then iteratively
         remove the lowest-scoring sentence from the largest message until
         the budget is met.  System messages and the last user/assistant
         exchange are always preserved in full.

    Args:
        messages: List of ``{"role": str, "content": str}`` dicts.
        target_tokens: Maximum token budget for the returned conversation.
        method: Scoring method.

    Returns:
        Compressed list of messages (same schema).
    """
    if not messages:
        return messages

    total_tokens = sum(_approx_tokens(m.get("content", "")) for m in messages)
    if total_tokens <= target_tokens:
        return messages

    ratio = target_tokens / total_tokens

    # Never compress system messages or the final two messages
    protected_indices: set[int] = set()
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            protected_indices.add(i)
    # Protect last exchange (last 2 messages)
    if len(messages) >= 2:
        protected_indices.update([len(messages) - 2, len(messages) - 1])
    elif messages:
        protected_indices.add(len(messages) - 1)

    result: list[dict] = []
    for i, msg in enumerate(messages):
        if i in protected_indices:
            result.append(dict(msg))
            continue
        content = msg.get("content", "")
        compressed = distill(content, target_ratio=ratio, method=method)
        result.append({**msg, "content": compressed})

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``distill`` CLI."""
    parser = argparse.ArgumentParser(
        prog="distill",
        description="Information-theoretic context compression for LLMs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # distill compress
    comp = sub.add_parser("compress", help="Compress a text file.")
    comp.add_argument("input", help="Input file path (use - for stdin).")
    comp.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Target compression ratio (default: 0.3).",
    )
    comp.add_argument(
        "--method",
        choices=["entropy", "tfidf", "combined"],
        default="combined",
        help="Scoring method (default: combined).",
    )
    comp.add_argument(
        "--stats",
        action="store_true",
        help="Print compression statistics to stderr.",
    )

    # distill conversation
    conv = sub.add_parser("conversation", help="Compress a conversation JSON file.")
    conv.add_argument(
        "input",
        help='Input JSON file path (list of {"role","content"} objects). Use - for stdin.',
    )
    conv.add_argument(
        "--target-tokens",
        type=int,
        default=4000,
        help="Target token budget (default: 4000).",
    )
    conv.add_argument(
        "--method",
        choices=["entropy", "tfidf", "combined"],
        default="combined",
        help="Scoring method (default: combined).",
    )
    conv.add_argument(
        "--stats",
        action="store_true",
        help="Print compression statistics to stderr.",
    )

    args = parser.parse_args()

    if args.command == "compress":
        if args.input == "-":
            text = sys.stdin.read()
        else:
            with open(args.input, encoding="utf-8") as fh:
                text = fh.read()

        original_tokens = _approx_tokens(text)
        result = distill(text, target_ratio=args.ratio, method=args.method)
        compressed_tokens = _approx_tokens(result)

        print(result)

        if args.stats:
            actual_ratio = compressed_tokens / max(original_tokens, 1)
            print(
                f"\n[distill] {original_tokens} → {compressed_tokens} tokens "
                f"({actual_ratio:.1%} of original)",
                file=sys.stderr,
            )

    elif args.command == "conversation":
        if args.input == "-":
            raw = sys.stdin.read()
        else:
            with open(args.input, encoding="utf-8") as fh:
                raw = fh.read()

        messages = json.loads(raw)
        if not isinstance(messages, list):
            print("Error: input must be a JSON array of message objects.", file=sys.stderr)
            sys.exit(1)

        original_tokens = sum(_approx_tokens(m.get("content", "")) for m in messages)
        result = distill_conversation(
            messages,
            target_tokens=args.target_tokens,
            method=args.method,
        )
        compressed_tokens = sum(_approx_tokens(m.get("content", "")) for m in result)

        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.stats:
            actual_ratio = compressed_tokens / max(original_tokens, 1)
            print(
                f"\n[distill] {original_tokens} → {compressed_tokens} tokens "
                f"({actual_ratio:.1%} of original)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
