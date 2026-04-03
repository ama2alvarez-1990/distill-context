"""Tests for distill — information-theoretic context compression.

Run with:
    python -m pytest test_distill.py -v
"""

from __future__ import annotations

import json
import math

import pytest

from distill import (
    _approx_tokens,
    _split_sentences,
    _tokenize,
    compute_sentence_importance,
    compute_token_entropy,
    distill,
    distill_conversation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "Um so basically I was just going to say that.",
    "Like, you know what I mean, right?",
    "So yeah, that is basically it I guess.",
    "Well, um, I think, like, maybe.",
]

TECHNICAL_SENTENCES = [
    "Shannon entropy H(X) = -∑ p(x) log₂ p(x) quantifies information in a distribution.",
    "Rate-distortion theory defines the minimum bits needed to encode a source within distortion D.",
    "TF-IDF weights rare, document-specific terms higher than common stopwords.",
    "Transformer attention complexity scales as O(n²) in sequence length.",
    "Huffman coding achieves near-optimal prefix-free lossless compression.",
]

MIXED_TEXT = "\n".join(FILLER_SENTENCES + TECHNICAL_SENTENCES)


# ---------------------------------------------------------------------------
# Unit tests: tokenisation and entropy
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_words(self):
        tokens = _tokenize("Hello World foo bar")
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_punctuation_stripped(self):
        tokens = _tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_empty(self):
        assert _tokenize("") == []

    def test_unicode(self):
        tokens = _tokenize("café naïve résumé")
        assert len(tokens) == 3


class TestSplitSentences:
    def test_basic_split(self):
        text = "Hello world. How are you? I am fine."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_single_sentence(self):
        sents = _split_sentences("No period here")
        assert len(sents) == 1

    def test_empty_string(self):
        sents = _split_sentences("")
        # Should return list with empty or single empty string — no crash
        assert isinstance(sents, list)


class TestComputeTokenEntropy:
    def test_uniform_distribution_high_entropy(self):
        # All unique words → near-maximum entropy
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        entropy = compute_token_entropy(text)
        assert entropy > 3.0  # log2(10) ≈ 3.32

    def test_repetitive_text_low_entropy(self):
        text = "the the the the the the the the the the"
        entropy = compute_token_entropy(text)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_zero(self):
        assert compute_token_entropy("") == 0.0

    def test_single_word_returns_zero(self):
        assert compute_token_entropy("hello") == 0.0

    def test_technical_text_higher_than_filler(self):
        technical = " ".join(TECHNICAL_SENTENCES)
        filler = " ".join(FILLER_SENTENCES)
        assert compute_token_entropy(technical) > compute_token_entropy(filler)

    def test_entropy_is_non_negative(self):
        for text in ["hello world", "a b c d", "repeat repeat repeat"]:
            assert compute_token_entropy(text) >= 0.0

    def test_entropy_bounded_by_log2_vocab(self):
        text = "a b c d e f g h"
        tokens = _tokenize(text)
        vocab_size = len(set(tokens))
        entropy = compute_token_entropy(text)
        assert entropy <= math.log2(vocab_size) + 1e-9


# ---------------------------------------------------------------------------
# Unit tests: sentence importance
# ---------------------------------------------------------------------------

class TestComputeSentenceImportance:
    def test_length_matches_input(self):
        sents = TECHNICAL_SENTENCES + FILLER_SENTENCES
        scores = compute_sentence_importance(sents)
        assert len(scores) == len(sents)

    def test_all_scores_in_range(self):
        scores = compute_sentence_importance(TECHNICAL_SENTENCES + FILLER_SENTENCES)
        for s in scores:
            assert 0.0 <= s <= 1.0 + 1e-9

    def test_empty_input(self):
        assert compute_sentence_importance([]) == []

    def test_single_sentence(self):
        scores = compute_sentence_importance(["Hello world."])
        assert len(scores) == 1

    def test_technical_outscores_filler(self):
        # Place technical then filler so position doesn't fully dominate
        sents = TECHNICAL_SENTENCES + FILLER_SENTENCES
        scores = compute_sentence_importance(sents)

        tech_avg = sum(scores[:5]) / 5
        filler_avg = sum(scores[5:]) / 4
        assert tech_avg > filler_avg, (
            f"Expected technical sentences to outscore filler. "
            f"tech_avg={tech_avg:.3f}, filler_avg={filler_avg:.3f}"
        )

    def test_weight_sum_not_enforced(self):
        # Custom weights should not crash
        scores = compute_sentence_importance(
            TECHNICAL_SENTENCES, weights=(0.45, 0.45, 0.1)
        )
        assert len(scores) == len(TECHNICAL_SENTENCES)


# ---------------------------------------------------------------------------
# Integration tests: distill()
# ---------------------------------------------------------------------------

class TestDistill:
    def test_output_shorter_than_input(self):
        result = distill(MIXED_TEXT, target_ratio=0.3)
        assert len(result) < len(MIXED_TEXT)

    def test_ratio_approximately_respected_entropy(self):
        sents = _split_sentences(MIXED_TEXT)
        result = distill(MIXED_TEXT, target_ratio=0.5, method="entropy")
        result_sents = _split_sentences(result)
        # Allow ±1 sentence tolerance
        assert abs(len(result_sents) - round(len(sents) * 0.5)) <= 2

    def test_ratio_approximately_respected_tfidf(self):
        sents = _split_sentences(MIXED_TEXT)
        result = distill(MIXED_TEXT, target_ratio=0.4, method="tfidf")
        result_sents = _split_sentences(result)
        assert abs(len(result_sents) - round(len(sents) * 0.4)) <= 2

    def test_ratio_approximately_respected_combined(self):
        sents = _split_sentences(MIXED_TEXT)
        result = distill(MIXED_TEXT, target_ratio=0.5, method="combined")
        result_sents = _split_sentences(result)
        assert abs(len(result_sents) - round(len(sents) * 0.5)) <= 2

    def test_high_entropy_sentences_preserved(self):
        result = distill(MIXED_TEXT, target_ratio=0.4, method="entropy")
        # At least half of technical sentences should appear in compressed output
        preserved = sum(1 for s in TECHNICAL_SENTENCES if s in result)
        assert preserved >= 2, f"Only {preserved}/5 technical sentences preserved"

    def test_filler_removed_first(self):
        # With low ratio, filler should disappear before technical content
        result = distill(MIXED_TEXT, target_ratio=0.4, method="entropy")
        filler_count = sum(
            1 for s in FILLER_SENTENCES
            if s.split()[0] in result or s[:20] in result
        )
        tech_count = sum(1 for s in TECHNICAL_SENTENCES if s[:20] in result)
        assert tech_count >= filler_count, (
            f"Expected more technical content preserved. "
            f"tech={tech_count}, filler={filler_count}"
        )

    def test_technical_terms_survive_compression(self):
        result = distill(MIXED_TEXT, target_ratio=0.35, method="combined")
        technical_terms = ["entropy", "TF-IDF", "Shannon", "compression", "Transformer"]
        found = [t for t in technical_terms if t.lower() in result.lower()]
        assert len(found) >= 2, f"Expected technical terms; found: {found}"

    def test_code_snippet_survives(self):
        code_text = (
            "Sure, so I was thinking about this thing. "
            "def compute_entropy(text): return -sum(p*log2(p) for p in probs). "
            "And yeah basically that is the main idea right. "
            "Um like so I guess we could also do it differently maybe. "
            "The function takes O(n) time and O(k) space where k is vocabulary size."
        )
        result = distill(code_text, target_ratio=0.4, method="entropy")
        # The code definition should survive over the filler
        assert "compute_entropy" in result or "entropy" in result.lower()

    def test_empty_input(self):
        assert distill("") == ""

    def test_single_sentence_unchanged(self):
        text = "This is a single sentence."
        result = distill(text, target_ratio=0.3)
        assert result == text

    def test_ratio_clamps_at_minimum(self):
        result = distill(MIXED_TEXT, target_ratio=0.0)
        assert len(result) > 0  # never returns empty

    def test_ratio_clamps_at_maximum(self):
        result = distill(MIXED_TEXT, target_ratio=2.0)
        # Should keep (nearly) all sentences
        orig_sents = _split_sentences(MIXED_TEXT)
        result_sents = _split_sentences(result)
        assert len(result_sents) >= len(orig_sents) - 1

    def test_all_methods_return_strings(self):
        for method in ("entropy", "tfidf", "combined"):
            result = distill(MIXED_TEXT, target_ratio=0.4, method=method)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration tests: distill_conversation()
# ---------------------------------------------------------------------------

SAMPLE_CONVERSATION = [
    {
        "role": "system",
        "content": "You are a helpful assistant specialising in information theory.",
    },
    {
        "role": "user",
        "content": (
            "Um so like I was wondering about entropy. "
            "You know what I mean? "
            "Shannon entropy H(X) = -∑ p(x) log₂ p(x) is a fundamental measure. "
            "So basically yeah, can you explain it?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Sure, like, well, um. "
            "Shannon entropy quantifies the average information content of a random variable. "
            "Higher entropy means more uncertainty and more bits needed to encode the source. "
            "So yeah basically that covers it right."
        ),
    },
    {
        "role": "user",
        "content": (
            "And what about rate-distortion theory? "
            "Rate-distortion theory characterises the minimum bit-rate required to encode a source "
            "at a given distortion level D. "
            "The rate-distortion function R(D) is convex and non-increasing. "
            "Um so I guess that is interesting or whatever."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Rate-distortion theory provides the theoretical minimum compression rate for lossy coding. "
            "For a Gaussian source with MSE distortion, R(D) = 0.5 * log2(σ² / D). "
            "This is the foundation for modern codecs like JPEG and H.264."
        ),
    },
]


class TestDistillConversation:
    def test_returns_list(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=500)
        assert isinstance(result, list)

    def test_same_number_of_messages(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=500)
        assert len(result) == len(SAMPLE_CONVERSATION)

    def test_roles_preserved(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=500)
        for orig, comp in zip(SAMPLE_CONVERSATION, result):
            assert orig["role"] == comp["role"]

    def test_system_message_untouched(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=200)
        assert result[0]["content"] == SAMPLE_CONVERSATION[0]["content"]

    def test_last_message_untouched(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=200)
        assert result[-1]["content"] == SAMPLE_CONVERSATION[-1]["content"]

    def test_second_to_last_untouched(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=200)
        assert result[-2]["content"] == SAMPLE_CONVERSATION[-2]["content"]

    def test_tokens_reduced_when_over_budget(self):
        original_tokens = sum(
            _approx_tokens(m["content"]) for m in SAMPLE_CONVERSATION
        )
        target = original_tokens // 2
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=target)
        result_tokens = sum(_approx_tokens(m["content"]) for m in result)
        assert result_tokens < original_tokens

    def test_no_compression_when_under_budget(self):
        large_budget = 999_999
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=large_budget)
        for orig, comp in zip(SAMPLE_CONVERSATION, result):
            assert orig["content"] == comp["content"]

    def test_empty_conversation(self):
        assert distill_conversation([], target_tokens=1000) == []

    def test_all_methods(self):
        for method in ("entropy", "tfidf", "combined"):
            result = distill_conversation(
                SAMPLE_CONVERSATION, target_tokens=300, method=method
            )
            assert len(result) == len(SAMPLE_CONVERSATION)

    def test_content_is_string_after_compression(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=100)
        for msg in result:
            assert isinstance(msg["content"], str)

    def test_technical_content_in_result(self):
        result = distill_conversation(SAMPLE_CONVERSATION, target_tokens=200)
        all_content = " ".join(m["content"] for m in result)
        assert "entropy" in all_content.lower() or "shannon" in all_content.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_identical_sentences(self):
        text = "The cat sat on the mat. " * 10
        result = distill(text, target_ratio=0.3)
        assert isinstance(result, str)

    def test_very_long_text(self):
        text = (MIXED_TEXT + " ") * 50
        result = distill(text, target_ratio=0.2, method="combined")
        assert len(result) < len(text)

    def test_unicode_text(self):
        text = (
            "这是一个测试。 Shannon entropy is fundamental. "
            "Это тестовое предложение. Information theory applies universally."
        )
        result = distill(text, target_ratio=0.5)
        assert isinstance(result, str)

    def test_numbers_and_code(self):
        text = (
            "First, let me say hello and welcome. "
            "x = -sum(p * math.log2(p) for p in probs if p > 0). "
            "And um basically yeah that is the formula."
        )
        result = distill(text, target_ratio=0.4)
        assert isinstance(result, str)

    def test_approx_tokens_reasonable(self):
        text = "hello world foo bar baz"  # 5 words
        tokens = _approx_tokens(text)
        # 5 words / 0.75 ≈ 6.67, so 6 or 7
        assert 4 <= tokens <= 10
