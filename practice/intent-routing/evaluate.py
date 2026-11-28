"""Aggregate intent-routing judgments and compare matching submissions."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from judge import IN_SCOPE_INTENTS, OUT_OF_SCOPE, Judgment, grade_reply


@dataclass(frozen=True)
class Submission:
    sample_id: str
    expected_intent: str
    raw_reply: str
    same_domain_oos: bool = False


@dataclass(frozen=True)
class ScoredSubmission:
    submission: Submission
    judgment: Judgment


def score_submissions(submissions: Iterable[Submission]) -> dict[str, ScoredSubmission]:
    """Keep every reply, including malformed ones, keyed by the manifest ID."""
    scored = {}
    for submission in submissions:
        sample_id = submission.sample_id
        if not sample_id.strip():
            raise ValueError("Empty sample ID")
        if sample_id in scored:
            raise ValueError(f"Duplicate sample ID: {sample_id!r}")
        if submission.same_domain_oos and submission.expected_intent != OUT_OF_SCOPE:
            raise ValueError(f"Expected out_of_scope for same-domain sample {sample_id!r}")
        judgment = grade_reply(submission.raw_reply, submission.expected_intent)
        scored[sample_id] = ScoredSubmission(submission, judgment)
    return scored


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def summarize(scored: Mapping[str, ScoredSubmission]) -> dict:
    """Report the main split and the same-domain OOS slice separately."""
    main = [row for row in scored.values() if not row.submission.same_domain_oos]
    same_domain = [row for row in scored.values() if row.submission.same_domain_oos]
    official_oos = [row for row in main if row.submission.expected_intent == OUT_OF_SCOPE]
    in_scope = [row for row in main if row.submission.expected_intent != OUT_OF_SCOPE]
    references = Counter(row.submission.expected_intent for row in in_scope)
    predictions = Counter(row.judgment.predicted_intent for row in in_scope)
    true_positives = Counter(
        row.submission.expected_intent for row in in_scope if row.judgment.task_correct
    )
    per_intent = {}
    for intent in IN_SCOPE_INTENTS:
        true_positive = true_positives[intent]
        false_positive = predictions[intent] - true_positive
        false_negative = references[intent] - true_positive
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_intent[intent] = {
            "support": references[intent],
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        }
    false_rejections = predictions[OUT_OF_SCOPE]
    format_errors = predictions[None]
    return {
        "task_accuracy": _rate(sum(row.judgment.task_correct for row in main), len(main)),
        "format_valid_rate": _rate(sum(row.judgment.format_valid for row in main), len(main)),
        "in_scope_macro_f1": (
            sum(intent["f1"] for intent in per_intent.values()) / len(IN_SCOPE_INTENTS)
            if in_scope else None
        ),
        "in_scope_support": len(in_scope),
        "in_scope_per_intent": per_intent,
        "official_oos_recall": _rate(
            sum(row.judgment.task_correct for row in official_oos), len(official_oos)
        ),
        "same_domain_oos_pass_rate": _rate(
            sum(row.judgment.task_correct for row in same_domain), len(same_domain)
        ),
        "in_scope_false_reject_rate": _rate(false_rejections, len(in_scope)),
        "in_scope_false_reject_or_format_error_rate": _rate(
            false_rejections + format_errors, len(in_scope)
        ),
    }


def compare_paired(
    baseline: Mapping[str, ScoredSubmission], candidate: Mapping[str, ScoredSubmission]
) -> dict:
    """Match rows by ID, rejecting any change to the paired reference or slice."""
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired runs have different sample IDs")
    transitions = {
        (True, True): "both_correct",
        (False, False): "both_wrong",
        (False, True): "wrong_to_correct",
        (True, False): "correct_to_wrong",
    }
    groups = {
        group: {transition: [] for transition in transitions.values()}
        for group in ("main", "same_domain_oos")
    }
    for sample_id in sorted(baseline):
        before = baseline[sample_id]
        after = candidate[sample_id]
        if before.submission.expected_intent != after.submission.expected_intent:
            raise ValueError(f"Paired reference changed for {sample_id!r}")
        if before.submission.same_domain_oos != after.submission.same_domain_oos:
            raise ValueError(f"Paired slice changed for {sample_id!r}")
        group = "same_domain_oos" if before.submission.same_domain_oos else "main"
        transition = transitions[before.judgment.task_correct, after.judgment.task_correct]
        groups[group][transition].append(sample_id)
    return {
        group: {
            transition: {"count": len(ids), "sample_ids": ids}
            for transition, ids in changes.items()
        }
        for group, changes in groups.items()
    }
