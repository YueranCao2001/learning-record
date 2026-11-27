"""Score one intent-routing reply against the project's JSON contract."""

import json
from dataclasses import dataclass

IN_SCOPE_INTENTS = (
    "balance",
    "bill_balance",
    "bill_due",
    "pay_bill",
    "transfer",
    "transactions",
    "report_fraud",
    "report_lost_card",
    "card_declined",
    "credit_limit",
    "credit_limit_change",
)
OUT_OF_SCOPE = "out_of_scope"
VALID_INTENTS = frozenset((*IN_SCOPE_INTENTS, OUT_OF_SCOPE))


@dataclass(frozen=True)
class Judgment:
    predicted_intent: str | None
    task_correct: bool
    format_error: str | None

    @property
    def format_valid(self) -> bool:
        return self.format_error is None


class _ObjectPairs(list[tuple[str, object]]):
    """Preserve repeated object keys and distinguish objects from arrays."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON literal")


def grade_reply(raw_reply: str, expected_intent: str) -> Judgment:
    """Return the prediction and judgment for one submitted reply."""
    if expected_intent not in VALID_INTENTS:
        raise ValueError(f"Unknown reference intent: {expected_intent!r}")
    if not raw_reply.strip(" \t\n\r"):
        return Judgment(None, False, "empty_reply")

    try:
        payload = json.loads(
            raw_reply,
            object_pairs_hook=_ObjectPairs,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        return Judgment(None, False, "invalid_json")

    if not isinstance(payload, _ObjectPairs):
        return Judgment(None, False, "not_object")
    keys = [key for key, _ in payload]
    if len(set(keys)) != len(keys):
        return Judgment(None, False, "duplicate_key")
    if keys != ["intent"]:
        return Judgment(None, False, "wrong_keys")

    intent = payload[0][1]
    if not isinstance(intent, str):
        return Judgment(None, False, "intent_not_string")
    if intent not in VALID_INTENTS:
        return Judgment(None, False, "unknown_intent")
    return Judgment(intent, intent == expected_intent, None)
