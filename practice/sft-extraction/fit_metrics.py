"""Masked next-token diagnostics and evidence replay, sharing strict raw scoring."""
import math
from pathlib import Path
from contract import score
from prompt_diagnostic import read
from review_repair import paired


def target_statistics(logits, labels):
    import torch
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("sequence logits and labels required")
    mask = labels[1:] != -100
    if not mask.any():
        raise ValueError("no supervised next-token targets")
    selected = logits[:-1][mask].float()
    targets = labels[1:][mask]
    log_probs = torch.log_softmax(selected, dim=-1)
    target_logps = log_probs.gather(1, targets[:, None]).squeeze(1)
    top_logps, top_ids = log_probs.max(dim=-1)
    if not torch.isfinite(target_logps).all():
        raise ValueError("nonfinite target log probability")
    return {"positions": torch.arange(1, labels.shape[0], device=labels.device)[mask].cpu().tolist(),
            "target_ids": targets.cpu().tolist(), "target_logps": target_logps.cpu().tolist(),
            "argmax_ids": top_ids.cpu().tolist(), "argmax_logps": top_logps.cpu().tolist()}


def first_difference(gold, generated):
    for i, (a, b) in enumerate(zip(gold, generated)):
        if a != b:
            return i
    return None if len(gold) == len(generated) else min(len(gold), len(generated))


def condition_summary(rows, outputs, teacher, fixtures):
    expected_ids = [r["id"] for r in rows]
    if [o["id"] for o in outputs] != expected_ids or [t["id"] for t in teacher] != expected_ids:
        raise ValueError("missing or reordered evidence")
    total_nll, total_tokens, argmax_correct, exact, losses, failed, differences = 0.0, 0, 0, 0, [], [], []
    for row, output, actual in zip(rows, outputs, teacher):
        raw_score = score(output["raw"], row["expected"])
        if raw_score != output["score"]:
            raise ValueError("forged score")
        f = fixtures[row["id"]]
        gold = f["input_ids"][f["prefix_tokens"]:]
        tokens = actual["tokens"]
        if [t["target_id"] for t in tokens] != gold or [t["input_position"] for t in tokens] != list(range(f["prefix_tokens"], len(f["input_ids"]))):
            raise ValueError("teacher targets differ from fixture/mask")
        if any(not math.isfinite(t["target_logp"]) or t["target_logp"] > 0.000001 for t in tokens):
            raise ValueError("invalid target log probability")
        nll = -sum(t["target_logp"] for t in tokens)
        mean = nll / len(tokens)
        if abs(mean - actual["model_loss"]) > 0.00001:
            raise ValueError("reconstructed NLL disagrees with model loss")
        mismatch = first_difference(gold, output["token_ids"])
        if actual["first_difference_index"] != mismatch:
            raise ValueError("incorrect first difference")
        exact += raw_score["exact"]
        if not raw_score["exact"]: failed.append(row["id"])
        if mismatch is not None:
            differences.append({"id": row["id"], "index": mismatch, "serialization_only": raw_score["exact"],
                                "gold_token": tokens[mismatch] if mismatch < len(tokens) else None,
                                "free_token_id": output["token_ids"][mismatch] if mismatch < len(output["token_ids"]) else None})
        total_nll += nll; total_tokens += len(tokens); losses.append(mean)
        argmax_correct += sum(t["target_id"] == t["argmax_id"] for t in tokens)
    return {"cases": len(rows), "exact": exact, "failed_ids": failed, "supervised_tokens": total_tokens,
            "target_nll_sum": total_nll, "token_weighted_mean_nll": total_nll / total_tokens,
            "example_mean_nll": sum(losses) / len(losses), "teacher_argmax_correct_tokens": argmax_correct,
            "first_differences": differences}


def review(folder):
    folder = Path(folder)
    rows, protocol = read(folder / "cases.json"), read(folder / "protocol.json")
    fixtures = {r["id"]: r for r in read(folder / "token-fixtures.json")}
    summaries, outputs = {}, {}
    for condition in protocol["conditions"]:
        outputs[condition] = read(folder / f"{condition}-generation.json")
        summaries[condition] = condition_summary(rows, outputs[condition], read(folder / f"{condition}-teacher.json"), fixtures)
    transitions = {a + "-to-" + b: paired(rows, outputs[a], outputs[b])
                   for a, b in (("baseline", "control"), ("baseline", "repair"), ("control", "repair"))}
    weights_unchanged = all(read(folder / f"{c}-weights.json")["unchanged"] for c in protocol["conditions"])
    return {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
            "total_generation_calls": sum(s["cases"] for s in summaries.values()),
            "teacher_forced_forwards": sum(len(read(folder / f"{c}-teacher.json")) for c in protocol["conditions"]),
            "weights_unchanged": weights_unchanged, "summaries": summaries, "transitions": transitions,
            "scope": protocol["scope"]}
