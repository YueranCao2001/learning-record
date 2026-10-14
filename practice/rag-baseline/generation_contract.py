"""Local contract v1: shape, request binding and provenance, not answer scoring."""
import hashlib
import json
from dataclasses import asdict

from citations import check_quote


def context_hash(evidence):
    raw = json.dumps([asdict(c) for c in evidence], ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_request(request_id, query, evidence):
    if not all(isinstance(x, str) and x.strip() for x in (request_id, query)):
        raise ValueError("nonempty request ID and query required")
    if len({c.chunk_id for c in evidence}) != len(evidence):
        raise ValueError("duplicate evidence ID")
    return {"contract_version": 1, "request_id": request_id, "query": query,
            "context_sha256": context_hash(evidence),
            "evidence": [asdict(c) for c in evidence]}


def validate_response(response, request, evidence, snapshots, allowed_sources):
    # request/evidence/permissions are retained by the trusted application.
    if type(response) is not dict or set(response) != {
        "contract_version", "request_id", "context_sha256", "status",
        "answer", "claims", "reason"
    }:
        raise ValueError("invalid response fields")
    if type(response["contract_version"]) is not int or response["contract_version"] != 1:
        raise ValueError("unsupported contract version")
    if request["context_sha256"] != context_hash(evidence):
        raise ValueError("retained context differs from request")
    if (response["request_id"], response["context_sha256"]) != (
            request["request_id"], request["context_sha256"]):
        raise ValueError("response bound to a different request")
    status = response["status"]
    if status not in ("completed", "insufficient_evidence", "failed", "truncated"):
        raise ValueError("unknown status")
    answer, claims, reason = response["answer"], response["claims"], response["reason"]
    if not isinstance(answer, str) or type(claims) is not list:
        raise ValueError("invalid answer or claims")
    if status != "completed":
        if claims or not isinstance(reason, str) or not reason.strip():
            raise ValueError("noncompleted response needs reason and no claims")
        return {"status": status, "eligible_for_answer_review": False,
                "checked_citations": 0, "semantic_support": "not_evaluated"}
    if reason is not None or not answer.strip() or not claims:
        raise ValueError("completed response requires answer and cited claims")
    count = 0
    for claim in claims:
        if type(claim) is not dict or set(claim) != {"start", "end", "citations"}:
            raise ValueError("invalid claim fields")
        start, end = claim["start"], claim["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(answer):
            raise ValueError("claim range outside answer")
        citations = claim["citations"]
        if type(citations) is not list or not citations:
            raise ValueError("claim needs citations")
        for citation in citations:
            if type(citation) is not dict or set(citation) != {
                "chunk_id", "source_id", "version", "start", "end", "quote"
            }:
                raise ValueError("invalid citation fields")
            if not all(isinstance(citation[key], str) and citation[key]
                       for key in ("chunk_id", "source_id", "version")):
                raise ValueError("invalid citation identity")
            check_quote(citation, evidence, snapshots, allowed_sources)
            count += 1
    return {"status": status, "eligible_for_answer_review": True,
            "checked_citations": count, "semantic_support": "not_evaluated"}
