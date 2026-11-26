"""Trusted independent scoring. Only inputs go to candidate execution."""
from dataclasses import dataclass, asdict
import hashlib
import json

from .value_codec import decode, equal


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass
class Score:
    reward: float | None
    reason: str
    cases: list

    def to_dict(self):
        return asdict(self)


class IndependentScorer:
    def __init__(self, runner):
        self.runner = runner

    def score(self, task, code, public=False):
        if code is None:
            return Score(0.0, "no_submission", [])
        cases = task["public_cases" if public else "score_cases"]
        if not cases:
            return Score(None, "empty_case_manifest", [])
        records = []
        for case in cases:
            execution = self.runner.run(code, task["function"], case["args"], case["kwargs"])
            entry = {"case_id": case["case_id"], "input_hash": fingerprint([case["args"],case["kwargs"]]),
                     "execution": execution.to_dict()}
            if execution.status == "infra_error":
                entry["passed"] = None
                records.append(entry)
                return Score(None, "infrastructure", records)
            try:
                passed = execution.status == "returned" and equal(execution.value, case["expected"])
            except (ValueError, TypeError, RecursionError):
                passed = False
            entry["passed"] = passed
            # Controller stores expected values; never put them into runner.run.
            entry["expected"] = case["expected"]
            records.append(entry)
        ok = all(x["passed"] for x in records)
        return Score(float(ok), "all_passed" if ok else "candidate_failure", records)


def public_feedback(score):
    if score.reward is None:
        return {"status": "infrastructure_error", "error": score.reason}
    def preview(value):
        if value is None:
            return None
        decoded = repr(decode(value))
        return decoded if len(json.dumps(decoded)) <= 100 else decoded[:60] + '…'

    brief = []
    for row in score.cases:
        e = row["execution"]
        brief.append({"case": row["case_id"], "passed": row["passed"], "status": e["status"],
                      "actual": preview(e["value"]), "expected": preview(row["expected"]),
                      "error": e["error"][:120]})
    # The full results stay in the private trace. Bound the observation before
    # templating. Only public cases may appear here.
    result = {"status": "ok", "passed": score.reward == 1, "cases": []}
    for case in brief:
        result['cases'].append(case)
        if len(json.dumps(result, ensure_ascii=True)) > 450:
            result['cases'].pop()
            result['omitted_cases'] = len(brief) - len(result['cases'])
            break
    return result
