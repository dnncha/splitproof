from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evaluate import SATISFIED, SATISFIED_UNPROVEN, VIOLATED

STATUS_WORD = {
    SATISFIED: "proven",
    SATISFIED_UNPROVEN: "no violation found (not proven)",
    VIOLATED: "VIOLATED",
}


def summary(certificate: Dict[str, Any]) -> str:
    status = certificate.get("status", "unknown")
    units = certificate.get("units", {})
    assignment = certificate.get("assignment", {})
    entities = certificate.get("dataset", {}).get("entities", 0)
    folds = ", ".join(
        f"{stat['name']}={stat['count']}" for stat in assignment.get("stats", [])
    )
    lines = [
        f"{entities} entities in {units.get('units', 0)} independent units; folds: {folds or 'none'}.",
        f"Constraint status: {STATUS_WORD.get(status, status)}.",
    ]
    worst = _worst_neighbour(certificate)
    if worst is not None:
        key, value, with_neighbour, queries = worst
        lines.append(
            f"Worst held-out similarity to the reference fold: {value} on {key} "
            f"({with_neighbour} of {queries} held-out items have any neighbour)."
        )
    missing = _max_unusable_count(certificate)
    if missing:
        entities_without_values = "1 entity" if missing == 1 else f"{missing} entities"
        lines.append(f"For at least one declared constraint, {entities_without_values} had no usable value.")
    if status != VIOLATED:
        lines.append(
            "That status covers only the constraints the policy declared. Read the numbers below "
            "before treating this split as hard."
        )
    for item in certificate.get("constraints", []):
        word = STATUS_WORD.get(item.get("status"), item.get("status"))
        lines.append(f"- {item.get('description')}: {word}.")
    largest = units.get("largest_unit_fraction")
    if isinstance(largest, (int, float)) and largest >= 0.2:
        lines.append(
            f"Warning: the largest single unit holds {largest:.1%} of the data, so fold sizes are "
            "constrained by homology, not by your requested fractions."
        )
    for key, entry in (certificate.get("diagnostics", {}).get("keys") or {}).items():
        baseline = entry.get("nearest_neighbour_baseline") or {}
        if baseline.get("value") is not None:
            lines.append(
                f"Similarity baseline on {key}: {baseline['metric']}={baseline['value']} over "
                f"{baseline['compared']} held-out items. Report any model score next to this one."
            )
        distribution = entry.get("nearest_reference_similarity") or {}
        if distribution:
            lines.append(
                f"Held-out items' nearest reference similarity on {key}: median {distribution.get('p50')}, "
                f"p99 {distribution.get('p99')}, max {distribution.get('max')}."
            )
    return "\n".join(lines)


def _max_unusable_count(certificate: Dict[str, Any]) -> int:
    total = 0
    for item in certificate.get("constraints", []):
        measured = item.get("measured") or {}
        here = (
            int(measured.get("values_absent") or 0)
            + int(measured.get("values_unparsable") or 0)
            + int(measured.get("missing_values") or 0)
            + int(measured.get("items_without_a_date") or 0)
        )
        total = max(total, here)
    return total


def _worst_neighbour(certificate: Dict[str, Any]) -> Optional[Tuple[str, float, Optional[int], Optional[int]]]:
    best = None
    for key, entry in (certificate.get("diagnostics", {}).get("keys") or {}).items():
        distribution = entry.get("nearest_reference_similarity") or {}
        value = distribution.get("max")
        if value is None:
            continue
        candidate = (key, value, entry.get("with_a_neighbour"), entry.get("queries"))
        if best is None or value > best[1]:
            best = candidate
    return best


def markdown(certificate: Dict[str, Any]) -> str:
    status = certificate.get("status", "unknown")
    dataset = certificate.get("dataset", {})
    units = certificate.get("units", {})
    assignment = certificate.get("assignment", {})
    out: List[str] = []
    out.append("# splitproof report")
    out.append("")
    out.append(f"**Status: {STATUS_WORD.get(status, status)}**")
    out.append("")
    out.append(f"- entities: {dataset.get('entities')}")
    out.append(f"- data digest: `{dataset.get('digest')}`")
    out.append(f"- policy digest: `{certificate.get('policy_digest')}`")
    out.append(f"- assignment digest: `{assignment.get('digest')}`")
    out.append(f"- splitproof {certificate.get('splitproof', {}).get('version')}, seed {certificate.get('seed')}")
    out.append("")

    out.append("## Folds")
    out.append("")
    out.append("| fold | entities | target | drift | units |")
    out.append("| --- | --- | --- | --- | --- |")
    for stat in assignment.get("stats", []):
        drift = stat.get("drift")
        drift_text = f"{drift:+.1%}" if isinstance(drift, (int, float)) else "n/a"
        out.append(
            f"| {stat['name']} | {stat['count']} | {stat['target']} | {drift_text} | {stat['units']} |"
        )
    if assignment.get("dropped"):
        out.append("")
        out.append(f"{assignment['dropped']} entities were dropped by policy.")
    out.append("")

    out.append("## Constraints")
    out.append("")
    for item in certificate.get("constraints", []):
        out.append(f"### {item.get('id')}")
        out.append("")
        out.append(f"{item.get('description')} — **{STATUS_WORD.get(item.get('status'), item.get('status'))}**")
        out.append("")
        for name, value in (item.get("measured") or {}).items():
            out.append(f"- {name}: {value}")
        for note in item.get("notes") or []:
            out.append(f"- note: {note}")
        examples = item.get("examples") or []
        if examples:
            out.append("")
            out.append("Worst offenders:")
            out.append("")
            for example in examples[:10]:
                out.append(f"- `{example}`")
        out.append("")

    out.append("## Independent units")
    out.append("")
    out.append(f"- units: {units.get('units')}")
    out.append(f"- largest unit: {units.get('largest_unit')} ({_pct(units.get('largest_unit_fraction'))})")
    out.append(f"- singleton units: {units.get('singleton_units')}")
    out.append(f"- size percentiles: {units.get('size_percentiles')}")
    out.append(f"- similarity exactness: {units.get('exactness')}")
    if units.get("spanning_units"):
        out.append(f"- units spanning the temporal cutoff: {units['spanning_units']}")
    out.append("")

    diagnostics = certificate.get("diagnostics", {})
    keys = diagnostics.get("keys") or {}
    if keys:
        out.append("## Leakage diagnostics")
        out.append("")
        out.append(f"Reference fold: `{diagnostics.get('reference_fold')}`")
        out.append("")
        for key, entry in keys.items():
            out.append(f"### {key}")
            out.append("")
            out.append(f"- metric: {entry.get('metric')} via {entry.get('backend')}")
            out.append(f"- held-out items compared: {entry.get('queries')}")
            out.append(f"- nearest reference similarity: {entry.get('nearest_reference_similarity')}")
            out.append(f"- at or above the declared threshold: {entry.get('at_or_above_threshold')}")
            baseline = entry.get("nearest_neighbour_baseline") or {}
            if baseline:
                out.append(
                    f"- nearest-neighbour baseline: {baseline.get('metric')}="
                    f"{baseline.get('value')} over {baseline.get('compared')} items"
                )
                if baseline.get("note"):
                    out.append(f"- {baseline['note']}")
            out.append("")

    out.append("## What this does and does not prove")
    out.append("")
    out.append(
        "splitproof proves exactly the constraints the policy declares, measured by the named backend. "
        "A `proven` status means no pair at or above the declared threshold crosses folds, and the "
        "backend searched every pair. A `no violation found` status means the backend is sensitive but "
        "heuristic, so absence of a crossing pair is evidence, not proof. splitproof sets no thresholds of "
        "its own and passes no judgement on constraints you did not declare."
    )
    out.append("")
    return "\n".join(out)


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"


def text_checks(checks: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for check in checks:
        mark = "ok  " if check.get("ok") else "FAIL"
        lines.append(f"{mark} {check.get('name')}: {check.get('describes')}")
        if not check.get("ok") and check.get("difference"):
            lines.append(f"       {check['difference']}")
    return "\n".join(lines)
