from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__, api, cert as cert_module, dataset as dataset_module, report as report_module
from .errors import SplitproofError, InfeasibleSplit
from .evaluate import SATISFIED_UNPROVEN, VIOLATED
from .policy import Policy

EXIT_OK = 0
EXIT_VIOLATED = 1
EXIT_ERROR = 2
EXIT_INFEASIBLE = 3

PROTEIN_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_LETTERS = set("BXZUO*-")
NUCLEOTIDE_LETTERS = set("ACGTUN-")
ID_HINTS = ("id", "accession", "uniprot", "name", "identifier", "key", "entry")
SEQUENCE_HINTS = ("seq", "sequence", "protein", "aa", "peptide", "dna", "rna")
MOLECULE_HINTS = ("smiles", "smile", "canonical_smiles", "structure")
DATE_HINTS = ("date", "deposited", "released", "published", "year", "created", "collected")
GROUP_HINTS = ("donor", "patient", "subject", "cell_line", "cellline", "plate", "site", "batch", "run", "study", "lab", "organism", "species", "source")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_ERROR
    try:
        return args.handler(args)
    except InfeasibleSplit as error:
        _emit_error(error, args)
        return EXIT_INFEASIBLE
    except SplitproofError as error:
        _emit_error(error, args)
        return EXIT_ERROR
    except BrokenPipeError:
        return EXIT_OK


def _emit_error(error: SplitproofError, args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        json.dump({"ok": False, "error": error.to_dict()}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    print(f"splitproof: {error.code}: {error.message}", file=sys.stderr)
    if error.remedy:
        print(f"  {error.remedy}", file=sys.stderr)
    for name, value in sorted(error.details.items()):
        print(f"  {name}: {value}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitproof",
        description="Build, audit and prove leakage-free train/test splits for biological data.",
    )
    parser.add_argument("--version", action="version", version=f"splitproof {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")
    common.add_argument("--data", type=Path, default=None, help="data file, overriding dataset.path")
    common.add_argument(
        "--exact-max-n",
        type=int,
        default=api.DEFAULT_EXACT_MAX_N,
        help="above this many entities, fall back to approximate similarity search",
    )
    common.add_argument(
        "--backend",
        action="append",
        default=[],
        metavar="KEY=BACKEND",
        help="force a similarity backend for a key, e.g. sequence=exact_jaccard",
    )
    common.add_argument(
        "--no-neighbours",
        action="store_true",
        help="skip the nearest-neighbour diagnostics (faster, less informative)",
    )
    common.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="refuse rather than exhaust memory when a threshold relates this many pairs",
    )
    common.add_argument(
        "--require-exact",
        action="store_true",
        help="fail unless every constraint was proven by an exhaustive backend, not merely unviolated",
    )

    split = subparsers.add_parser(
        "split", parents=[common], help="build a split and write a certificate proving it"
    )
    split.add_argument("policy", type=Path)
    split.add_argument("--out", type=Path, default=Path("split.csv"), help="where to write id,fold")
    split.add_argument("--cert", type=Path, default=Path("splitproof.cert.json"))
    split.add_argument("--report", type=Path, default=None, help="also write a Markdown report here")
    split.add_argument(
        "--tolerance",
        type=float,
        default=api.DEFAULT_TOLERANCE,
        help=(
            "how far the largest unit may exceed a fold's requested size before the split is "
            "refused as infeasible; final fold drift is reported, never refused"
        ),
    )
    split.set_defaults(handler=_cmd_split)

    audit = subparsers.add_parser(
        "audit", parents=[common], help="measure leakage in a split somebody else made"
    )
    audit.add_argument("policy", type=Path)
    audit.add_argument("--split-column", default=None, help="column in the data file holding the fold")
    audit.add_argument("--split", type=Path, default=None, help="two-column id,fold file")
    audit.add_argument("--cert", type=Path, default=None, help="write the measurements as a certificate")
    audit.add_argument("--report", type=Path, default=None, help="write a Markdown report here")
    audit.set_defaults(handler=_cmd_audit)

    verify = subparsers.add_parser(
        "verify", parents=[common], help="re-prove a certificate against data and a split"
    )
    verify.add_argument("cert", type=Path)
    verify.add_argument("--split", type=Path, default=None)
    verify.add_argument("--split-column", default=None)
    verify.set_defaults(handler=_cmd_verify)

    units = subparsers.add_parser(
        "units", parents=[common], help="inspect the independent units a policy implies"
    )
    units.add_argument("policy", type=Path)
    units.set_defaults(handler=_cmd_units)

    init = subparsers.add_parser("init", help="write a starter policy by inspecting a data file")
    init.add_argument("data", type=Path)
    init.add_argument("--out", type=Path, default=Path("policy.yaml"))
    init.add_argument("--id", default=None, help="id column, if splitproof should not guess")
    init.add_argument("--json", action="store_true")
    init.set_defaults(handler=_cmd_init)

    mcp = subparsers.add_parser(
        "mcp", help="serve splitproof to an AI agent as an MCP server over stdio"
    )
    mcp.set_defaults(handler=_cmd_mcp)

    explain = subparsers.add_parser("explain", help="print a certificate in plain language")
    explain.add_argument("cert", type=Path)
    explain.add_argument("--markdown", action="store_true")
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(handler=_cmd_explain)

    return parser


def _pair_budget(args: argparse.Namespace) -> Dict[str, int]:
    value = getattr(args, "max_pairs", None)
    return {} if value is None else {"max_pairs": value}


def _overrides(values: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SplitproofError(
                f"--backend expects KEY=BACKEND, got {item!r}",
                remedy="For example: --backend sequence=exact_jaccard",
            )
        key, backend = item.split("=", 1)
        out[key.strip()] = backend.strip()
    return out


def _cmd_split(args: argparse.Namespace) -> int:
    policy = Policy.load(args.policy)
    outcome = api.make_split(
        policy,
        data_path=args.data,
        tolerance=args.tolerance,
        exact_max_n=args.exact_max_n,
        **_pair_budget(args),
        skip_neighbours=args.no_neighbours,
        backend_overrides=_overrides(args.backend),
    )
    api.write_split(args.out, outcome.data, outcome.assignment, policy.dataset.id_column)
    cert_module.write(args.cert, outcome.certificate)
    if args.report:
        Path(args.report).write_text(report_module.markdown(outcome.certificate), encoding="utf-8")
    payload = {
        "ok": outcome.ok,
        "status": outcome.certificate["status"],
        "summary": report_module.summary(outcome.certificate),
        "split": str(args.out),
        "certificate": str(args.cert),
        "report": str(args.report) if args.report else None,
        "folds": outcome.certificate["assignment"]["stats"],
        "units": outcome.certificate["units"],
        "constraints": outcome.certificate["constraints"],
        "diagnostics": outcome.certificate["diagnostics"],
    }
    _render(args, payload, outcome.certificate)
    return _outcome_exit(args, outcome.certificate, outcome.ok)


def _cmd_audit(args: argparse.Namespace) -> int:
    policy = Policy.load(args.policy)
    outcome = api.audit(
        policy,
        data_path=args.data,
        split_column=args.split_column,
        split_path=args.split,
        exact_max_n=args.exact_max_n,
        **_pair_budget(args),
        skip_neighbours=args.no_neighbours,
        backend_overrides=_overrides(args.backend),
    )
    if args.cert:
        cert_module.write(args.cert, outcome.certificate)
    if args.report:
        Path(args.report).write_text(report_module.markdown(outcome.certificate), encoding="utf-8")
    payload = {
        "ok": outcome.ok,
        "status": outcome.certificate["status"],
        "summary": report_module.summary(outcome.certificate),
        "folds": outcome.certificate["assignment"]["stats"],
        "units": outcome.certificate["units"],
        "constraints": outcome.certificate["constraints"],
        "diagnostics": outcome.certificate["diagnostics"],
    }
    _render(args, payload, outcome.certificate)
    return _outcome_exit(args, outcome.certificate, outcome.ok)


def _cmd_verify(args: argparse.Namespace) -> int:
    result = api.verify(
        args.cert,
        data_path=args.data,
        split_path=args.split,
        split_column=args.split_column,
        exact_max_n=args.exact_max_n,
        skip_neighbours=args.no_neighbours,
        **_pair_budget(args),
        backend_overrides=_overrides(args.backend),
    )
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(report_module.text_checks(result.checks))
        print()
        print("verified" if result.ok else "NOT verified")
    if not result.ok:
        return EXIT_ERROR
    if result.status == VIOLATED:
        if not args.json:
            print(
                "the certificate verifies, and it records a violated constraint",
                file=sys.stderr,
            )
        return EXIT_VIOLATED
    if args.require_exact and result.status == SATISFIED_UNPROVEN:
        if not args.json:
            print(
                "the certificate verifies, but --require-exact was set and it is not proven "
                "exhaustively",
                file=sys.stderr,
            )
        return EXIT_VIOLATED
    return EXIT_OK


def _cmd_units(args: argparse.Namespace) -> int:
    from . import units as units_module

    policy = Policy.load(args.policy)
    data = dataset_module.load_for(policy, args.data)
    graph = units_module.build(
        policy,
        data,
        exact_max_n=args.exact_max_n,
        backend_overrides=_overrides(args.backend),
        **_pair_budget(args),
    )
    payload = graph.to_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for name, value in payload.items():
            print(f"{name}: {value}")
    return EXIT_OK


def _cmd_mcp(args: argparse.Namespace) -> int:
    from . import mcp as mcp_module

    return mcp_module.serve()


def _cmd_explain(args: argparse.Namespace) -> int:
    certificate = cert_module.read(args.cert)
    if args.json:
        json.dump(
            {"status": certificate.get("status"), "summary": report_module.summary(certificate)},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    elif args.markdown:
        print(report_module.markdown(certificate))
    else:
        print(report_module.summary(certificate))
    return EXIT_OK if certificate.get("status") != VIOLATED else EXIT_VIOLATED


def _render(args: argparse.Namespace, payload: Dict[str, Any], certificate: Dict[str, Any]) -> None:
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    print(report_module.summary(certificate))


def _outcome_exit(args: argparse.Namespace, certificate: Dict[str, Any], ok: bool) -> int:
    if not ok:
        return EXIT_VIOLATED
    if getattr(args, "require_exact", False) and certificate.get("status") == SATISFIED_UNPROVEN:
        unproven = [
            item["id"]
            for item in certificate.get("constraints", [])
            if item.get("status") == SATISFIED_UNPROVEN
        ]
        if not args.json:
            print(
                f"splitproof: --require-exact was set, but {unproven} were not proven exhaustively",
                file=sys.stderr,
            )
        return EXIT_VIOLATED
    return EXIT_OK


def _looks_like_sequence(values: Sequence[Optional[str]]) -> Optional[str]:
    present = [value for value in values if value][:200]
    if len(present) < 2:
        return None
    if min(len(value) for value in present) < 12:
        return None
    letters = set("".join(present[:50]).upper())
    if letters <= NUCLEOTIDE_LETTERS:
        return "dna"
    if letters <= PROTEIN_LETTERS | AMBIGUOUS_LETTERS and not letters <= AMBIGUOUS_LETTERS:
        return "protein"
    return None


def _is_unambiguous_sequence(values: Sequence[Optional[str]]) -> bool:
    """Whether values look like sequences without the column name saying so.

    English clinical terms are spelled from a subset of the amino-acid letters, so
    "ADENOCARCINOMA" and "MELANOCYTOMA" used to be classified as proteins. Requiring
    a long, varied alphabet keeps real words out.
    """
    present = [value for value in values if value][:200]
    if not present:
        return False
    if min(len(value) for value in present) < 40:
        return False
    alphabet = set("".join(present[:50]).upper())
    return len(alphabet & PROTEIN_LETTERS) >= 15


def _looks_like_molecule(name: str, values: Sequence[Optional[str]]) -> bool:
    if any(hint in name.lower() for hint in MOLECULE_HINTS):
        return True
    present = [value for value in values if value][:50]
    if not present:
        return False
    hits = sum(1 for value in present if re.search(r"[=#\[\]()]", value) and re.search(r"[CcNnOo]", value))
    return hits >= max(2, int(0.8 * len(present)))


def _looks_like_date(values: Sequence[Optional[str]]) -> bool:
    present = [value for value in values if value][:50]
    if len(present) < 2:
        return False
    for value in present:
        try:
            _dt.date.fromisoformat(str(value)[:10])
        except ValueError:
            return False
    return True


def guess_policy(path: Path, id_column: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Guess a starter policy from a data file, returning the policy and what it saw."""
    header_id = id_column
    if header_id is None:
        header_id = _guess_id(_probe_header(path))
    data = dataset_module.load(path, header_id)
    similarity: Dict[str, Any] = {}
    groups: List[str] = []
    dates: Dict[str, str] = {}
    label: Optional[str] = None
    for name, values in data.columns.items():
        if name == header_id:
            continue
        lowered = name.lower()
        kind = _looks_like_sequence(values)
        named_like_a_sequence = any(hint in lowered for hint in SEQUENCE_HINTS)
        if kind and (named_like_a_sequence or _is_unambiguous_sequence(values)):
            similarity.setdefault("sequence", {"column": name, "kind": kind})
            continue
        if _looks_like_molecule(name, values):
            similarity.setdefault("molecule", {"column": name, "kind": "molecule"})
            continue
        if _looks_like_date(values):
            dates.setdefault(lowered.replace(" ", "_"), name)
            continue
        distinct = len({value for value in values if value is not None})
        if any(hint in lowered for hint in GROUP_HINTS) and 1 < distinct < max(2, len(data) // 2):
            groups.append(name)
            continue
        if label is None and lowered in {"label", "y", "target", "activity", "class", "outcome", "value"}:
            label = name
    constraints: List[Dict[str, Any]] = []
    if "sequence" in similarity:
        constraints.append(
            {"kind": "similarity", "key": "sequence", "metric": "identity", "threshold": 0.3}
        )
    if "molecule" in similarity:
        constraints.append(
            {"kind": "similarity", "key": "molecule", "metric": "ring_systems"}
        )
    for group in groups:
        constraints.append({"kind": "disjoint", "key": group})
    if not constraints:
        raise SplitproofError(
            "could not recognise a sequence, molecule, group or date column to constrain",
            remedy=(
                "splitproof will not invent a constraint. Write the policy by hand: declare the "
                "column under dataset.similarity or dataset.groups, then add the constraint that "
                "matches your claim. See the README for a worked policy."
            ),
            details={"columns": sorted(data.columns), "id": header_id},
        )

    policy: Dict[str, Any] = {
        "splitproof": 1,
        "dataset": {
            "path": str(path),
            "id": header_id,
            "similarity": similarity,
            "groups": groups,
            "dates": dates,
        },
        "scheme": {"kind": "holdout", "fractions": {"train": 0.8, "test": 0.2}},
        "constraints": constraints,
        "seed": 7,
    }
    if label:
        policy["dataset"]["label"] = label
    seen = {
        "rows": len(data),
        "id": header_id,
        "similarity": sorted(similarity),
        "groups": list(groups),
        "dates": sorted(dates),
        "label": label,
        "needs_mmseqs": any(c.get("metric") == "identity" for c in constraints),
    }
    return policy, seen


def _cmd_init(args: argparse.Namespace) -> int:
    policy, seen = guess_policy(Path(args.data), args.id)
    _write_policy(Path(args.out), policy)
    if args.json:
        json.dump(
            {"ok": True, "policy": str(args.out), "guessed": policy, "seen": seen},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(
            f"wrote {args.out} for {seen['rows']} rows: id='{seen['id']}', "
            f"similarity={seen['similarity']}, groups={seen['groups']}, dates={seen['dates']}"
        )
        print("Review it: splitproof guessed these columns, and a guess is not a policy.")
        if seen["needs_mmseqs"]:
            print("The sequence constraint uses metric 'identity', which needs MMseqs2 installed.")
    return EXIT_OK


def _probe_header(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in {".fasta", ".fa", ".faa", ".fna"}:
        return ["id", "sequence"]
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    first = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return [name.strip() for name in first.split(delimiter)]


def _guess_id(header: Sequence[str]) -> str:
    for hint in ID_HINTS:
        for name in header:
            if name.lower() == hint:
                return name
    for hint in ID_HINTS:
        for name in header:
            if hint in name.lower():
                return name
    return header[0]


def _write_policy(path: Path, policy: Dict[str, Any]) -> None:
    try:
        import yaml

        text = yaml.safe_dump(policy, sort_keys=False, default_flow_style=False)
    except ModuleNotFoundError:
        text = json.dumps(policy, indent=2)
        path = path.with_suffix(".json")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
