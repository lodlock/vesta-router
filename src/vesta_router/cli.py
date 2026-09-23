"""``python -m vesta_router`` — the fast CI tier, runnable by hand.

Everything here runs with the standard library alone and touches no GPU, no
network and no model. It is the whole of the fast tier: if this is green, a pull
request has not broken the corpus, the schema or a manifest.

    python -m vesta_router validate            # corpora + schema + disjointness
    python -m vesta_router render <case-id>    # show what the model would see
    python -m vesta_router manifest <path>     # validate one manifest
    python -m vesta_router verify <manifest> <artifact>
    python -m vesta_router gate <eval-report>  # apply promotion thresholds
    python -m vesta_router failures <cases.jsonl>            # group a run by failure family
    python -m vesta_router regressions <before> <after>      # diff two runs, per case
    python -m vesta_router traindata --out <path.jsonl>      # render the training corpus
    python -m vesta_router inspect <a.cact> [<b.cact> --compare]  # what is in an archive
    python -m vesta_router compare-runs <summary.json>...    # N runs, one metric table

One command is the exception, and it is marked as one:

    python -m vesta_router eval --model <artifact.cact>

`eval` loads a real runtime and runs a real model. It is not part of the fast
tier, it is not run in pull-request CI, and it is the only thing here that
imports anything outside the standard library. Everything it needs to be right
about — classification, scoring, aggregation, report serialization — is tested
without it, against recorded envelopes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import check_disjoint, load_corpus, validate_corpus
from .gates import apply_gates, load_gates, promotable
from .manifest import validate_manifest, verify_artifact
from .render import render_example
from .schema import load_tool_schema
from .severity import TOOL_SEVERITY, schema_coverage

REPO = Path(__file__).resolve().parents[2]
ACTIVE_SCHEMA = REPO / "tools" / "tool-schema-v2.json"
TRAIN_DIR = REPO / "data" / "train"
EVAL_DIR = REPO / "data" / "eval"
THRESHOLDS = REPO / "eval" / "thresholds.json"
REPORTS_DIR = REPO / "reports" / "baseline"


def _report(problems, header: str) -> int:
    if not problems:
        print(f"  {header}: ok")
        return 0
    print(f"  {header}: {len(problems)} problem(s)")
    for problem in problems:
        print(f"    {problem}")
    return len(problems)


def cmd_validate(_args: argparse.Namespace) -> int:
    schema = load_tool_schema(ACTIVE_SCHEMA)
    print(f"tool schema v{schema.version} ({len(schema.tools)} tools): {', '.join(schema.tool_names)}")

    failures = 0

    train, train_load = load_corpus(TRAIN_DIR)
    evaluation, eval_load = load_corpus(EVAL_DIR)
    print(f"\ntraining corpus: {len(train)} cases")
    failures += _report(train_load, "parse")
    failures += _report(validate_corpus(train, schema), "schema")

    print(f"\neval corpus: {len(evaluation)} cases")
    failures += _report(eval_load, "parse")
    failures += _report(validate_corpus(evaluation, schema), "schema")

    print("\ndisjointness")
    failures += _report(check_disjoint(train, evaluation), "train vs eval")

    print("\nrendering")
    # Determinism is not an abstract property here: it is checked by rendering
    # every case twice and comparing, which catches any accidental dependence on
    # dict ordering or set iteration.
    render_problems = []
    for case in train + evaluation:
        first = render_example(case.utterance, case.state, schema)
        second = render_example(case.utterance, case.state, schema)
        if first != second:
            render_problems.append(f"{case.file}:{case.line} [{case.id}] renders differently on repeat")
    failures += _report(render_problems, "determinism")

    # Both directions. A schema tool with no severity is a hole in the safety
    # model that scores it at the worst tier without saying so; a severity for
    # a tool the schema does not declare is a rule about something the app
    # cannot dispatch.
    print("\nfalse-action severity")
    coverage = schema_coverage(schema.tool_names)
    severity_problems = [f"{name}: no severity assigned" for name in coverage["unclassified"]]
    severity_problems += [
        f"{name}: has a severity but is not in the tool schema"
        for name in coverage["classifiedButNotInSchema"]
    ]
    failures += _report(severity_problems, f"coverage ({len(TOOL_SEVERITY)} tools classified)")

    print(f"\n{'FAILED' if failures else 'PASSED'} - {failures} problem(s)")
    return 1 if failures else 0


def cmd_render(args: argparse.Namespace) -> int:
    schema = load_tool_schema(ACTIVE_SCHEMA)
    cases = load_corpus(TRAIN_DIR)[0] + load_corpus(EVAL_DIR)[0]
    for case in cases:
        if case.id == args.case_id:
            print(render_example(case.utterance, case.state, schema))
            return 0
    print(f"no case with id {args.case_id!r}", file=sys.stderr)
    return 1


def cmd_manifest(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
    problems = validate_manifest(doc)
    if not problems:
        print(f"{args.path}: valid")
        return 0
    for problem in problems:
        print(f"{args.path}: {problem}", file=sys.stderr)
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    problems = verify_artifact(args.manifest, args.artifact)
    if not problems:
        print(f"{args.artifact}: matches {args.manifest}")
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1


def cmd_gate(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    gates = load_gates(THRESHOLDS)
    results = apply_gates(metrics, gates)
    for result in results:
        print(result)
    ok = promotable(results)
    print(f"\n{'PROMOTABLE' if ok else 'BLOCKED'}")
    if any(g.gate.provisional for g in results):
        print("note: some thresholds are provisional; see eval/thresholds.json")
    return 0 if ok else 1


def _emit(document: dict, out: str | None) -> None:
    """Print a machine-readable document, or write it where it was asked for.

    Sorted keys and a trailing newline, for the same reason the eval report
    serializes that way: two runs of the same analysis must diff to nothing.
    """
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text, end="")


def cmd_failures(args: argparse.Namespace) -> int:
    from .failures import analyze, load_cases

    document = analyze(load_cases(args.cases))
    _emit(document, args.output)
    if args.output:
        for family in document["families"]:
            if family["count"]:
                mark = "SAFETY" if family["safety"] else "      "
                print(f"  {mark} {family['family']:<34} {family['count']}")
    return 0


def cmd_regressions(args: argparse.Namespace) -> int:
    from .failures import compare, load_cases

    document = compare(load_cases(args.before), load_cases(args.after))
    _emit(document, args.output)
    if args.output:
        print(
            f"  improved {len(document['improved'])}  regressed {len(document['regressed'])}  "
            f"changed {len(document['changed'])}  unchanged failures {len(document['unchangedFailures'])}"
        )
        print(f"  verdict  {document['verdict']}")
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    """Put several eval summaries in one metric table.

    Two runs can only ever say "these differ". Three — a published artifact, a
    locally re-exported control built from the same checkpoint, and a trained
    candidate — can say WHICH of the build path and the training the difference
    belongs to, which is the question two runs structurally cannot answer.
    """
    from .failures import compare_summaries

    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summary]
    document = compare_summaries(summaries)
    _emit(document, args.output)
    if args.output:
        labels = [run["label"] for run in document["runs"]]
        if not document["comparable"]:
            print("  NOT COMPARABLE:")
            for problem in document["incomparabilities"]:
                print(f"    {problem['field']}: {problem['reason']}")
        print(f"  runs     {len(labels)}")
        print(f"  identical across runs  {len(document['metricsIdenticalAcrossRuns'])}")
        print(f"  differing              {len(document['metricsDiffering'])}")
        for name in document["metricsDiffering"]:
            row = next(r for r in document["metrics"] if r["metric"] == name)
            rendered = "  ".join(f"{label}={row['values'][label]}" for label in labels)
            print(f"    {name:<34} {rendered}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Read a ``.cact``'s header and tensor directory, or diff two of them.

    Part of the fast tier despite naming an artifact: it reads a file, it loads
    no runtime and it imports nothing outside the standard library. What it
    answers is what quantization and what tensor set an archive actually holds,
    which is the question the 35.3 MB / 63.4 MB discrepancy turned out to be.
    """
    from .artifact import compare_artifacts, inspect_artifact, render_text

    try:
        if len(args.artifact) == 2 and args.compare:
            _emit(compare_artifacts(args.artifact[0], args.artifact[1]), args.output)
            return 0
        if args.compare:
            print("--compare needs exactly two artifacts", file=sys.stderr)
            return 1
        documents = [inspect_artifact(path) for path in args.artifact]
    except (OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.output or args.json:
        _emit(documents[0] if len(documents) == 1 else {"artifacts": documents}, args.output)
        return 0
    for document in documents:
        print(render_text(document))
    return 0


def cmd_traindata(args: argparse.Namespace) -> int:
    """Render data/train into the JSONL the Needle trainer reads.

    Refuses on an invalid or non-disjoint corpus for the same reason ``eval``
    refuses on one: a model trained from cases that do not validate is a model
    nobody can say what was asked of, and an eval utterance that reached
    training turns the regression corpus into a memorization test.
    """
    from .engines import tool_definitions
    from .engines.needle import NEEDLE_SYSTEM_PROMPT
    from .training import corpus_digest, write_training_jsonl

    schema = load_tool_schema(ACTIVE_SCHEMA)
    train, train_load = load_corpus(TRAIN_DIR)
    evaluation, eval_load = load_corpus(EVAL_DIR)
    problems = (
        train_load
        + eval_load
        + validate_corpus(train, schema)
        + validate_corpus(evaluation, schema)
        + check_disjoint(train, evaluation)
    )
    if problems:
        print(f"{len(problems)} corpus problem(s); refusing to render training data", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    tools = tool_definitions(ACTIVE_SCHEMA)
    summary = write_training_jsonl(args.out, train, schema, tools, system_prompt=NEEDLE_SYSTEM_PROMPT)
    print(f"train corpus   {len(train)} cases from {TRAIN_DIR}")
    print(f"               content sha256 {corpus_digest(TRAIN_DIR)['sha256']}")
    print(f"rendered       {summary['examples']} examples -> {args.out}")
    print(f"               sha256 {summary['sha256']}  {summary['sizeBytes']} bytes")
    for kind, count in sorted(summary["byOutcome"].items()):
        print(f"  {kind:<12} {count}")
    return 0


def _select_cases(cases, wanted, limit):
    """Narrow the corpus for a debugging run, in the corpus's own order.

    ``--case`` is exact-id and repeatable, because chasing one failure should
    not mean running sixty-one other cases first. ``--limit`` takes a prefix.
    Either one makes the run PARTIAL, and the report says so in a field rather
    than in a comment — a partial run's rates are rates over what it ran.
    """
    if wanted:
        by_id = {case.id: case for case in cases}
        unknown = [name for name in wanted if name not in by_id]
        if unknown:
            raise KeyError(", ".join(unknown))
        return [case for case in cases if case.id in set(wanted)]
    if limit is not None:
        return cases[:limit]
    return cases


def _print_case(result) -> None:
    mark = {
        "tool_call": "call",
        "no_call": "none",
        "suppressed_only": "held",
        "malformed": "MALF",
        "engine_error": "ERR ",
    }[result.raw_outcome]
    executable = "EXEC" if result.executable_action else "    "
    print(f"  {mark} {executable}  {result.case_id:<32} {result.utterance[:46]!r}")
    for call in result.response.function_calls:
        print(f"        -> {call.name} {json.dumps(call.arguments, sort_keys=True)}")
    for call in result.response.suppressed_calls:
        print(f"        ~  {call.name} {json.dumps(call.arguments, sort_keys=True)} (engine withheld)")
    if result.response.error:
        print(f"        !! {result.response.error}")


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate one artifact against the live eval corpus.

    Refuses to score an invalid corpus. A metric computed over cases that do
    not validate is a number about nothing, and it would be indistinguishable
    from a real one once it is in a report.
    """
    from .engines import tool_definitions
    from .engines.needle import NEEDLE_SYSTEM_PROMPT, NeedleEngine, NeedleUnavailable
    from .runner import baseline_identifier, run_evaluation, write_report

    schema = load_tool_schema(ACTIVE_SCHEMA)
    cases_dir = Path(args.cases) if args.cases else EVAL_DIR
    cases, load_problems = load_corpus(cases_dir)
    problems = load_problems + validate_corpus(cases, schema)
    if problems:
        print(f"{cases_dir}: {len(problems)} corpus problem(s); refusing to evaluate", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if not cases:
        print(f"{cases_dir}: no cases", file=sys.stderr)
        return 1

    try:
        selected = _select_cases(cases, args.case, args.limit)
    except KeyError as exc:
        print(f"no case with id {exc.args[0]}", file=sys.stderr)
        return 1
    partial = len(selected) != len(cases)

    system_prompt = NEEDLE_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    tools = tool_definitions(ACTIVE_SCHEMA)

    def factory():
        return NeedleEngine(
            artifact=args.model,
            tools=tools,
            system_prompt=system_prompt,
            needle_path=args.needle_path,
            auto_date=args.auto_date,
        )

    try:
        probe = factory()
    except NeedleUnavailable as exc:
        # Stop, do not substitute. A report produced without the artifact is a
        # number about something else.
        print(f"cannot evaluate {args.model}: {exc}", file=sys.stderr)
        return 2
    description = probe.describe()
    probe.close()

    baseline_id = baseline_identifier(args.model, description["artifactSha256"])
    print(f"artifact   {description['artifactPath']}")
    print(f"           sha256 {description['artifactSha256']}")
    print(f"           {description['artifactSizeBytes']} bytes")
    print(f"runtime    {description['engine']} {description['engineVersion']}")
    print(f"baseline   {baseline_id}")
    print(f"corpus     {len(selected)} case(s) from {cases_dir}" + (" (PARTIAL)" if partial else ""))
    print(f"schema     v{schema.version}\n")

    outcome = run_evaluation(
        factory,
        selected,
        schema,
        isolation_check=not args.no_isolation_check,
        on_case=_print_case if args.verbose else None,
    )

    notes = [
        "Baseline of an UNTRAINED artifact. This is a benchmark result, not a promotion.",
        "Post-guardrail metrics use vesta_router.grounding, an evaluator-side approximation "
        "of Vesta's deterministic screen — not the app's own code.",
    ]
    if partial:
        notes.append("PARTIAL RUN: --case or --limit was used, so every rate is over a subset.")
    if args.no_isolation_check:
        notes.append(
            "Isolation pass skipped: stateLeakCount and nondeterministicCaseCount are "
            "reported as 0 because nothing looked, not because nothing was found."
        )
    if args.auto_date:
        notes.append(
            "auto_date was ON: the runtime prefixed a date fact to the system prompt, "
            "so this run is not comparable with a default one."
        )

    directory = write_report(
        args.output or REPORTS_DIR,
        outcome,
        schema,
        Path(ACTIVE_SCHEMA),
        sorted(cases_dir.glob("*.jsonl")),
        baseline_id,
        notes=notes,
        limited=partial,
    )

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    print("\nmetrics")
    for key in sorted(summary["metrics"]):
        print(f"  {key:<32} {summary['metrics'][key]}")
    print("\nperformance")
    for key in sorted(summary["performance"]):
        print(f"  {key:<32} {summary['performance'][key]}")
    print(f"\nreport     {directory}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vesta_router", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="validate the corpora against the active tool schema")

    render = sub.add_parser("render", help="print what the model would see for one case")
    render.add_argument("case_id")

    manifest = sub.add_parser("manifest", help="validate a release manifest")
    manifest.add_argument("path")

    verify = sub.add_parser("verify", help="check an artifact against its manifest")
    verify.add_argument("manifest")
    verify.add_argument("artifact")

    gate = sub.add_parser("gate", help="apply promotion thresholds to an eval report")
    gate.add_argument("report")

    failures = sub.add_parser("failures", help="group one eval report's cases.jsonl by failure family")
    failures.add_argument("cases", help="path to a report's cases.jsonl")
    failures.add_argument("--output", help="write the JSON here instead of to stdout")

    regressions = sub.add_parser("regressions", help="diff two eval reports, per case and per family")
    regressions.add_argument("before", help="baseline cases.jsonl")
    regressions.add_argument("after", help="candidate cases.jsonl")
    regressions.add_argument("--output", help="write the JSON here instead of to stdout")

    traindata = sub.add_parser("traindata", help="render data/train into the trainer's JSONL")
    traindata.add_argument("--out", required=True, help="output .jsonl path")

    compare_runs = sub.add_parser("compare-runs", help="put several eval summaries in one metric table")
    compare_runs.add_argument("summary", nargs="+", help="two or more report summary.json paths")
    compare_runs.add_argument("--output", help="write the JSON here instead of to stdout")

    inspect = sub.add_parser("inspect", help="read a .cact archive's header and tensor directory")
    inspect.add_argument("artifact", nargs="+", help="one .cact to describe, or two with --compare")
    inspect.add_argument("--compare", action="store_true", help="diff two archives and decompose the size")
    inspect.add_argument("--json", action="store_true", help="machine-readable output")
    inspect.add_argument("--output", help="write the JSON here instead of to stdout")

    evaluate = sub.add_parser(
        "eval",
        help="run the eval corpus against a .cact artifact (loads a real runtime)",
    )
    evaluate.add_argument("--model", required=True, help="path to the .cact artifact to evaluate")
    evaluate.add_argument("--cases", help=f"corpus directory (default: {EVAL_DIR})")
    evaluate.add_argument("--output", help=f"report root (default: {REPORTS_DIR})")
    evaluate.add_argument("--limit", type=int, help="evaluate only the first N cases")
    evaluate.add_argument("--case", action="append", default=[], help="evaluate one case id; repeatable")
    evaluate.add_argument("--verbose", action="store_true", help="print each case as it runs")
    evaluate.add_argument(
        "--no-isolation-check",
        action="store_true",
        help="skip the second pass that verifies reset held between cases",
    )
    evaluate.add_argument(
        "--needle-path", help="a local cactus-needle checkout to import instead of an installed one"
    )
    evaluate.add_argument("--system-prompt-file", help="override the system prompt (recorded in the report)")
    evaluate.add_argument(
        "--auto-date",
        action="store_true",
        help="let the runtime prefix a date fact to the system prompt; off by default, "
        "because Vesta owns time and a daily-changing prefix makes two runs incomparable",
    )

    args = parser.parse_args(argv)
    return {
        "validate": cmd_validate,
        "render": cmd_render,
        "manifest": cmd_manifest,
        "verify": cmd_verify,
        "gate": cmd_gate,
        "eval": cmd_eval,
        "failures": cmd_failures,
        "regressions": cmd_regressions,
        "traindata": cmd_traindata,
        "inspect": cmd_inspect,
        "compare-runs": cmd_compare_runs,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
