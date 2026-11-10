"""Verify the finite bundle and run only its stdlib synthetic test scope."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
MODULES = (
    "egraph_lab",
    "local_neighborhood",
    "feedback_neighborhood",
    "region_cycle_filter",
    "region_cycle_scc_reference",
    "feedback_neighborhood_filtered",
    "feedback_neighborhood_lazy",
)
EXCLUDED = {
    "tests.test_egraph_lab.EGraphLabTests.test_milp_matches_exhaustive_oracle_with_cycles":
        "Optional SciPy/MILP solver is outside the synthetic stdlib scope.",
    "tests.test_egraph_lab.EGraphLabTests.test_milp_rejects_self_cycle_and_finds_joint_improvement":
        "Optional SciPy/MILP solver is outside the synthetic stdlib scope.",
    "tests.test_feedback_neighborhood_filtered.FilteredNeighborhoodTests.test_real_adder_initial_centers_include_256_attempt_rejected_region":
        "Archived physical-development data is outside this finite bundle.",
}


def verify_manifest():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest contains a non-relative file path")
        data = (ROOT / relative).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Integrity check failed: {item['path']}")
        expected.add(relative.as_posix())
    exempt = {"SOURCE_MANIFEST.json", *manifest["generated_report_paths"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not {"__pycache__", ".git", ".omx"}.intersection(path.relative_to(ROOT).parts)
    }
    if actual - exempt != expected:
        raise ValueError("Bundle file set differs from the manifest")
    return len(expected)


def verify_import_closure():
    local = {f"research.{name}" for name in MODULES}
    optional = []
    for directory in ("research", "tests"):
        for path in sorted((ROOT / directory).glob("*.py")):
            tree = ast.parse(path.read_bytes(), filename=path.relative_to(ROOT).as_posix())
            solver_nodes = {
                id(child)
                for function in ast.walk(tree)
                if isinstance(function, ast.FunctionDef) and function.name == "milp_extract"
                for child in ast.walk(function)
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        raise ValueError("Unexpected relative import in copied source")
                    imports = [node.module or ""]
                    if node.module == "research":
                        imports = [f"research.{alias.name}" for alias in node.names]
                else:
                    continue
                for name in imports:
                    top = name.split(".")[0]
                    if top == "research":
                        if name not in local:
                            raise ValueError(f"Missing local import: {name}")
                    elif top in {"numpy", "scipy"}:
                        if (path.relative_to(ROOT).as_posix() != "research/egraph_lab.py"
                                or id(node) not in solver_nodes):
                            raise ValueError("Unexpected optional dependency location")
                        optional.append(name)
                    elif top not in sys.stdlib_module_names and top != "__future__":
                        raise ValueError(f"Non-stdlib import in finite closure: {name}")
    return sorted(set(optional))


def cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from cases(test)
        else:
            yield test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="SYNTHETIC_SMOKE_RESULT.json")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("This finite artifact is validated with Python 3.12; use that version.")
    report_path = Path(args.report)
    if report_path.name != args.report or report_path.name != "SYNTHETIC_SMOKE_RESULT.json":
        raise SystemExit("The report must use the bundle-local SYNTHETIC_SMOKE_RESULT.json path.")
    sys.dont_write_bytecode = True
    verified_files = verify_manifest()
    optional_imports = verify_import_closure()
    observations = {"external_process_launch_attempts": 0, "network_attempts": 0,
                    "optional_solver_import_attempts": 0}

    def audit(event, arguments):
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
            observations["external_process_launch_attempts"] += 1
            raise RuntimeError("External processes are outside the synthetic smoke scope")
        if event.startswith("socket."):
            observations["network_attempts"] += 1
            raise RuntimeError("Network access is outside the synthetic smoke scope")
        if event == "import" and arguments[0].split(".")[0] in {"scipy", "numpy"}:
            observations["optional_solver_import_attempts"] += 1
            raise RuntimeError("Optional solver imports are outside the synthetic smoke scope")

    sys.addaudithook(audit)
    sys.path.insert(0, str(ROOT))
    loaded = []
    for name in MODULES:
        module = importlib.import_module(f"tests.test_{name}")
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError("A test module was imported from outside the bundle")
        loaded.extend(cases(unittest.defaultTestLoader.loadTestsFromModule(module)))
    all_ids = {test.id() for test in loaded}
    if not set(EXCLUDED).issubset(all_ids):
        raise RuntimeError("A declared scope exclusion no longer names a test")
    selected = [test for test in loaded if test.id() not in EXCLUDED]
    if len(selected) != 71:
        raise RuntimeError("The finite synthetic test inventory changed")
    for module_name, module in tuple(sys.modules.items()):
        if module_name.split(".")[0] in {"research", "tests"}:
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise RuntimeError("A local module was imported from outside the bundle")
    selected_ids = sorted(test.id() for test in selected)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=1).run(unittest.TestSuite(selected))
    report = {
        "scope": "synthetic structural behavior only; no physical QoR claims",
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "command": "python3.12 -I -B run_synthetic_smoke.py --report SYNTHETIC_SMOKE_RESULT.json",
        "manifest_files_verified": verified_files,
        "local_import_closure": "complete",
        "optional_imports_present_in_unmodified_source_but_not_executed": optional_imports,
        "discovered_tests": len(loaded),
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "explicitly_excluded": EXCLUDED,
        "test_ids": selected_ids,
        **observations,
        "success": result.wasSuccessful() and not result.skipped and not any(observations.values()),
    }
    (ROOT / report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest: {verified_files} files verified; import closure complete.")
    print(f"Synthetic tests: {result.testsRun} run, {report['passed']} passed, "
          f"{len(result.failures)} failures, {len(result.errors)} errors, {len(result.skipped)} skipped.")
    print(f"Explicit exclusions: {len(EXCLUDED)}; external process/network/solver attempts: "
          f"{sum(observations.values())}.")
    if not report["success"]:
        print(output.getvalue().replace(str(ROOT), "<bundle>"))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
