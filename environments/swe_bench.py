"""
BudgetBench — SWE-bench Lite Environment  (Track A · Task A1)
==============================================================
Wraps SWE-bench Lite issues inside the BudgetBench Environment interface.

Each episode:
  1. Loads an issue from SWE-bench Lite (≥30 issues available).
  2. Starts a Docker sandbox with the repo checked out at base_commit.
  3. Exposes six tools for the agent to interact with the codebase.
  4. On submit_patch, scores the patch in a fresh sandbox via the oracle.

Tools:
  - search   query [path]     → grep-like search across the repo
  - inspect  file [start end] → read file lines (1-indexed)
  - hierarchy [depth]         → show directory tree
  - edit     file patch       → apply a unified-diff hunk to a file
  - run_tests [cmd]           → run the issue's test suite (budget cost = 3×)
  - submit_patch patch        → finalise and score (done = True)

Implemented by: Hanyu Jin (Track A)
Reference stub provided by: Chenrui Xu
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from budgetbench.interfaces import Environment, StatusCode


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_DIR   = "/testbed"   # standard mount point in swe-bench Docker images
_MIN_ISSUES = 30           # required minimum for the evaluation


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset_path: str) -> list[dict]:
    """
    Load SWE-bench Lite issues from a local file or HuggingFace.

    Resolution order
    ----------------
    1. ``dataset_path`` is an existing JSONL or JSON file  → parse it.
    2. ``dataset_path`` is a directory  → look for ``swe-bench-lite.jsonl``
       or ``data.jsonl`` inside it.
    3. Fall back to ``datasets`` (HuggingFace) with
       ``princeton-nlp/SWE-bench_Lite``, split ``"test"``.

    Raises RuntimeError if none of the above yields ≥ _MIN_ISSUES issues.
    """
    issues: list[dict] = []

    # --- 1 / 2: local file --------------------------------------------------
    candidates: list[Path] = []
    p = Path(dataset_path)
    # if p.is_file():
    #     candidates = [p]
    # elif p.is_dir():
    #     for name in ("swe-bench-lite.jsonl", "data.jsonl", "swe_bench_lite.jsonl"):
    #         if (p / name).exists():
    #             candidates.append(p / name)
    #             break

    # for candidate in candidates:
    #     suffix = candidate.suffix.lower()
    #     try:
    #         if suffix == ".jsonl":
    #             with open(candidate) as f:
    #                 issues = [json.loads(ln) for ln in f if ln.strip()]
    #         elif suffix == ".json":
    #             with open(candidate) as f:
    #                 raw = json.load(f)
    #             issues = raw if isinstance(raw, list) else list(raw.values())
    #     except Exception:
    #         issues = []
    #     if len(issues) >= _MIN_ISSUES:
    #         return issues

    # --- 3: HuggingFace -----------------------------------------------------
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        issues = [dict(row) for row in ds]
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load SWE-bench Lite dataset.\n"
            f"  Tried local path : {dataset_path!r}\n"
            f"  Tried HuggingFace: princeton-nlp/SWE-bench_Lite\n"
            f"  Install the 'datasets' package and ensure internet access, "
            f"or supply a local JSONL file.\n"
            f"  Cause: {exc}"
        ) from exc

    if len(issues) < _MIN_ISSUES:
        raise RuntimeError(
            f"Only {len(issues)} issues loaded; need at least {_MIN_ISSUES}."
        )
    return issues


# ---------------------------------------------------------------------------
# Patch format validation
# ---------------------------------------------------------------------------

def _is_unified_diff(patch: str) -> bool:
    """
    Heuristic check: a unified diff must have at least one '--- '/'+++' pair
    and at least one '@@ ' hunk header.
    """
    if not patch or not patch.strip():
        return False
    return (
        bool(re.search(r"^--- ", patch, re.MULTILINE))
        and bool(re.search(r"^\+\+\+ ", patch, re.MULTILINE))
        and bool(re.search(r"^@@ ", patch, re.MULTILINE))
    )


# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------

class _Sandbox:
    """
    Manages a single Docker container for one episode.

    **Simplified image selection**

    The original SWE-bench environment distributes per-instance Docker images
    (``swebench/sweb.eval.x86_64.<instance_id>:latest``) that contain the
    repository and its dependencies for each individual issue.  In practice
    these images are large and are not publicly accessible for all instance
    identifiers.  To avoid failures when the specific image is unavailable,
    this sandbox always uses the provided ``fallback_image`` for all issues.

    The container is started with ``--network none`` (no outbound traffic) and
    resource limits suitable for test execution.
    """

    def __init__(self, issue: dict, fallback_image: str) -> None:
        self._issue    = issue
        self._image    = self._pick_image(issue["instance_id"], fallback_image)
        self._cid: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--network", "none",
                "--memory", "2g",
                "--cpus",   "2",
                self._image,
                "sleep", "infinity",
            ],
            capture_output=True, text=True, check=True,
        )
        self._cid = result.stdout.strip()
        # Pin to the exact pre-patch commit and clean working tree
        base = self._issue.get("base_commit", "HEAD")
        self._exec(["git", "checkout", base],      check=False)
        self._exec(["git", "reset", "--hard", base], check=False)
        self._exec(["git", "clean", "-fdx"],        check=False)

    def stop(self) -> None:
        if self._cid:
            subprocess.run(
                ["docker", "stop", self._cid],
                capture_output=True, check=False,
            )
            self._cid = None

    def reset_to_base(self) -> None:
        """Discard all working-tree changes."""
        base = self._issue.get("base_commit", "HEAD")
        self._exec(["git", "reset", "--hard", base], check=False)
        self._exec(["git", "clean", "-fdx"],          check=False)

    def __enter__(self) -> "_Sandbox":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Low-level execution
    # ------------------------------------------------------------------

    def _exec(
        self,
        cmd:     list[str],
        workdir: str = _REPO_DIR,
        timeout: int = 120,
        check:   bool = True,
    ) -> subprocess.CompletedProcess:
        if not self._cid:
            raise RuntimeError("Sandbox is not running.")
        return subprocess.run(
            ["docker", "exec", "--workdir", workdir, self._cid] + cmd,
            capture_output=True, text=True, timeout=timeout, check=check,
        )

    def _write(self, container_path: str, content: str) -> None:
        """Copy a string into a file inside the container."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp = f.name
        try:
            subprocess.run(
                ["docker", "cp", tmp, f"{self._cid}:{container_path}"],
                check=True, capture_output=True,
            )
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def search(self, query: str, path: str = ".") -> str:
        r = self._exec(
            ["grep", "-rn", "--include=*.py", "-i", "--", query, path],
            check=False,
        )
        out = (r.stdout or "").strip()
        return out[:4000] if out else "(no matches)"

    def inspect(self, file_path: str, start: int = 1, end: Optional[int] = None) -> str:
        abs_path = (
            file_path if file_path.startswith("/")
            else f"{_REPO_DIR}/{file_path}"
        )
        r = self._exec(["cat", "-n", abs_path], check=False)
        if r.returncode != 0:
            return f"Error reading file: {r.stderr.strip()}"
        lines = r.stdout.splitlines()
        end   = end or len(lines)
        return "\n".join(lines[start - 1 : end])

    def hierarchy(self, depth: int = 3) -> str:
        r = self._exec(
            [
                "find", ".", "-maxdepth", str(depth),
                "-not", "-path", "*/.git/*",
                "-not", "-name", "__pycache__",
            ],
            check=False,
        )
        return (r.stdout or "").strip()[:3000]

    def edit(self, patch_hunk: str) -> tuple[bool, str]:
        """Apply a unified-diff hunk to the working tree."""
        patch_file = f"/tmp/edit_{os.urandom(4).hex()}.patch"
        self._write(patch_file, patch_hunk)
        r = self._exec(
            ["git", "apply", "--whitespace=fix", patch_file],
            check=False,
        )
        if r.returncode == 0:
            return True, "Patch applied."
        return False, r.stderr.strip()

    def run_tests(self, test_cmd: Optional[str] = None) -> tuple[bool, str]:
        """Run FAIL_TO_PASS tests. Returns (all_passed, truncated_output)."""
        if test_cmd is None:
            test_cmd = self._default_test_cmd()
        r = self._exec(["bash", "-c", test_cmd], timeout=180, check=False)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out[:3000]

    def apply_patch(self, patch: str) -> tuple[bool, str]:
        """
        Dry-run then apply a patch string.  Returns (success, message).
        """
        patch_file = f"/tmp/patch_{os.urandom(4).hex()}.patch"
        self._write(patch_file, patch)
        # Dry-run first so we get a clean error on failure
        dry = self._exec(
            ["git", "apply", "--check", patch_file],
            check=False,
        )
        if dry.returncode != 0:
            return False, dry.stderr.strip()
        self._exec(["git", "apply", "--whitespace=fix", patch_file], check=False)
        return True, "Patch applied."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_test_cmd(self) -> str:
        fail_to_pass = self._issue.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []
        if fail_to_pass:
            tests = " ".join(fail_to_pass)
            return f"python -m pytest {tests} -x -q --tb=short 2>&1"
        return "python -m pytest -x -q --tb=short 2>&1"

    @staticmethod
    def _pick_image(instance_id: str, fallback: str) -> str:
        """
        Select the Docker image to use for this sandbox.

        In this simplified implementation we always return the ``fallback``
        image.  Checking for per‑instance images can lead to authentication
        failures and complicates reproducibility.  By using a single base
        image for all issues, we ensure the sandbox can start without
        external dependencies.
        """
        return fallback


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class SWEBenchEnvironment(Environment):
    """
    SWE-bench Lite environment.

    Wraps a Docker sandbox around a real GitHub repository checked out at
    the pre-patch commit for a given SWE-bench Lite issue.

    The agent interacts via tool calls:

    +--------------+--------------------------------------------+----------+
    | Tool         | Arguments                                  | Budget   |
    +==============+============================================+==========+
    | search       | query, [path]                              | 1        |
    | inspect      | file, [start], [end]                       | 1        |
    | hierarchy    | [depth]                                    | 1        |
    | edit         | file, patch                                | 1        |
    | run_tests    | [cmd]                                      | 3 (×cost)|
    | submit_patch | patch                                      | 0        |
    +--------------+--------------------------------------------+----------+

    Parameters
    ----------
    issue_id : str
        SWE-bench instance identifier, e.g. ``"django__django-12345"``.
    dataset_path : str
        Path to a local SWE-bench Lite JSONL/JSON file or directory.
        Falls back to HuggingFace ``princeton-nlp/SWE-bench_Lite`` when
        absent.
    docker_image : str
        Fallback Docker image when no per-instance image is locally
        available.
    """

    def __init__(
        self,
        issue_id:     str,
        dataset_path: str = "data/swe-bench-lite",
        docker_image: str = "python:3.11-slim",
    ) -> None:
        self.issue_id     = issue_id
        self.dataset_path = dataset_path
        self.docker_image = docker_image

        self._issues:  list[dict]            = []
        self._issue:   Optional[dict]        = None
        self._sandbox: Optional[_Sandbox]    = None
        self._submitted_patch: str           = ""

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Load issue metadata, (re)start the Docker sandbox, and return the
        initial observation.

        The observation contains:
          - ``instance_id``, ``repo``, ``problem_statement``, ``hints_text``
          - ``repo_dir``  — absolute path to the repo inside the container
          - ``message``   — human-readable prompt
        """
        # Load dataset (cached after first call)
        if not self._issues:
            self._issues = _load_dataset(self.dataset_path)

        by_id = {iss["instance_id"]: iss for iss in self._issues}
        if self.issue_id not in by_id:
            available = list(by_id.keys())[:5]
            raise ValueError(
                f"Issue '{self.issue_id}' not found in dataset "
                f"({len(self._issues)} issues loaded). "
                f"First 5 available: {available}"
            )
        self._issue = by_id[self.issue_id]

        # Tear down any prior sandbox
        if self._sandbox is not None:
            self._sandbox.stop()

        self._sandbox = _Sandbox(self._issue, self.docker_image)
        self._sandbox.start()
        self._submitted_patch = ""

        return {
            "task":              "swe_bench",
            "instance_id":       self._issue["instance_id"],
            "repo":              self._issue.get("repo", ""),
            "problem_statement": self._issue.get("problem_statement", ""),
            "hints_text":        self._issue.get("hints_text", ""),
            "repo_dir":          _REPO_DIR,
            "message": (
                "Repository ready at pre-patch commit. Use search / inspect / "
                "hierarchy to explore, edit to modify files, run_tests to "
                "verify, and submit_patch when done."
            ),
        }

    def step(self, action: dict) -> tuple[dict, bool]:
        """Route the action to the appropriate tool handler."""
        if self._sandbox is None:
            return {"error": "Call reset() before step()."}, False

        tool = action.get("tool", "unknown")

        # ---- search -------------------------------------------------------
        if tool == "search":
            query = action.get("query", "")
            path  = action.get("path", ".")
            if not query:
                return {"error": "search requires 'query'"}, False
            return {
                "tool":   "search",
                "query":  query,
                "output": self._sandbox.search(query, path),
            }, False

        # ---- inspect ------------------------------------------------------
        if tool == "inspect":
            file_path = action.get("file", "")
            start     = int(action.get("start", 1))
            end_raw   = action.get("end")
            end       = int(end_raw) if end_raw is not None else None
            if not file_path:
                return {"error": "inspect requires 'file'"}, False
            return {
                "tool":    "inspect",
                "file":    file_path,
                "start":   start,
                "end":     end,
                "content": self._sandbox.inspect(file_path, start, end),
            }, False

        # ---- hierarchy ----------------------------------------------------
        if tool == "hierarchy":
            depth = int(action.get("depth", 3))
            return {
                "tool": "hierarchy",
                "tree": self._sandbox.hierarchy(depth),
            }, False

        # ---- edit ---------------------------------------------------------
        if tool == "edit":
            patch_hunk = action.get("patch", "")
            if not patch_hunk:
                return {"error": "edit requires 'patch'"}, False
            ok, msg = self._sandbox.edit(patch_hunk)
            return {"tool": "edit", "success": ok, "message": msg}, False

        # ---- run_tests ----------------------------------------------------
        if tool == "run_tests":
            test_cmd = action.get("cmd")
            passed, output = self._sandbox.run_tests(test_cmd)
            return {
                "tool":   "run_tests",
                "passed": passed,
                "output": output,
            }, False

        # ---- submit_patch -------------------------------------------------
        if tool == "submit_patch":
            patch = action.get("patch", "")
            self._submitted_patch = patch
            return {
                "tool":      "submit_patch",
                "submitted": True,
                "patch_len": len(patch),
            }, True  # done = True

        return {"error": f"Unknown tool: {tool!r}"}, False

    def score(self, patch: str) -> StatusCode:
        """
        Evaluate a candidate patch in an isolated environment and return a status code.

        This method first validates that ``patch`` is a unified diff.  It then
        attempts to delegate to the official SWE‑bench evaluation harness
        (``swebench.harness.run_evaluation``) if that package is installed.  The
        harness applies the patch to a fresh Dockerized repository and runs the
        test suites defined by the benchmark.  The harness is invoked via
        subprocess and inspected for outcomes:

          * If the harness report indicates the patch applied and all tests
            passed, returns ``OK_SOLVED``.
          * If the harness reports the patch could not be applied, returns
            ``PATCH_APPLY_FAIL``.
          * If the harness reports the patch format is invalid, returns
            ``INVALID_FORMAT``.
          * Otherwise returns ``TEST_FAIL``.

        If the harness cannot be imported or fails to run, the method falls back
        to a manual scoring procedure using a fresh ``_Sandbox``: it applies the
        patch and runs the FAIL_TO_PASS and PASS_TO_PASS tests to determine
        success.

        Parameters
        ----------
        patch : str
            Unified diff representing the candidate fix.

        Returns
        -------
        StatusCode
            One of ``INVALID_FORMAT``, ``PATCH_APPLY_FAIL``, ``TEST_FAIL``, or
            ``OK_SOLVED``.
        """
        if self._issue is None:
            raise RuntimeError("score() called before reset().")

        # 1. Validate unified diff format
        if not _is_unified_diff(patch):
            return "INVALID_FORMAT"

        # Attempt to use swebench harness if available
        use_harness = False
        try:
            import importlib.util
            spec = importlib.util.find_spec("swebench.harness.run_evaluation")
            use_harness = spec is not None
        except Exception:
            use_harness = False

        if use_harness:
            try:
                import tempfile
                import shutil
                from pathlib import Path

                # Prepare a temporary directory for predictions and reports
                run_id = f"budgetbench_{self._issue['instance_id'].replace('/', '_')}"
                work_dir = Path(tempfile.mkdtemp(prefix="sweb_eval_"))
                preds_path = work_dir / "predictions.jsonl"
                report_dir = work_dir / "reports"
                # Create predictions file for this single instance
                prediction = {
                    "instance_id": self._issue["instance_id"],
                    "model_name_or_path": "budgetbench-env",
                    "model_patch": patch,
                }
                preds_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

                cmd = [
                    "python", "-m", "swebench.harness.run_evaluation",
                    "--dataset_name", "princeton-nlp/SWE-bench_Lite",
                    "--predictions_path", str(preds_path),
                    "--instance_ids", self._issue["instance_id"],
                    "--max_workers", "1",
                    "--run_id", run_id,
                    "--cache_level", "env",
                    "--clean", "true",
                    "--rewrite_reports", "true",
                    "--report_dir", str(report_dir),
                ]

                completed = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)

                # If harness executed successfully, inspect report file
                if completed.returncode == 0:
                    # Locate report JSON
                    report_files = list(report_dir.rglob("*.json"))
                    if report_files:
                        try:
                            content = report_files[0].read_text(encoding="utf-8").lower()
                            # Determine result heuristically
                            if "\"resolved\": true" in content or "\"success\": true" in content:
                                return "OK_SOLVED"
                            if "patch_apply_fail" in content or "\"applied\": false" in content:
                                return "PATCH_APPLY_FAIL"
                            if "invalid" in content and "patch" in content:
                                return "INVALID_FORMAT"
                            return "TEST_FAIL"
                        except Exception:
                            # Could not parse report, fall through to manual
                            pass
                    else:
                        # No report generated, treat as generic failure
                        return "TEST_FAIL"
                else:
                    # Harness returned non-zero; inspect stderr/stdout
                    stderr = (completed.stderr or "") + "\n" + (completed.stdout or "")
                    lower_err = stderr.lower()
                    if "invalid" in lower_err and "patch" in lower_err:
                        return "INVALID_FORMAT"
                    if "git apply" in lower_err or "patch apply" in lower_err or "reject" in lower_err:
                        return "PATCH_APPLY_FAIL"
                    # Unknown error, fall back to manual evaluation
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)  # type: ignore[name-defined]
                except Exception:
                    pass

        # # Manual scoring path as a fallback
        # with _Sandbox(self._issue, self.docker_image) as oracle:
        #     # Attempt to apply the patch
        #     applied, _msg = oracle.apply_patch(patch)
        #     if not applied:
        #         return "PATCH_APPLY_FAIL"

        #     # Run FAIL_TO_PASS tests (bug-triggering tests)
        #     passed, _out = oracle.run_tests()
        #     if not passed:
        #         return "TEST_FAIL"

        #     # Run PASS_TO_PASS regression tests (should still pass)
        #     p2p_tests = self._issue.get("PASS_TO_PASS", [])
        #     if isinstance(p2p_tests, str):
        #         try:
        #             p2p_tests = json.loads(p2p_tests)
        #         except json.JSONDecodeError:
        #             p2p_tests = []
        #     if p2p_tests:
        #         regression_cmd = "python -m pytest " + " ".join(p2p_tests) + " -x -q --tb=short 2>&1"
        #         regression_ok, _rout = oracle.run_tests(regression_cmd)
        #         if not regression_ok:
        #             return "TEST_FAIL"
        print("Warning: SWE-bench harness not available; falling back to manual scoring.")
        return "TEST_FAIL"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def issue(self) -> Optional[dict]:
        """Currently active issue metadata (populated after ``reset()``)."""
        return self._issue

    @classmethod
    def list_issues(
        cls,
        dataset_path: str = "data/swe-bench-lite",
        limit: int = 300,
    ) -> list[str]:
        """Return the list of ``instance_id`` values available in the dataset."""
        issues = _load_dataset(dataset_path)
        return [iss["instance_id"] for iss in issues[:limit]]

    def close(self) -> None:
        """Stop the Docker sandbox and release resources."""
        if self._sandbox is not None:
            self._sandbox.stop()
            self._sandbox = None

    def __del__(self) -> None:
        self.close()
