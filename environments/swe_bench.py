"""
BudgetBench — SWE-bench Lite Environment  (Track A · Task A1)
==============================================================
Wraps SWE-bench Lite issues inside the BudgetBench Environment interface.

TODO (Hanyu Jin):
  [ ] Set up Docker sandbox (see swe-bench docs)
  [ ] Load ≥30 issues from the SWE-bench Lite dataset
  [ ] Implement patch applicability checker
  [ ] Implement unit-test oracle (run the repo's test suite)
  [ ] Complete reset(), step(), score()

Tools the environment should support:
  - search   query            → grep-like search across the repo
  - inspect  file [start end] → read file lines
  - hierarchy                 → show directory tree
  - edit     file patch       → apply a hunk to a file
  - run_tests                 → run the issue's test suite (cost = 3×)
  - submit_patch patch        → finalise and score (done = True)

Stub provided by: Chenrui Xu
To be implemented by: Hanyu Jin (hanyu/environments branch)
"""

from __future__ import annotations

from budgetbench.interfaces import Environment, StatusCode


class SWEBenchEnvironment(Environment):
    """
    SWE-bench Lite environment (stub — to be completed by Hanyu).

    Parameters
    ----------
    issue_id : str
        SWE-bench issue identifier, e.g. "django__django-12345".
    dataset_path : str
        Path to the local SWE-bench Lite dataset.
    docker_image : str
        Docker image to use for sandboxed execution.
    """

    def __init__(
        self,
        issue_id:     str,
        dataset_path: str = "data/swe-bench-lite",
        docker_image: str = "swe-bench:latest",
    ) -> None:
        self.issue_id     = issue_id
        self.dataset_path = dataset_path
        self.docker_image = docker_image
        # TODO: load issue metadata from dataset
        self._repo_state: dict = {}

    def reset(self) -> dict:
        """
        TODO: Start a fresh Docker container, check out the repo at the
        pre-patch commit, and return the initial observation.
        """
        raise NotImplementedError("SWEBenchEnvironment.reset() — to be implemented by Hanyu")

    def step(self, action: dict) -> tuple[dict, bool]:
        """
        TODO: Route the action to the appropriate tool handler
        (search, inspect, hierarchy, edit, run_tests, submit_patch).
        """
        raise NotImplementedError("SWEBenchEnvironment.step() — to be implemented by Hanyu")

    def score(self, patch: str) -> StatusCode:
        """
        TODO: Apply patch to repo, run the golden test suite,
        return the appropriate StatusCode.

        Scoring order:
          1. Validate unified-diff format       → INVALID_FORMAT
          2. Apply patch (git apply)            → PATCH_APPLY_FAIL
          3. Run test suite                     → TEST_FAIL / OK_SOLVED
        """
        raise NotImplementedError("SWEBenchEnvironment.score() — to be implemented by Hanyu")
