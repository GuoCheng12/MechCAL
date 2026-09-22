# Release Scope

The standalone repository and distribution are named MechCAL. Python imports
use `mechcal`; environment variables use `MECHCAL_`. The first release is
0.1.0. These names replace the research workspace's internal version names.

On the main branch, GitHub Actions runs offline tests before publishing the
version in `pyproject.toml`. Existing releases are not overwritten. Each new
release includes the wheel and source distribution. Pull requests cannot
publish releases. The workflow can also be started manually after a failed
publication attempt.

The export retains the existing scientific implementation, mechanism pool,
agent prompts, rubric, and regression tests. Historical rubric and prompt
revision identifiers remain intact. Packaging does not establish that a
particular archived experiment used the current defaults.

Only source code, configuration templates, documentation, and synthetic test
fixtures are included. Research outputs, chain-of-thought records, source
papers, hidden references, credentials, relay configuration, model weights,
and vendored third-party repositories are excluded. Generated output is
local and ignored by Git.

## Reproducibility

Release checks cover Python compilation, core tests, optional MCP transport,
wheel installation, package resources, CLI help, offline metric examples,
and distribution content. Real model API and Amesp runs require user-supplied
credentials and software. CI does not call paid model endpoints.

Numerical results, token counts, confidence intervals, and estimated resource
tables are not distributed as verified measurements. Recompute them from
the relevant complete run when reporting a new experiment.

## Third-Party Software

RDKit, ASE, NumPy, SciPy, OpenAI's Python client, MCP libraries, and optional
ChemGraph dependencies retain their respective licenses. Amesp is external.
No claim is made that the project's license grants rights to those packages,
model checkpoints, papers, or datasets.

MechCAL is distributed under the MIT License. Paper citation metadata will
be added when the authors provide the final public reference.
