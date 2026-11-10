# Release Scope

The standalone repository and distribution are named MechCAL. Python imports
use `mechcal`; environment variables use `MECHCAL_`. Version 0.2.0 integrates
the current command workflow and adds a portable configuration and launcher.
The earlier 0.1.0 release remains available in Git history.

Pushes and pull requests run offline tests without publishing a release.
Publishing requires manually starting the Tests workflow on `main` with
`publish_release` enabled. Tests must pass first. Existing releases are not
overwritten; each new release includes the wheel and source distribution.

The integration retains the current research implementation, mechanism pool,
agent prompts and rubrics. It adds no new scientific decision rules. Portable
synthetic regression tests are included; private experiment replays remain in
the research archive. Historical rubric and prompt revision identifiers remain
intact. Packaging does not establish that a particular archived experiment used
the current defaults.

Source code, configuration templates, documentation, synthetic test fixtures,
and the versioned 119-case PhotoMechBench SMILES input set are included.
The public input CSV is shipped with the Git repository and source distribution,
not the Python wheel. Research outputs, chain-of-thought records, source
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
