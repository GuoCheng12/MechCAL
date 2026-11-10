# Deployment Validation

The 0.2.0 integration was checked on Linux x86_64 with Python 3.11.
The checks below concern the portable code and current workflow. They are not
a new benchmark evaluation or a reproduction of all historical measurements.

## Completed Checks

| Check | Result | Scope |
| --- | --- | --- |
| Offline regression | 3,375 passed, 4 skipped | Portable tests, no paid model calls |
| Local MCP integration | 3 passed | Real local subprocess transport and registry |
| Deployment and packaging subset | 19 passed | Included in the offline total, not additional tests |
| Installed wheel outside repository | Passed | Isolated Python import, packaged profile and prompt |
| Installed MCP discovery | 67 capabilities | Explicit local tool paths, no old workspace imports |
| RDKit preparation | Passed | Small formaldehyde molecule |
| Amesp computation | Passed | Parsed ground-state energy and one excited state |
| Official DeepSeek connectivity | Passed | `deepseek-flash`, high reasoning, one short request |
| Ruff and Python compilation | Passed | Source and portable tests |
| Publication content scan | Passed | Credential patterns, private endpoints and machine paths |

The connectivity probe returned `OK` and reported 44 input plus 20 output
tokens. It did not run a molecular reasoning workflow or call a judge.
The small Amesp check does not validate every numerical method, element,
large-molecule calculation or optional backend.

## Portability Boundary

The wheel was installed separately and imported using Python isolated mode
from outside the checkout. Its current-workflow profile and task prompt are
package resources. Configuration supplies scientific executable locations;
inference does not import experiment scripts or require benchmark records.

Local regression checks reused installed dependencies. A separate clean
dependency-installation check is not included in the results above. GitHub
Actions performs a fresh install using `env/constraints-linux-py311.txt`, runs
offline and MCP tests, checks small RDKit preparation, and builds distributions.
The constraints fix tested direct dependencies, not every transitive package.
Consult the workflow result for the exact commit before relying on CI success.

Existing scientific source files were compared with the active research
implementation and retained unchanged during deployment preparation. The
new launcher, configuration profile and deployment checks are additive.
Historical CLI defaults differ from the new command profile; select the
entry point and limits explicitly when comparing experiments.

## Excluded Material

Replay tests that require private experiment directories, original model
outputs or raw calculation fixtures remain in the research archive. Portable
synthetic tests were retained; unavailable private fixtures were not published
or replaced with fabricated successful results.

Credentials, inference histories, hidden references, source papers, model
weights and external scientific binaries are excluded. Pattern scans are
supplemented by file-scope review but cannot establish scientific correctness.
The previously published 119-case SMILES input set remains unchanged.
