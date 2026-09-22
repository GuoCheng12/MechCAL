# Scientific Tool Installation

Follow this guide after cloning MechCAL. Commands use a Linux shell. The
full scientific workflow is tested on Linux x86_64 with Python 3.11 and a
CPU Amesp executable. Native macOS, Windows, ARM, and GPU Amesp execution
have not been validated here; use a compatible Linux host for the full run.

## What You Need

| Component | Purpose | Installation |
| --- | --- | --- |
| Python 3.11 and pip | Run MechCAL | Install before creating the environment |
| RDKit | SMILES parsing, descriptors, conformers, force fields | Included in `.[science]` |
| ASE, NumPy, SciPy | Structure files and numerical operations | Included in `.[science]` |
| OpenAI Python client, HTTPX, Pydantic, Typer | Model access, schemas, CLI | Installed with MechCAL |
| Amesp | Quantum-chemical calculations | Separate upstream binary distribution |
| FastMCP and MCP | Optional stdio tool transport | Included in `.[mcp]` |
| pytest, Ruff, build | Optional development checks | Included in `.[dev]` |
| ChemGraph | Optional comparison baseline | Separate environment; see [Baselines](baselines.md) |

Core installation (`pip install -e .`) alone does not install the scientific
tools. Neither pip nor conda installs Amesp through MechCAL. No model weights
or GPU are needed when using a hosted model API and the CPU tools described
here. The MechCAL adapter invokes Amesp directly; PyAmesp is not required.

## Python and RDKit

From the cloned repository root, choose **one** installation route.

### pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[science,mcp]"
python -m pip check
python examples/check_tools.py --mcp
```

The last command imports the scientific packages and prepares a small
molecule with the same RDKit adapter used by MechCAL. It prints installed
versions and `structure_preparation: passed`. It makes no model API calls
and does not run Amesp unless `--amesp` is supplied.

### conda

```bash
conda env create -f environment.yml
conda activate mechcal
python -m pip check
python examples/check_tools.py --mcp
```

The environment file also installs the development dependencies. Run it
from the repository root because its editable installation refers to `.`.
Do not mix this environment with the `.venv` from the pip route.

RDKit's supported installation routes are documented in the
[official installation guide](https://www.rdkit.org/docs/Install.html).
The pip package is named `rdkit`, not the historical `rdkit-pypi`.
Conformer preparation records its configuration and selected force field;
installing a new RDKit release can change numerical results.

## Amesp

### Download and Extract

Obtain the binary distribution from the
[official download page](https://www.amesp.xyz/download.html), and follow
the terms and citation instructions supplied by Amesp. The page currently
links an Amesp 2.1(dev) archive. MechCAL does not redistribute that archive
or grant rights to it through its MIT license.

For the currently linked ZIP, these commands download and extract it into
a user-owned directory without `sudo`:

```bash
mkdir -p "$HOME/Downloads" "$HOME/opt"
curl --fail --location https://www.amesp.xyz/Amesp_2.1_dev.zip \
  --output "$HOME/Downloads/Amesp_2.1_dev.zip"
unzip -l "$HOME/Downloads/Amesp_2.1_dev.zip"
unzip -n "$HOME/Downloads/Amesp_2.1_dev.zip" -d "$HOME/opt"
find "$HOME/opt" -type f -path '*/Bin/amesp' -print
```

`curl` and `unzip` must be available on the host. If the link changes, use
the current archive linked by the official download page. Inspect the
archive listing before extraction. Preserve the whole distribution,
including `Bin/`, `Lib/`, the manual, and examples; do not copy only the
`amesp` executable. If an older installation already exists, extract into
a separate directory rather than mixing files from two builds.

### Configure the Shell

Set `AMESP_HOME` to the extracted directory that contains `Bin` and `Lib`.
Adjust the first line if the archive has a different top-level name:

```bash
export AMESP_HOME="$HOME/opt/Amesp"
test -f "$AMESP_HOME/Bin/amesp"
chmod u+x "$AMESP_HOME/Bin/amesp"
export PATH="$AMESP_HOME/Bin:$PATH"
export MECHCAL_AMESP_BIN="$AMESP_HOME/Bin/amesp"
export KMP_STACKSIZE=4G
ulimit -s unlimited

export MECHCAL_AMESP_TIMEOUT=300
export MECHCAL_AMESP_NPARA=1
export MECHCAL_AMESP_MAXCORE_MB=1000
```

The PATH and stack settings follow the
[official Amesp instructions](https://www.amesp.xyz/doc.html). If a cluster
does not permit an unlimited stack, follow its administrator's limits.
`AMESP_HOME` above is a shell convenience; MechCAL reads the explicit
`MECHCAL_AMESP_BIN` path. These exports apply to the current shell only.

The timeout is per Amesp subprocess, not per complete molecule. Start with
one CPU worker and a 1000 MB Amesp memory setting. These settings do not
impose a process-wide memory limit or a total MechCAL runtime limit.

### Verify Real Computation

From the MechCAL repository root in the activated Python environment:

```bash
python examples/check_tools.py --amesp --mcp
```

This prepares formaldehyde (`C=O`), runs an aTB1 ground-state optimization
and a TDA-aTB1 excitation calculation, and checks normal termination plus
parsed energy and excited-state output. Expect `ground_state: passed` and
`excited_state: passed`. The calculation uses a temporary directory and
does not call a model API. A successful import or `command -v amesp` alone
does not establish that quantum calculations work.

The local installation used for this check has an Amesp 2.1(dev) manual
dated 2026-02-03. Its Linux executable SHA-256 is
`253efef65355ab698622a6131b880a7435c044f001ecdc9c187b527881b5a690`.
The upstream download is a rolling development build, not that exact
archived binary. Verify your own installation and record its output banner
and checksum; matching the name `2.1(dev)` is not sufficient to reproduce
an older environment.

### Enable It in MechCAL

After also configuring the [model API](configuration.md):

```bash
mechcal-run-case \
  --smiles 'C(c1ccccc1)(c1ccccc1)=C(c1ccccc1)c1ccccc1' \
  --case-id example-001 \
  --llm-planner --amesp --max-rounds 30 \
  --output-dir outputs/example-full
```

`--amesp` makes microscopic calculations available; the Planner still
selects routes. It does not force every route to run on every molecule.

| Entry point | Amesp behavior |
| --- | --- |
| `mechcal-run-case` | Off unless `--amesp` or `MECHCAL_ENABLE_AMESP=true` is supplied |
| `mechcal-evaluate` | On by default; `--no-amesp` explicitly disables it |

Use `--no-amesp` for an explicitly reduced-tool demonstration. Such a run
does not reproduce the full microscopic evidence route. Missing tools and
unsupported calculations are recorded, not replaced with invented results.

## MCP Transport

The default local tool backend calls Python adapters directly. MCP is
optional; it exposes the same capabilities through a local stdio process.
No external MCP service, web retrieval, or listening TCP port is required.

```bash
python -m pip install -e ".[science,mcp]"
mechcal-run-case \
  --smiles 'C(c1ccccc1)(c1ccccc1)=C(c1ccccc1)c1ccccc1' \
  --case-id example-mcp \
  --llm-planner --amesp --tool-backend mcp-stdio \
  --max-rounds 30 --output-dir outputs/example-mcp
```

The command starts its own stdio server. Do not start `mechcal-mcp` in
another terminal first; that entry point is for clients that launch and
communicate with a stdio server themselves. To test the transport without
calling a model API or running real Amesp calculations:

```bash
python -m pip install -e ".[science,mcp,dev]"
python -m pytest -q -m integration
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `mechcal-run-case: command not found` | Activate the environment used for installation; run `python -m pip show mechcal`. |
| RDKit, ASE, or SciPy import fails | Install `.[science]` in that same environment; run `python -m pip check`. |
| RDKit has no compatible wheel | Use the conda route and a supported Python version; see RDKit's installation guide. |
| `amesp_binary_missing` | Set `MECHCAL_AMESP_BIN` to the executable, not its directory. |
| `Permission denied` | Check executable permission and whether the filesystem is mounted `noexec`. |
| `Exec format error` | Use a binary matching the OS and CPU architecture; the local validated build is Linux x86_64. |
| Missing shared library or basis data | Preserve the full distribution and follow the requirements for your downloaded build. Do not assume another build has the same linkage. |
| Stack allocation failure | Check `KMP_STACKSIZE` and the host's permitted stack size. |
| Amesp timeout or abnormal termination | Inspect the run's `.aop` and stderr artifacts, charge/multiplicity and supported method; start with one CPU worker. |
| Metal-containing molecule cannot be embedded or optimized | Inspect the recorded tool failure; installation success does not guarantee force-field coverage for every molecule. |
| MCP server closes immediately | Install `.[mcp]`, keep the environment active, and run the integration tests. |
| Model output ends before valid JSON | Check provider thinking settings and output-token limits. This is separate from tool installation. |

Structural and packing proxies are not measurements of the actual crystal
or aggregate. Passing an installation check does not remove that boundary.
