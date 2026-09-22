# Scientific Tools

Macro tools use RDKit to obtain structural and geometry-based evidence.
Microscopic tools invoke bounded Amesp calculations when Amesp is enabled.
Both expose the capability catalog in `mechcal/capabilities/`.

## RDKit

Install the `science` extra or use `environment.yml`. The structure
preparation code fixes its random seed and records preparation information
with each generated artifact. Inspect `mechcal/chem/structure_prep.py` for
conformer and force-field settings. Structural and packing proxies are not
measurements of a molecule's actual aggregate or crystal structure.

## Amesp

Amesp binaries and third-party libraries are not redistributed. Install an
authorized version locally and set:

```bash
export MECHCAL_AMESP_BIN="/path/to/amesp"
export MECHCAL_AMESP_TIMEOUT="300"
export MECHCAL_AMESP_NPARA="1"
export MECHCAL_AMESP_MAXCORE_MB="1000"
```

`mechcal/tools/amesp_baseline.py` controls execution; bounded follow-up
routes are in `mechcal/tools/amesp_microscopic.py`. Tool failures and
unsupported capabilities are recorded explicitly. A missing executable
does not produce a substitute quantum calculation.

## MCP

Install the `mcp` extra. `mechcal-mcp` starts the stdio server. The
single-case command can launch it through `--tool-backend mcp-stdio`.
This exposes the same local capabilities; it does not add web retrieval.

```bash
pytest -q -m integration
```

MCP integration tests launch local subprocesses and require a host that
permits stdio communication. Real Amesp execution is a separate validation
step requiring the external executable.
