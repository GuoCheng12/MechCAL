# Baselines

| Method | Entry point | Behavior |
| --- | --- | --- |
| Direct LLM | `mechcal-direct` | Structure-only response, no scientific tools |
| Tool-augmented LLM | `mechcal-tool-llm` | ReAct loop over the available tools |
| CommonPrior@3 | `mechcal-common-prior` | Fixed three-family prediction |
| Adapted ChemGraph | `mechcal-chemgraph` | Upstream Planner--Executor graph and final aggregation |

Direct inference supports locally served checkpoints and compatible hosted
APIs. Chem-R uses `--native-think-answer`; Intern models use
`--native-thinking-json --intern-api-thinking` with an appropriate endpoint.
Use `--evidence-format --strict-json-schema` when requiring the corresponding
three-prediction evidence format. Model weights and provider-specific
deployment scripts are not included.

## Adapted ChemGraph

The adapter was developed against upstream ChemGraph commit
`203e18c8529869531fa4addbcdb3f1395a831eaa` (v0.6.0). Install that
revision separately in a dedicated environment, along with
`mechcal/baselines/chemgraph_mas/requirements-core.txt`. The upstream source
is available at https://github.com/argonne-lcf/ChemGraph.

From the MechCAL repository root, create a separate environment so the
pinned LangChain dependencies do not replace those in another project:

```bash
mkdir -p third_party
python3.11 -m venv third_party/chemgraph-env
source third_party/chemgraph-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[science,chemgraph]" \
  -r mechcal/baselines/chemgraph_mas/requirements-core.txt
git clone https://github.com/argonne-lcf/ChemGraph.git third_party/ChemGraph
git -C third_party/ChemGraph checkout --detach 203e18c8529869531fa4addbcdb3f1395a831eaa
python -m pip install --no-deps -e third_party/ChemGraph
python -m pip check
mechcal-chemgraph --help
```

This is the dependency set for the adapted baseline, not an installation
of all upstream ChemGraph features. The `--no-deps` flag intentionally
avoids unrelated upstream tool integrations. Any upstream dependency
warnings from `pip check` must be reviewed against that limited scope;
do not silently ignore missing packages used by the adapter. Configure
the model API and Amesp as described in [Tools](tools.md) before a real run.
ChemGraph is not needed to run MechCAL itself.

The adapter exposes the same Macro and Microscopic capability implementations.
It does not add MechCAL's reviewers, support audit, or claim gate. PubChem,
literature retrieval, web search, and upstream default chemistry tools are
not exposed.

```bash
mechcal-chemgraph --case-dir /path/to/cases --output-dir outputs/chemgraph \
  --max-model-calls 30 --max-tool-calls 30 --max-total-tokens 200000 \
  --max-output-tokens-per-call 2400
```

Use `--responses-api` for a compatible Responses endpoint. This path omits
unsupported sampling controls. The tool-call cap applies to ChemGraph;
MechCAL's evidence-round limit has different semantics.

The source contains a Responses client used by historical Codex-related
experiments. That helper alone does not reproduce an external general-purpose
Codex agent harness. Only the workflows actually provided above are packaged
as runnable baseline commands.
