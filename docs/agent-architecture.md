# Agent Architecture Plan (release 0.3.0)

Status: proposal, 2026-11-11. To be implemented on a new branch in a new package
(`mechcal/agent/`); the 0.2.0 `command_v1` workflow stays untouched for reference runs.

## 1. Why a rewrite rather than more patches

Observed on the current system (code survey, handoff records, one tracked case):

| Problem | Evidence |
| --- | --- |
| Cost makes the benchmark unrunnable | Case 0003: ~3.85M tokens, 122 requests, ~2 h, 19 computations, no final answer |
| Complexity hides behaviour | ~62k LOC, 110 runtime modules, ~50 profile flags, 69 prompts (275 KB) |
| Effort goes into plumbing | Recent commits mostly fix context transmission, ID confusion, delivery repair |
| Evidence is not verifiable | Final evidence is free text; numbers are not tied to artifact values |
| Known tool traps leak into agent decisions | e.g. `td_tout=1` silently blocks later hole-electron analysis |

The story stays: evidence should help *discover and compare* mechanisms, not only
explain a prediction. The redesign keeps the scientific tools and adapters, the gateway and
the grounded evidence contract, and replaces the orchestration.

## 2. Design principles

1. **One state object, small context.** A hypothesis board holds the current belief
   for all 11 families. Each model call sees the board plus what changed, never the
   full history.
2. **Every claim is born typed and cited.** Interpretation of a tool result produces
   grounded claims (structural / computed with artifact pointers / prior) at the moment the
   result arrives. The final submission is a projection of the board, not a rewrite.
3. **Tools through the recorded gateway only.** Same gateway as the ReAct baseline, so
   costs and artifacts are comparable by construction.
4. **Recipes instead of worker LLMs for known chains.** Deterministic multi-step tool
   chains for frequent observables remove Worker calls and parameter traps.
5. **Explicit budgets, reported.** Max computations, max model calls, max tokens; the
   stop reason is part of the output.
6. **No hidden-reference, rubric or judge access.** The internal self-check is the
   Layer A artifact-consistency check only (objective, rubric-free), and it is ablated.

## 3. Components

```text
            SMILES ─► Gateway.prepare ─► structural facts (RDKit, deterministic)
                                   │
          ┌────────────────────────▼─────────────────────────┐
          │ Hypothesis board (11 families)                     │
          │  status: open / supported / weakened / unresolved  │
          │  plausibility (0-1), typed claims, open questions   │
          └──────┬───────────────────────────────▲─────────────┘
                 │ board + deltas                │ claim/board updates
          ┌──────▼──────┐   choose test   ┌──────┴──────┐
          │  Planner    │ ───────────────►│  Gateway    │ recipes / single tools
          │  (1 LLM)    │◄─── observation │ (recorded)  │
          └──────┬──────┘                 └─────────────┘
                 │ every k computations / before submit
          ┌──────▼──────┐
          │  Critic     │ alternatives the board under-weights + claim/artifact
          │  (1 LLM)    │ consistency (Layer A results as input), advice only
          └──────┬──────┘
                 ▼
          Grounded submission = top-3 of board, claims filtered by Layer A
```

- **Planner** (single role, one call per step). Input: board, latest observation
  (values by pointer), remaining budget. Output (one native structured response):
  board edits (plausibility changes with reasons, new typed claims citing the
  observation) and the next action: a recipe, a single tool, or `submit`.
  The plausibility edit and its justification are recorded per step; this is the data
  for the "does evidence change rankings in the right direction" analysis.
- **Critic** (merges the current Scientific Reviewer and Exploration Agent). Called
  after every k=3 computations and once before submission. It sees the board and the
  Layer A report of current claims; it returns at most three proposals (an
  under-weighted family plus a discriminating test, or a claim to correct). Planner may
  accept or decline; acceptance is logged. Critic never edits the board.
- **Gateway** (`mechcal/eval/tool_gateway.py`): executes tools and recipes, persists
  artifacts, returns pointer-indexed values.
- **Recipes** (`mechcal/agent/recipes.py`): named chains with fixed, validated parameters,
  e.g.
  - `excited_states`: ground-state optimisation → baseline bundle (td_tout ≥ 10)
  - `charge_transfer_character`: `excited_states` → hole-electron analysis
  - `torsion_response`: rotatable dihedrals → torsion brightness scan
  - `spin_states`: singlet/triplet comparison (+ SOC when elements are in scope)
  - `dimer_contacts`: dimer search → contact torsion response on the best pose
  - `redox_alignment`: vertical IP/EA
  Each recipe step is a normal recorded gateway call; failures are returned, not hidden.

## 4. Budgets (initial targets, to be measured)

| Quantity | Target per case | Current system |
| --- | --- | --- |
| Computations | ≤ 10 (recipes count their steps) | 19+ |
| Model calls | ≤ 25 | 122+ |
| Tokens | ≤ 400k | ~3.85M (unfinished) |
| Wall time | ≤ 30 min | ~2 h (unfinished) |

## 5. Experiments this design enables

- **Budget-matched comparison**: Direct, ReAct and the redesigned MechCAL share the gateway
  and the grounded submission contract; ReAct gets the same computation and model-call budgets.
  Also best-of-n Direct at equal tokens.
- **Component ablations** (R1 W4/W5, R2 Q2): no Critic; no recipes (single tools only);
  no iteration (plan once, run all, submit); no tools (= Direct); no Layer A self-check.
- **Evidence-driven revision analysis** (story): from the per-step board edits, count
  ranking changes after new evidence that move toward vs away from the reference.
- **Calibration** (R1 W5): board plausibility vs judge support level and vs reference
  membership (ECE, Brier). If calibration is poor, the paper does not claim it.

## 6. Implementation plan

| Window | Deliverable |
| --- | --- |
| 11/12–11/16 | `mechcal/agent/` board + planner loop + recipes over the gateway; offline tests with scripted models |
| 11/17–11/20 | Critic; budgets and stop reasons; submission projection; cost ledger |
| 11/21–11/24 | Pilot on ~10 dev cases (needs API key): cost, failure modes, fixes |
| 11/25–12/08 | Iterate on pilot findings; freeze the 0.3.0 configuration before main runs |

0.2.0 reference runs (`command_v1` profile) are kept only as a cost/behaviour baseline.
