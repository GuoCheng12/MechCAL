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
4. **Recipes for known chains, a scoped Worker for adaptive ones.** Deterministic multi-step tool
   chains remove parameter traps for frequent observables; open-ended tasks are delegated.
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
  after every k=3 computations and once before submission. It works in two steps:
  first it ranks the 11 families from a *facts-only view* (structural facts and artifact
  values, without the Planner's plausibilities, reasons or ranking), then it compares its
  ranking with the board. Disagreements, untested families and claims whose wording goes
  beyond the cited values become at most three proposals, each with a discriminating
  test. Planner must answer each proposal (accept / adapt / decline with a reason); the
  answer is logged. Critic never edits the board. This keeps the useful part of the
  0.2.0 Exploration Agent (an isolated fact view) without its per-result cost.

## 3a. Who does what (mapping from 0.2.0)

| 0.2.0 role | Responsibility | Where it goes |
| --- | --- | --- |
| Worker: choose parameters, chain steps | operational | Recipes for standard chains; the scoped Worker (below) for adaptive tasks |
| Worker: explain a result | interpretation | Planner, in the same call that receives the observation, writing cited claims |
| Coverage Reviewer (cold start) | list possibilities | Deterministic structural facts + all 11 families on the board from step 0 |
| Impact Reviewer (after every result) | which beliefs should move | Planner board update on each observation; factual consistency by Layer A self-check |
| Exploration Agent | independent alternatives | Critic (facts-only view, periodic) |
| Final Reviewer / delivery repair | text vs records consistency | Not needed: submission is a projection of cited board claims, checked by Layer A |

### Worker (retained, scoped) — revised 2026-11-12 after author review

Recipes are a fast path, not the only path. Real cases need adaptive experimentation that
fixed chains cannot anticipate; the full 0.2.0 Case 0003 run already needed it (a
medium-grid hole-electron analysis failed its numerical checks and was redone at high
grid; a PBE0 spin-state route timed out after SCF and a lighter PBE route returned
partial states; contact scans needed coordinate redefinition). So the Planner has three
ways to obtain evidence:

| Action | Use when | LLM calls |
| --- | --- | --- |
| `recipe` | a standard observable with validated defaults | 0 |
| `tool` | one calculation with Planner-chosen scientific arguments | 0 |
| `delegate` | an adaptive task: choose states/conformers/poses from results, troubleshoot a failed or numerically rejected calculation, vary a coordinate range, compare settings | Worker, bounded |

The Worker receives a task (goal, the families it should separate, allowed tools/recipes,
budget) and works through the same gateway, so it sees pointer-indexed values and never
navigates record directories. It may retry with different parameters; any change of
scientific method (functional, basis, state, geometry source, grid) is recorded as a
separate artifact and stated in its claims, never substituted silently. It returns cited
claims, a short account of what it tried and why, and unresolved failures. There is no
separate explanation or impact-review call after it returns.

Whether delegation pays for itself is an experiment, not an assumption: pilots run
Planner+recipes+tools with and without `delegate` at equal budget, and the paper reports
the Worker as an ablation either way.

## 3b. Against anchoring on the first three mechanisms

The 0.2.0 runs showed the Planner repeatedly confirming its initial top labels with easy
evidence. The redesign does not rely on a reviewer prompt alone:

1. **Comparison, not confirmation.** Every computation request names at least two
   families it is meant to separate and the outcome expected under each. A result then
   updates both, and a request that only "supports X" is rejected by the schema.
2. **All 11 families stay visible.** The board lists every family with a status
   (untested / tested-supported / tested-weakened / not testable with available tools)
   and the cheapest test that would change it.
3. **Blind Critic.** The facts-only ranking step above is independent of the Planner's
   beliefs; when its top-3 differs, the Planner must address the difference.
4. **Submission check.** Before submit, a family ranked 4–6 whose discriminating test is
   untried and affordable within budget triggers a Critic proposal that must be answered.
   This is a prompt-for-decision, not a forced computation or a forced rank change.
5. **Measured, not assumed.** Logged per case: share of computations targeting the
   initial top-3, families tested, rank changes after evidence and whether they moved
   toward the reference. These numbers go into the paper either way.
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
