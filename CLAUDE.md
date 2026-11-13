## Core Expectations

- Think before coding. Do not assume, do not hide confusion, and surface tradeoffs when there are multiple reasonable interpretations.
- Prefer simplicity. Write the minimum code needed for the current request, with no speculative flexibility, configuration, or future-facing features.
- Make surgical changes. Touch only the files and lines required by the current task; do not opportunistically refactor, reformat, or clean up adjacent code.
- Execute toward verifiable goals. For multi-step work, state a brief plan and define how each step will be checked; actually verify before calling the task done.
- Architecture changes require explicit user approval before implementation.

## Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask before finalizing: would a senior engineer say this is overcomplicated? If yes, simplify.

## Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless it directly supports the request.
- Do not refactor things that are not broken.
- Match the existing style, even if you would choose a different style.
- If unrelated dead code is noticed, mention it rather than deleting it.

When your changes create orphans:
- Remove imports, variables, functions, and tests that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## Goal-Driven Execution

Transform tasks into verifiable goals:

- "Add validation" means write or identify checks for invalid inputs, then make them pass.
- "Fix the bug" means reproduce or reason from a concrete failure, then verify the fix.
- "Refactor X" means preserve behavior and run relevant checks before calling it done.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Weak criteria require clarification before implementation.

## Workspace And Git

- Do not create a new branch unless explicitly asked.
- Do not create a new worktree.
- Do not modify files outside the current task target.
- Do not reset, checkout, delete, or overwrite user or externally generated uncommitted changes.
- If the worktree is dirty in files related to the task, understand and adapt to the existing changes; ask only when safe progress is not possible.

## Note
Avoid excessive praise when answering questions. Please remember that your answer may not be correct, and my judgment may not be correct either. Carefully consider all questions, prioritizing accuracy. If necessary, you can proactively request supplementary information or evidence from me. Maintain a structured output and ensure your answers are clear and logical.

## MechCAL Project Context

- The user is the author of MechCAL and PhotoMechBench. The AAAI-26 submission was rejected; the target is ACL via ARR (deadline 2027-01-04). Since 2026-11-11 Claude is the chief designer and executor, replacing earlier work done with Codex.
- Treat the Codex handoff's vision as authoritative, but re-verify its design rationale, test counts, "fixed" items and reported results against code and runs before relying on them.
- Vision: evidence should help discover and compare mechanisms, not only explain predictions.
- The task is frozen: SMILES-only input, the fixed pool of 11 mechanism families, a top-3 ranking with evidence.
- Current release: MechCAL 0.3.0 on branch `evidence-evaluation`. As of 2026-11-13, MechCAL development is paused until the dataset is rebuilt; the rebuilt dataset will replace the current 102 test cases entirely. Paper writing comes later.

## Communication

- Reply in Chinese (Traditional), in plain language and with evidence: say who received what, where it failed and what results remain; separate fact, interpretation and speculation; point to code, logs or numbers.
- Never use v1/v2/v3/v4 naming for systems, rubrics, modules, files or in conversation. Use the release version (e.g. 0.3.0) and descriptive names; refer to existing old files as "the 0.2.0 rubric/workflow".

## Design Principles

- Prefer general fixes over per-case patches, and never tune the system to one case's reference answer; but do not over-restrict the system out of fear of errors.
- Keep hard-coding minimal and let the LLM decide. Capability cards describe what a tool computes and what can be varied; they never name a mechanism family or recommend a tool for a purpose. No family-to-test dictionaries. Prompts describe data; they do not impose an interpretation of it.
- Do not let the current dataset drive development. Judge changes first by dataset-independent signals (evidence reliability under perturbation, grounding and fabrication, discriminativeness, repeat stability, protocol and cost); ranking scores are only a rough reference. Do not adopt a change that improves ranking scores but worsens these signals.
- Record every design decision in `docs/dev-decision-log.md`, stating whether it was informed by development results.
- Keep baselines fair: Direct and ReAct get the same inputs as MechCAL (SMILES, structural facts, family definitions, capability cards, cost information, tool gateway).
- Professor–student protocol: the Planner decides the science, the Worker handles technical settings; any scientific-method change (functional, basis, solvent, target state, geometry source) needs the Planner's approval; the Worker retries system errors at most twice, then reports.

## Evaluation Integrity

- Never give hidden references, the rubric or judge output to any evaluated system (Planner, Worker, Critic, baselines). No case identifier or local path may reach a model.
- Test cases are locked: never run an agent or baseline on them before the system (tools, cards, prompts, rubric, judge) is frozen (`--allow-test-case` / `--allow-test` only after freezing).
- A failed or blocked calculation is not evidence for or against a mechanism.
- Result tables need paper-clustered bootstrap confidence intervals (`examples/compare_methods.py`), the floors (best constant top-3: R@3 about 0.48 on 119 cases; structure kNN, leave-one-paper-out) and compute accounting. Still to do before the final experiments: a judge from a different model family, the equal-budget Worker-value ablation (reported regardless of outcome), and a human spot-check of labels.

## Data

- `benchmarks/photomechbench/raw_v2_119` is the frozen original snapshot (its checksums are tested); never edit it.
- Read SMILES and case files only through `mechcal/eval/dataset.py` (the structure-corrected copy `raw_v2_119_corrected`). Structure corrections go into `benchmarks/photomechbench/corrections/structure_corrections.json`; rebuild the copy with `examples/make_corrected_dataset.py` (and check with `--check`). Case 0141 (ferrocene) is marked not runnable.
- Source papers and supporting information are copyrighted and stay outside the repository (`/ai/workspace/PhotoMechBench_sources`).

## Running Experiments

- Secrets: the DeepSeek API key lives only in `~/.config/mechcal/secrets.env`. Never paste it in chat or write it to the repository, TOML files, README, command arguments or commits. Load it with `source ~/.config/mechcal/secrets.env && source env/deepseek.env.sh`. The user's email address is private.
- Models: backbone `deepseek-flash`; during development the judge is the same model at reasoning effort max.
- Large runs (agent batches, baselines, perturbation suites) only in DeepSeek off-peak hours. Peak is Monday–Friday 09:00–12:00 and 14:00–18:00 Beijing time (01:00–04:00 and 06:00–10:00 UTC), excluding Chinese public holidays; everything else is off-peak. The batch and perturbation runners enforce this by default (`--schedule any` overrides); smokes and single cases may run any time. Say when a run will spill into peak hours. deepseek-flash prices per 1M tokens at peak: cache hit $0.006, cache miss $0.30, output $1.20; off-peak is half. Tally tokens and cost with `examples/token_usage.py` (systems and judge separately).
- Do not edit `mechcal/` while a batch is running: every case starts a fresh process that imports the current code.
- Run long jobs detached (`setsid nohup ...`) so they survive the end of a session. Use `python -m mechcal.agent.batch` for batches (it skips finished cases, moves incomplete ones aside, `--grade` grades them).

## Repository Practice

- Commit locally; push only when the user explicitly says so; do not touch `main` without asking.
- Files in the working tree have mode 777: stage with `git -c core.fileMode=false add ...` and check `git diff --cached --summary` for mode changes before committing.
- Do not modify or commit another session's untracked work unless asked.

## Machine

- CPU-only box: 192 cores, 128 GB RAM.
- QC tools: Amesp (`/ai/workspace/tools/Amesp`), Multiwfn (`/ai/workspace/tools/Multiwfn_2026.11.10_bin_Linux_noGUI`), conda envs `mechcal-qc` (xTB, CREST), `mechcal-dftb` (DFTB+ 25.1), `mechcal-pyscf`, `mechcal-pysoc`; DFTB parameters in `/ai/workspace/tools/dftb-params`.
- Local settings are in the gitignored `mechcal.local.toml` (Amesp 1 thread, Multiwfn 16, PySCF 32 threads and 16 GB). External QC routes register only when `MECHCAL_QC_CONFIG` points at a QCConfig JSON (the agent CLI sets it).
