# Brief: re-annotate the 102 PhotoMechBench test cases

Paste this into a new Claude Code session (working folder `/ai/workspace/MechCAL`). Guideline §8 requires the test labels to be annotated in a session separate from development.

## Background

- PhotoMechBench labels single molecules (SMILES input) with a fixed pool of 7 mechanism families: RIM_RIR_RIV, ICT_TICT_CT, ESIPT_PT, PET_ET, AGGREGATE_EXCITON_EXCIMER, RADIATIVE_RATE_STATE_BALANCE, TRIPLET_ISC_RTP_TADF.
- The existing 119 cases have labels mechanically mapped from an older 11-family scheme.
- The author decided on 2026-11-16 to re-annotate them before recounting the gaps:
  - target: at least 30 primary labels per family;
  - the largest family stays at or below 15% (the author did not relax this).
- The 17 dev cases were re-annotated in the originating session:
  - 5 were flagged for removal as computation-only sources;
  - one primary label changed;
  - weak secondaries were dropped.

## Task

Re-annotate the 102 test cases from their source papers, following `benchmarks/photomechbench/annotation-guideline-zh.md`, especially:
- §3.2 family rules;
- §3.3–§3.6 boundary cases;
- §4 evidence units;
- §5 contrast groups.

Reply to the user in Traditional Chinese, briefly.

**Read first**
- The report section 「重新標註現有 119 個 case：開發集 17 個」 in `analysis/dataset_expansion/pilot-report-zh.md`.
- Template: `/ai/workspace/PhotoMechBench_sources/expansion_pilot/reannotation/build_reann_dev.py`.
- Library: `/ai/workspace/PhotoMechBench_sources/expansion_pilot/reann_lib.py`.
  - Functions: `pages`, `find`, `dump`, `ev`, `case`, `finish`, `pub_date`.
  - Run with `/home/devbox/miniconda3/envs/mechcal/bin/python`.

**Inputs**
- Test IDs: key `test` in `benchmarks/photomechbench/splits/raw_v2_119_dev_test.json`.
- Current records are read only through `mechcal.eval.dataset.case_record` (the library does this). Old labels are only pointers; annotate from the paper.
- PDFs and the DOI→case map: `/ai/workspace/PhotoMechBench_sources/PhotoMechBench_source_papers_76_20261113/` (`pdfs/`, `paper_case_mapping.csv`).
- SI exists only for 0004, 0038 and 0102 (`/ai/workspace/PhotoMechBench_sources/SI_20261113/`). `ev()` searches the main PDF only; extend it minimally if you quote SI.

**Steps and checks**
1. Work paper by paper and write builders `reannotation/build_reann_test_*.py` using `case()`.
   - For each family, record one of: supported (with role), contradicted (paper or structure basis), or not_established.
   - Add conditions, environment_context, out_of_pool and contrast groups where the paper supports them.
   - `finish()` must report no quote misses. Private-use or control glyphs (ligatures, dots) break matching; start the quote after them.
2. Computation-only sources (experimental data only from earlier literature): set `removal=` with the reason and still write the labels. The author decides on removal.
3. Output goes to `reannotation/cases/<case_id>.json`. Do not touch the dev files already there.
4. Recount when done. Include:
   - the 17 re-annotated dev cases;
   - the re-annotated test cases, minus suggested removals;
   - the 109 new cases in `/ai/workspace/PhotoMechBench_sources/expansion_pilot/cases/` (PILOT_*, EXPA_*).

   For each family give:
   - the primary/co_primary count;
   - the gap to 30;
   - the share of all primary labels (cap 15%);
   - the count by paper.
5. Write `analysis/dataset_expansion/reannotation-test-zh.md` (Traditional Chinese, short):
   - the counts;
   - removals, with IDs and reasons;
   - primary-label changes, with a few examples;
   - the recount table;
   - guideline gaps you met, as proposals.

**Done when**
- All 102 are re-annotated, or there is a written reason why one could not be done.
- The builders run clean.
- The report exists.

**Do not**
- Edit `raw_v2_119*`, the corrected dataset, `mechcal/`, the guideline, or `pilot-report-zh.md`.
- Run agents, baselines or judges, or do any MechCAL development (test labels must stay apart from system development).
- Commit, push or switch branches.
- Copy source papers or new labels into the repository; only the summary report goes in.
- Send personal or institutional details to OpenAlex.
