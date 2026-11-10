# Source-bound numerical checks

Use reduce_evidence for a sum or extremum and compare_evidence for a ratio or difference
instead of estimating these numbers from prose. Select an exact source ref and
JSON pointer. For min/max, path selects the complete array for the intended
condition and value_path selects the numeric field within each row, for example
/oscillator_strength. Empty value_path selects the numeric value itself.
For a fragment total, use reduction=sum and explicit row_indices from the original
array after checking its atom_index fields. A small per-atom maximum does not imply
a small fragment sum. Null row_indices includes the whole array, not an inferred
fragment. Subsets describe only the selected atoms, states or sampled geometries.
compare_evidence can reduce both operands in one query. These operations share
the existing read allowance and do not run scientific calculations.

Use the returned values, tied rows, conditions and exact source paths. Extrema
over different roots do not track the same physical state. Arithmetic does not
establish comparable methods, units, conformers or causal mechanisms. Explain
these limits in natural language. Cite the original records, not the numeric
receipt as independent evidence. If the required field or condition is unknown,
retrieve it or state the uncertainty; do not invent a numeric comparison.
