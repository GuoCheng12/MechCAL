You are correcting a rejected Planner response, not making a new scientific decision.
Return only a JSON object with a repairs list. Each entry has path (an array of
field names and list indices) and value (the replacement value for that field).
Use only the exact paths in allowed_paths. Do not return a full assimilation or
change any other field. The program keeps all other content exactly as submitted.
The output_contract defines the complete value type for each allowed path, with
its referenced definitions in $defs. Follow that field's schema, not the shape of
historical audit entries. Program-generated revision, observed_change and
previous_argument are not fields of an ArgumentChange and must not be submitted.
If last_failed_correction is present, inspect its submitted_correction and errors.
Correct that failure as well as the original issues; do not repeat a rejected shape.

Address every reported issue together. When change records are missing, describe
the supplied before/after differences honestly, including changes to limitations.
Do not invent a scientific justification. Copy reference IDs exactly from the
case-specific allowed_refs, selecting the actual result intended by the argument.
Do not substitute a merely similar or unrelated result, or delete a valid citation
just to pass validation. An unknown ID does not authorize new evidence.

For argument changes, inspect retained_previous_ids and removed_previous_ids.
A retained ID can be revised, with replacement_ids=[] or [that same ID]. A newly
added argument is separate, not a replacement for a retained argument. Only an
absent old ID can be withdrawn or superseded. Keep the submitted arguments and
ranking fixed; correct the change record to describe what actually changed.

The assembled response is checked again against every original constraint.
This correction uses the same bounded retry allowance as the original request.
