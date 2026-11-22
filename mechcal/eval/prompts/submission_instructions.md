Task: from the verified SMILES only, state which photophysical mechanism families
from the fixed pool are reported for this molecule, and give molecule-specific
evidence and limitations for each. Families can overlap. No experimental data,
compound name or source paper is available.

Submit one to seven predictions with distinct labels and ranks 1, 2, ... in order
of plausibility; rank 1 is your main mechanism. Claim a family only if you judge it
applies: a claimed family that is not reported counts against you, and so does a
reported family you leave out. Each prediction has a list of evidence claims and a
list of limitations. Every claim is
one of three kinds:

- "structural": a fact about this molecule's graph (groups, connectivity,
  conjugation, rings, rotatable bonds, donor/acceptor sites). Atom IDs are one-based
  in SMILES atom order and are optional. Do not put numbers in quantities for
  structural claims.
- "computed": a result of a tool call you actually made. Each number you write in
  the statement must also appear in "quantities" with its source: the artifact_id
  returned by the tool and the exact JSON pointer shown in the tool's values (or
  obtained with read_artifact). Use the unit of the statement; if the artifact
  stores another unit, give it as source.unit. Never cite a value you did not
  receive from a tool.
- "prior": general chemical knowledge or literature you remember. It may explain
  other claims but is never case-specific evidence and must not carry numbers.

Be accurate rather than exhaustive. A claim that misstates a value, cites the wrong
artifact, or presents a failed or modeled calculation as something else counts
against you. Interpretations must agree with the cited numbers (for example, do not
call a state dark while citing a large oscillator strength). State limitations
honestly: vertical vs relaxed states, isolated molecule vs aggregate, modeled
dimer vs crystal.

Prediction format:
{"label": "<pool label>", "rank": 1,
 "evidence": [
   {"statement": "...", "kind": "structural", "atom_ids": [2, 10]},
   {"statement": "S1 oscillator strength is 0.42.", "kind": "computed",
    "quantities": [{"name": "S1_oscillator_strength", "value": 0.42, "unit": "",
                    "source": {"artifact_id": "T003", "pointer": "/structured_results/..."}}]},
   {"statement": "...", "kind": "prior"}
 ],
 "limitations": ["..."]}
