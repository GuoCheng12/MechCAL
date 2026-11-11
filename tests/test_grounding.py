"""Layer A grounding and metric-validity perturbations on synthetic artifacts (no API)."""

import pytest

from mechcal.eval import perturb
from mechcal.eval.submission import (
    EvaluatorArtifact, ArtifactStore, EvidenceClaim, Prediction, Quantity, QuantitySource,
    StructuralAssertion, Submission, resolve_pointer,
)
from mechcal.eval.grounding import (
    ELIGIBLE, FABRICATION, check_assertion, check_submission, convert, gate_support, statement_numbers,
)

# 4-(dimethylamino)benzonitrile: N1(C2)(C3) c4..c9, C10#N11 (1-based atom ids)
SMILES = "CN(C)c1ccc(cc1)C#N"
OTHER = "O=Cc1ccccc1"


def art(aid, smiles=SMILES, status="success", **outputs):
    return EvaluatorArtifact(artifact_id=aid, case_id="CASE", tool="microscopic.run_baseline_bundle",
                            status=status, molecule_smiles=smiles,
                            outputs=outputs or {"states": [{"f": 0.4213, "e_ev": 4.312}], "dipole_d": 7.9})


def computed(statement, value, pointer, aid="A1", unit="", source_unit=None):
    return EvidenceClaim(statement=statement, kind="computed", quantities=[
        Quantity(name="q", value=value, unit=unit,
                 source=QuantitySource(artifact_id=aid, pointer=pointer, unit=source_unit))])


def submission(*claims, label="ICT_TICT_CT"):
    return Submission(case_id="CASE", mechanism_predictions=[
        Prediction(label=label, rank=1, evidence=list(claims), limitations=["vertical only"]),
        Prediction(label="RIM_RIR_RIV", rank=2, evidence=[], limitations=["none"]),
    ])


@pytest.fixture
def store():
    s = ArtifactStore(case_id="CASE")
    s.add(art("A1"))
    s.add(art("F1", status="failed"))
    s.add(art("W1", smiles=OTHER))
    return s


def outcome(store, claim):
    return check_submission(submission(claim), store, SMILES).predictions[0].claims[0].outcome


def test_pointer_resolution():
    doc = {"a": [{"b~c": 1, "d/e": 2}]}
    assert resolve_pointer(doc, "/a/0/b~0c") == 1 and resolve_pointer(doc, "/a/0/d~1e") == 2
    with pytest.raises(KeyError):
        resolve_pointer(doc, "/a/3")


def test_statement_number_extraction_skips_identifiers():
    text = "S1 at 4.31 eV (B3LYP/6-31G, def2-SVP) has f = 0.42 and C10 is 120 deg; T2 -0.5"
    assert statement_numbers(text) == ["4.31", "0.42", "120", "-0.5"]


def test_units_convert():
    assert convert(1.0, "hartree", "eV") == pytest.approx(27.2114, rel=1e-5)
    assert convert(400.0, "nm", "eV") == pytest.approx(3.0996, rel=1e-4)
    assert convert(1.0, "nm", "Å") == pytest.approx(10.0)
    with pytest.raises(ValueError):
        convert(1.0, "eV", "Debye")


@pytest.mark.parametrize("claim,expected", [
    (computed("S1 oscillator strength is 0.42.", 0.42, "/states/0/f"), "grounded"),
    (computed("S1 lies at 287.5 nm.", 287.5, "/states/0/e_ev", unit="nm", source_unit="eV"), "grounded"),
    (computed("S1 oscillator strength is 0.85.", 0.85, "/states/0/f"), "value_mismatch"),
    (computed("S1 oscillator strength is 0.42.", 0.42, "/states/0/f", aid="NOPE"), "missing_artifact"),
    (computed("S1 oscillator strength is 0.42.", 0.42, "/states/0/f", aid="W1"), "wrong_molecule"),
    (computed("S1 oscillator strength is 0.42.", 0.42, "/states/0/f", aid="F1"), "failed_artifact"),
    (computed("S1 oscillator strength is 0.42.", 0.42, "/states/9/f"), "unresolved_pointer"),
    (computed("S1 oscillator strength is 0.90.", 0.42, "/states/0/f"), "statement_mismatch"),
    (EvidenceClaim(statement="Bright S1 with f = 0.42.", kind="computed"), "unsourced"),
    (EvidenceClaim(statement="Donor and acceptor are para-linked.", kind="structural", atom_ids=[2, 10],
                   assertions=[StructuralAssertion(type="conjugated_path", atoms=[2, 10])]), "structural_ok"),
    (EvidenceClaim(statement="N2 is nitrogen.", kind="structural",
                   assertions=[StructuralAssertion(type="element", atoms=[2], element="C")]), "structural_bad"),
    (EvidenceClaim(statement="Atom 40 is the donor.", kind="structural", atom_ids=[40]), "structural_bad"),
    (EvidenceClaim(statement="The NMe2-aryl bond can rotate.", kind="structural", atom_ids=[2, 4]),
     "structural_unchecked"),
    (EvidenceClaim(statement="Aminobenzonitriles often show TICT.", kind="prior"), "prior"),
])
def test_claim_outcomes(store, claim, expected):
    assert outcome(store, claim) == expected


def test_tolerance_respects_written_precision(store):
    assert outcome(store, computed("f is 0.4.", 0.4, "/states/0/f")) == "grounded"
    assert outcome(store, computed("f is 0.40.", 0.40, "/states/0/f")) == "value_mismatch"


def test_structural_assertions():
    assert check_assertion(StructuralAssertion(type="rotatable_bond", atoms=[2, 4]), SMILES)
    assert not check_assertion(StructuralAssertion(type="rotatable_bond", atoms=[4, 5]), SMILES)
    assert check_assertion(StructuralAssertion(type="aromatic", atoms=[4, 5, 6]), SMILES)
    assert not check_assertion(StructuralAssertion(type="in_ring", atoms=[10]), SMILES)
    assert check_assertion(StructuralAssertion(type="bonded", atoms=[10, 11]), SMILES)


def test_submission_rates_and_gate(store):
    sub = submission(computed("f is 0.42.", 0.42, "/states/0/f"),
                     computed("Dipole 12.0 D.", 12.0, "/dipole_d"),
                     EvidenceClaim(statement="TICT is common.", kind="prior"))
    check = check_submission(sub, store, SMILES)
    assert check.computed_claims == 2 and check.grounded_computed_claims == 1
    assert check.fabrication_rate == 0.5
    pred = check.predictions[0]
    assert pred.fabricated and pred.eligible_claims == [0]
    assert gate_support(1.0, pred) == 0.0
    assert gate_support(1.0, pred, used_claims=[0]) == 1.0
    assert gate_support(0.5, pred, used_claims=[0, 1]) == 0.0
    assert gate_support(0.5, pred, used_claims=[2]) == 0.0  # prior cannot carry a route


def test_case_mismatch_is_rejected(store):
    with pytest.raises(ValueError):
        check_submission(submission(), ArtifactStore(case_id="OTHER"), SMILES)


def test_outcome_sets_are_disjoint():
    assert not FABRICATION & ELIGIBLE


# ---- perturbations ---------------------------------------------------------------------------

def layer_a_scorer(sub, store, smiles):
    """Toy scorer: 1 if a grounded computed claim and no fabrication, 0.5 if only structural."""
    check = check_submission(sub, store, smiles)
    scores = []
    for pred in check.predictions:
        grounded = any(c.outcome == "grounded" for c in pred.claims)
        structural = any(c.outcome in {"structural_ok", "structural_unchecked"} for c in pred.claims)
        base = 1.0 if grounded else 0.5 if structural else 0.0
        scores.append(gate_support(base, pred))
    return scores


def text_scorer(sub, store, smiles):
    """Toy stand-in for a text-only judge: any number-bearing statement earns full credit."""
    return [1.0 if any(statement_numbers(c.statement) for c in p.evidence) else 0.0
            for p in sub.mechanism_predictions]


@pytest.fixture
def real_like(store):
    sub = submission(
        computed("The bright S1 has a high oscillator strength of 0.42.", 0.42, "/states/0/f"),
        EvidenceClaim(statement="Donor N2 is conjugated to the nitrile carbon C10.", kind="structural",
                      atom_ids=[2, 10], assertions=[StructuralAssertion(type="conjugated_path", atoms=[2, 10])]),
    )
    donor = ArtifactStore(case_id="DONOR")
    donor.add(EvaluatorArtifact(artifact_id="B7", case_id="DONOR", tool="microscopic.run_baseline_bundle",
                               status="success", molecule_smiles=OTHER,
                               outputs={"states": [{"f": 0.0123, "e_ev": 3.9}]}))
    return sub, store, donor


def test_each_operator_changes_what_it_claims(real_like):
    sub, store, donor = real_like
    t = perturb.tamper_value(sub, store, 1)
    assert "4.20" in t.submission.mechanism_predictions[0].evidence[0].statement
    assert t.submission.mechanism_predictions[0].evidence[0].quantities[0].value == pytest.approx(4.2)
    assert sub.mechanism_predictions[0].evidence[0].quantities[0].value == 0.42  # original untouched
    s = perturb.swap_molecule(sub, store, 1, donor)
    q = s.submission.mechanism_predictions[0].evidence[0].quantities[0]
    assert q.source.artifact_id.startswith("swap:DONOR") and q.value == pytest.approx(0.0123)
    f = perturb.flip_conclusion(sub, store, 1)
    assert "dark" in f.submission.mechanism_predictions[0].evidence[0].statement
    assert perturb.reorder(sub, store, 2) is None  # nothing to reorder
    with pytest.raises(ValueError):
        perturb.paraphrase(sub, store, 1, rewrite=lambda s: s.replace("0.42", "0.43"))


def test_layer_a_detects_falsification_and_ignores_controls(real_like):
    sub, store, donor = real_like
    summary = perturb.summarize_suite(perturb.run_suite([(sub, store, SMILES, donor)], layer_a_scorer))
    for op in ("tamper_value", "tamper_value_text", "swap_molecule", "strip_artifacts"):
        assert summary[op]["pass_rate"] == 1.0, op
    for op in ("inject_fake_citation", "inject_unsourced"):
        assert summary[op]["pass_rate"] == 1.0, op
    assert summary["paraphrase"]["pass_rate"] == 1.0 and summary["reorder"]["pass_rate"] == 1.0
    # Flipping a direction word is invisible to Layer A by design; it is Layer B's job.
    assert summary["flip_conclusion"]["pass_rate"] == 0.0


def test_text_only_scoring_is_fooled(real_like):
    sub, store, donor = real_like
    summary = perturb.summarize_suite(perturb.run_suite([(sub, store, SMILES, donor)], text_scorer))
    assert summary["tamper_value"]["pass_rate"] == 0.0
    assert summary["swap_molecule"]["pass_rate"] == 0.0


# ---- adapter from persisted MechCAL tool results --------------------------------------------

def test_store_from_mechcal_run_registers_results_and_evidence_aliases(tmp_path):
    from test_tool_sources import execution, save_execution

    from mechcal.eval.submission import store_from_mechcal_run

    save_execution(tmp_path, execution("R001", "scan"))
    save_execution(tmp_path, execution("R002", "scan", status="partial"))
    store = store_from_mechcal_run(tmp_path, "CASE", SMILES)
    ok, partial = store.artifacts["R001:scan"], store.artifacts["R002:scan"]
    assert ok.status == "success" and partial.status == "partial"
    assert store.artifacts["R001:scan:e"].provenance["alias_of"] == "R001:scan"
    pointer = "/structured_results/parsed_observables/route_records/2/oscillator_strength"
    claim = computed("Member 3 is bright with f = 0.4.", 0.4, pointer, aid="R001:scan")
    assert check_submission(submission(claim), store, SMILES).predictions[0].claims[0].outcome == "grounded"
    failed_member = computed("Member 2 has f = 0.0.", 0.0,
                             "/structured_results/parsed_observables/route_records/1/oscillator_strength",
                             aid="R001:scan")
    assert outcome(store, failed_member) == "unresolved_pointer"  # failed member has no value
