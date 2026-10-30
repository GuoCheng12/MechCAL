"""Model-authored question changes with legacy persistence compatibility."""

from pydantic import ValidationError

from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.judgment_commands import QuestionCreate
from mechcal.schemas.mechanisms import EvidenceQuestion

CREATE_FIELDS = tuple(name for name, field in QuestionCreate.model_fields.items()
                      if field.is_required() and name not in {"operation", "question_id"})


def question_view(question):
    """Legacy bookkeeping stays readable on disk, not on the model's write surface."""
    return {k: v for k, v in question.items()
            if k not in {"status", "execution_state", "schema_version", "record_type"}}


def public_questions(case):
    questions = case.runtime.get("question_loop", {}).get("questions", {})
    return {qid: question_view(q) if case.runtime.get("incremental_judgment") else q
            for qid, q in questions.items()}


def apply_question_changes(previous, changes):
    records = {qid: EvidenceQuestion.model_validate(q).model_copy(deep=True)
               for qid, q in previous.items()}
    seen = set()
    for index, change in enumerate(changes):
        qid = change.question_id

        def require(condition, message, field=None):
            if not condition:
                raise RequestValidationError([{
                    "path": ["questions", index] + ([field] if field else []),
                    "record_selector": {"question_id": qid},
                    "code": "question_change", "message": message,
                }])

        require(qid not in seen, "Submit each question_id once.")
        seen.add(qid)
        fields = change.model_dump(exclude={"operation", "question_id"}, exclude_none=True)
        require(bool(fields), "Submit changed fields or omit this question.")
        if change.operation == "create":
            require(qid not in records, "This question already exists; use operation=update.", "operation")
            missing = [field for field in CREATE_FIELDS if field not in fields]
            if missing:
                raise RequestValidationError([{
                    "path": ["questions", index, field], "type": "missing",
                    "record_selector": {"question_id": qid}, "code": "question_change",
                    "message": f"New question {qid} requires {field}.",
                } for field in missing])
            base = {"question_id": qid}
        else:
            require(qid in records, "Unknown question_id; use operation=create with all required fields.", "operation")
            base = records[qid].model_dump()
        try:
            records[qid] = EvidenceQuestion.model_validate({**base, **fields})
        except ValidationError as exc:
            raise RequestValidationError([{
                "path": ["questions", index, *error["loc"]],
                "record_selector": {"question_id": qid}, "code": "question_change",
                "message": error["msg"],
            } for error in exc.errors(include_url=False)]) from exc
    return list(records.values())
