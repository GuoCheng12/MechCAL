"""Observable delivery contracts, not scientific support grades."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from mechcal.schemas.contracts import object_variants


class ObservableRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(pattern=r"^/[^\s]+", description="JSON pointer in structured_results to the needed value, not a filename or prose claim.")
    value_type: Literal["number", "string", "array", "object"]
    unit_path: str | None = Field(default=None, pattern=r"^/[^\s]+")
    unit: str | None = None

    @model_validator(mode="after")
    def paired_unit(self):
        if (self.unit is None) != (self.unit_path is None):
            raise ValueError("Supply both unit_path and unit, or neither.")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return object_variants(handler.resolve_ref_schema(handler(core_schema)), [
            {"unit_path": {"type": "null"}, "unit": {"type": "null"}},
            {"unit_path": {"type": "string", "pattern": r"^/[^\s]+"}, "unit": {"type": "string"}},
        ])
