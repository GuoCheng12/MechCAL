"""Numerical SOC output contract; no mechanism or evidence-grade decisions."""

from typing import Annotated, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator
from mechcal.tools.pysoc_source import SOURCE_HASHES

Finite = Annotated[float, Field(allow_inf_nan=False, ge=0)]
METHOD = "TD-DFTB/mio-1-1 + PySOC ZEFF"


class SOCPair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    singlet: int = Field(ge=0, le=3)
    triplet: int = Field(ge=1, le=3)
    rss_cm_inverse: Finite
    components_cm_inverse: list[Finite] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def rss_agrees(self):
        expected = math.sqrt(sum(v*v for v in self.components_cm_inverse))
        if abs(self.rss_cm_inverse-expected)>1e-10:
            raise ValueError("SOC magnitude disagrees with its components.")
        return self


class RawSOC(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["pysoc_zeff_v1"]
    status: Literal["success"]
    pysoc_version: Literal["2.3.0"]
    method: Literal["TD-DFTB/mio-1-1 + PySOC ZEFF"]
    unit: Literal["cm^-1"]
    operator: Literal["ZEFF"]
    scale: Literal[1.0]
    ci_threshold: Literal[0.00001]
    canonical_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes: dict[str,str]
    couplings: list[SOCPair] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def complete_states_and_source(self):
        if {(r.singlet,r.triplet) for r in self.couplings}!={(s,t) for s in range(4) for t in range(1,4)}:
            raise ValueError("SOC pairs are missing or duplicated.")
        if self.source_hashes != SOURCE_HASHES:
            raise ValueError("SOC numerical implementation differs from the validated source.")
        return self
