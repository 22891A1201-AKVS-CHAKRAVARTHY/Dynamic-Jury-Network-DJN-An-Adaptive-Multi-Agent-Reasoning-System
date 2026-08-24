# djn_engine/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .audit import stable_hash

Category = Literal["coding", "career", "planning", "factual", "opinion", "mathematical", "general"]
Role = Literal["PROPOSER", "CRITIC", "REFINER", "RISK"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]

def _norm_label(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")[:64] or "UNKNOWN"

    POS = {"YES", "APPROVE", "RECOMMEND", "RECOMMENDED", "GO", "GO_AHEAD", "DO_IT", "AGREE", "SUPPORT"}
    NEG = {"NO", "REJECT", "AVOID", "DISAGREE", "OPPOSE"}
    COND = {"CONDITIONAL", "DEPENDS", "MAYBE", "PARTIAL", "MIXED", "QUALIFIED", "YES_BUT", "CONDITIONAL_YES"}
    UNK = {"UNKNOWN", "UNCLEAR", "NOT_SURE", "INSUFFICIENT_INFO"}

    if s in POS or s.startswith("YES"):
        return "YES"
    if s in NEG or s.startswith("NO"):
        return "NO"
    if s in COND or "CONDITIONAL" in s or "DEPENDS" in s or "MAYBE" in s:
        return "CONDITIONAL"
    if s in UNK:
        return "UNKNOWN"
    return s  

class ModeratorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    category_confidence: float = Field(ge=0.0, le=1.0)
    missing_critical: List[str] = Field(default_factory=list)
    clarifier_questions: List[str] = Field(default_factory=list, max_length=3)
    
class AssumptionsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q_final: str = Field(min_length=1)
    assumptions: List[str] = Field(default_factory=list)

class JurorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict_label: str
    tldr: str
    reasoning: List[str] = Field(min_length=3, max_length=20)

    @field_validator("verdict_label")
    @classmethod
    def verdict_label_norm(cls, v: str) -> str:
        return _norm_label(v)

    @field_validator("tldr")
    @classmethod
    def tldr_cap(cls, v: str) -> str:
        v = (v or "").strip()
        words = v.split()
        return " ".join(words[:90])

class RoundSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    common_ground: List[str] = Field(default_factory=list)
    key_disagreements: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    current_best_label: str
    why_this_label: str

    @field_validator("current_best_label")
    @classmethod
    def best_label_norm(cls, v: str) -> str:
        return _norm_label(v)


class JudgeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_recommendation: str = Field(min_length=40, max_length=1200)
    why: list[str] = Field(min_length=2, max_length=6)
    confidence: str  
    common_ground: list[str] = Field(default_factory=list, max_length=8)
    main_disagreement: list[str] = Field(default_factory=list, max_length=6)
    conditional_guidance: list[str] = Field(default_factory=list, max_length=8)


class CallStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    err: Optional[str] = None
    raw: Optional[str] = None

class JurorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    juror_id: str
    model_id: str
    output: Optional[JurorOut] = None
    status: CallStatus

class RoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int
    outputs: List[JurorResult]
    agreement: float
    majority_label: str
    improvement: Optional[float] = None


class ExperimentConfig(BaseModel):
    """Immutable, serializable definition of one DJN experimental condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal[
        "single_model",
        "static_jury_one_round",
        "static_jury_multi_round",
        "full_djn",
    ] = "full_djn"
    selector_mode: Literal["dynamic", "fixed"] = "dynamic"
    role_mode: Literal["conditioned", "generic"] = "conditioned"
    handoff_mode: Literal["structured", "raw", "none"] = "structured"
    stopping_mode: Literal["dynamic", "fixed_rounds"] = "dynamic"
    synthesis_mode: Literal["judge", "majority"] = "judge"
    model_pool: List[str] = Field(default_factory=list)
    fixed_roster: List[str] = Field(default_factory=list)
    jury_size: int = Field(default=4, ge=1, le=20)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_rounds: int = Field(default=3, ge=1, le=20)
    min_ok_jurors: int = Field(default=2, ge=1, le=20)
    min_improvement: float = Field(default=0.05, ge=0.0, le=1.0)
    stagnation_rounds: int = Field(default=1, ge=1, le=20)
    temperature: float = Field(default=0.35, ge=0.0, le=2.0)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    seed: int = 0
    max_prompt_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    selector_version: str = "selector-v2"
    capability_version: str = "capabilities-v1"
    price_version: str = "prices-v1"

    def snapshot(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def config_id(self) -> str:
        return stable_hash(self.snapshot())

    @field_validator("fixed_roster")
    @classmethod
    def fixed_roster_is_unique(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("fixed_roster cannot contain duplicate model IDs")
        return value

    def validate_combination(self) -> None:
        if self.selector_mode == "fixed" and not self.fixed_roster:
            raise ValueError("fixed selector mode requires fixed_roster")
        if self.mode == "single_model" and (
            self.selector_mode != "fixed" or len(self.fixed_roster) != 1 or self.jury_size != 1
        ):
            raise ValueError("single_model mode requires jury_size=1 and exactly one fixed model")
        if self.mode.startswith("static_jury") and self.selector_mode != "fixed":
            raise ValueError("static jury modes require fixed selector mode")
        if self.selector_mode == "fixed" and len(self.fixed_roster) != self.jury_size:
            raise ValueError("fixed_roster length must equal jury_size")
        if self.min_ok_jurors > self.jury_size:
            raise ValueError("min_ok_jurors cannot exceed jury_size")
        if self.mode == "static_jury_one_round" and self.max_rounds != 1:
            raise ValueError("static_jury_one_round requires max_rounds=1")
        if self.handoff_mode == "none" and self.max_rounds > 1 and self.mode == "full_djn":
            # Valid as an ablation, but it must be declared explicitly; reaching here proves it was.
            return
