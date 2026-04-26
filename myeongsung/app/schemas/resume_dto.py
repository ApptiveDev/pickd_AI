from pydantic import BaseModel, Field
from typing import List, Optional, Union

# ── STAR 경험 입력 스키마 ──────────────────────────────────────
class StarContent(BaseModel):
    situation: str = Field(..., description="[S] 상황 - 어떤 배경/맥락에서 발생한 일인지")
    task: str      = Field(..., description="[T] 과제 - 내가 맡은 구체적 역할과 목표")
    action: str    = Field(..., description="[A] 행동 - 내가 취한 구체적 행동과 방법")
    result: str    = Field(..., description="[R] 결과 - 행동으로 얻은 성과 (수치 포함 권장)")

class ExperienceInput(BaseModel):
    id: Optional[str] = Field(
        None,
        description="경험 고유 ID (미입력 시 UUID 자동 생성)"
    )
    title: str    = Field(..., description="경험 제목")
    priority: str = Field(..., pattern="^(상|중|하)$", description="경험 중요도: 상/중/하")
    tags: List[str] = Field(default=[], description="기술/역량 태그 (선택, 추후 AI 자동 태깅)")
    star: StarContent = Field(..., description="STAR 형식 경험 본문")

# ── 응답 스키마 (플랫 구조) ──
class PlacementResult(BaseModel):
    essay_question:           str            = Field(..., description="자소서 문항 원문")
    matched_experience_id:    Optional[Union[str, int]] = Field(None, description="매핑된 경험 ID (문자열 혹은 숫자)")
    matched_experience_title: str            = Field(..., description="매핑된 경험 제목")
    strategy:                 str            = Field(..., description="선택된 SWOT 전략 (SO/ST/WO/WT/N/A)")
    jd_targeting:             str            = Field(..., description="[JD 타겟팅] JD에서 설정한 O/T 근거")
    dynamic_framing:          str            = Field(..., description="[동적 프레이밍] 페르소나 기반 S/W 해석")
    strategy_derivation:      str            = Field(..., description="[전략 도출] 전략 선택 최종 논증")
    writing_guide:            str            = Field(..., description="자소서 작성 가이드라인 및 핵심 키워드")

class PlacementResponse(BaseModel):
    placements: List[PlacementResult]
    errors: List[str] = []
