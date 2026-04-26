from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, Field, ValidationError
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# 애플리케이션 시작 전 .env 환경변수를 자동으로 불러옵니다.
load_dotenv()

from resume_strategist import create_workflow

app = FastAPI(
    title="Resume Strategist API",
    description="JD 분석(PDF/URL) 및 경험 배치를 수행하는 LangGraph 기반 AI 에이전트 API",
    version="1.2.0"
)

workflow = create_workflow()

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


@app.post("/analyze-and-place", response_model=PlacementResponse)
async def analyze_and_place(
    background_tasks: BackgroundTasks,
    jd_pdf: Optional[UploadFile] = File(None, description="채용공고 원문 PDF 파일 (업스테이지 파싱용)"),
    jd_url: Optional[str] = Form(None, description="채용공고 웹페이지 URL (웹 스크래핑용)"),
    experiences_json: str = Form(..., description="사용자 경험 데이터 JSON 문자열"),
    essay_prompts_json: str = Form(..., description="자소서 문항 리스트 JSON 문자열"),
    user_persona: str = Form("", description="지원자 성향/가치관 (예: '빠른 실행과 피보팅을 중시하는 개발자'). 동적 S/W 프레이밍에 사용됩니다."),
):
    """
    JD PDF 혹은 URL 중 하나와, 경험 JSON 목록, 자소서 문항 배열을 받아 LangGraph를 이용해 자소서를 매핑합니다.
    """
    
    # [유효성 검사] PDF나 URL 중 최소 하나는 반드시 존재해야 함
    if not jd_pdf and not (jd_url and jd_url.strip()):
        raise HTTPException(
            status_code=400, 
            detail="jd_pdf (업로드 파일) 또는 jd_url 중 최소 하나는 필수적으로 제공되어야 합니다."
        )

    # 1. JSON 검증 및 STAR → 내부 포맷 변환
    try:
        raw_experiences = json.loads(experiences_json)
        raw_prompts = json.loads(essay_prompts_json)

        validated_experiences = []
        for exp in raw_experiences:
            parsed = ExperienceInput(**exp)

            # UUID 자동 생성 (미입력 시)
            exp_id = parsed.id or str(uuid.uuid4())

            # STAR → LLM용 content 문자열 변환
            s = parsed.star
            content = (
                f"[상황] {s.situation}\n"
                f"[과제] {s.task}\n"
                f"[행동] {s.action}\n"
                f"[결과] {s.result}"
            )

            validated_experiences.append({
                "id":       exp_id,
                "title":    parsed.title,
                "priority": parsed.priority,
                "tags":     parsed.tags,
                "content":  content,       # 내부 LLM 처리용
                "star":     s.model_dump(), # 원본 보존 (추후 DB 저장용)
            })

        if not isinstance(raw_prompts, list):
            raise ValueError("essay_prompts_json 필드는 문자열 배열 형태여야 합니다.")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효하지 않은 JSON 문자열입니다.")
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"입력 데이터 검증 실패: {str(e)}")


    # 2. 우선순위 판별 및 처리 
    # 두 값이 모두 들어올 경우 jd_pdf 분석 결과를 우선 사용
    jd_markdown = ""
    if jd_pdf and jd_pdf.filename:
        jd_content = await jd_pdf.read()
        try:
            # 실제 서비스시엔 바이너리(jd_content)를 Upstage API에 넘기고 반환된 마크다운을 씁니다.
            jd_markdown = jd_content.decode("utf-8")
        except UnicodeDecodeError:
            jd_markdown = "# JD 텍스트 파싱 처리 (더미 마크다운. 실제론 Upstage API에서 넘어왔다고 가정)"


    # 3. LangGraph 상태(State) 설정
    initial_state = {
        "jd_markdown": jd_markdown,
        "jd_url": jd_url,
        "experiences": validated_experiences,
        "prompts": raw_prompts,
        "user_persona": user_persona,
        "jd_context": {},
        "placements": [],
        "remaining_indices": [],
        "errors": []
    }

    # 4. 워크플로우 실행
    try:
        final_state = workflow.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 파이프라인 실행 중 오류 발생: {str(e)}")
        
    # [개선] 한글 유니코드 이스케이프 방지 (ensure_ascii=False 적용)
    final_response = PlacementResponse(
        placements=final_state.get("placements", []),
        errors=final_state.get("errors", [])
    ).model_dump()
    
    return Response(
        content=json.dumps(final_response, ensure_ascii=False),
        media_type="application/json"
    )
