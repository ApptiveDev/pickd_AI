from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ValidationError
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 애플리케이션 시작 전 .env 환경변수를 자동으로 불러옵니다.
load_dotenv()

from resume_strategist import create_workflow

app = FastAPI(
    title="Resume Strategist API",
    description="JD 분석(PDF/URL) 및 경험 배치를 수행하는 LangGraph 기반 AI 에이전트 API",
    version="1.1.0"
)

workflow = create_workflow()

class ExperienceInput(BaseModel):
    id: int
    title: str
    content: str
    tags: List[str] = []
    priority: str = Field(pattern="^(상|중|하)$", description="'상', '중', '하' 중 하나로 입력")

class PlacementResult(BaseModel):
    question: str
    experience_id: Optional[int] = None
    experience_title: str
    selected_strategy: str
    reasoning: str
    writing_guide: str

class PlacementResponse(BaseModel):
    placements: List[PlacementResult]
    errors: List[str] = []


@app.post("/analyze-and-place", response_model=PlacementResponse)
async def analyze_and_place(
    background_tasks: BackgroundTasks,
    jd_pdf: Optional[UploadFile] = File(None, description="채용공고 원문 PDF 파일 (업스테이지 파싱용)"),
    jd_url: Optional[str] = Form(None, description="채용공고 웹페이지 URL (웹 스크래핑용)"),
    experiences_json: str = Form(..., description="사용자 경험 데이터 JSON 문자열"),
    essay_prompts_json: str = Form(..., description="자소서 문항 리스트 JSON 문자열")
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

    # 1. JSON 검증
    try:
        raw_experiences = json.loads(experiences_json)
        raw_prompts = json.loads(essay_prompts_json)
        
        validated_experiences = []
        for exp in raw_experiences:
            validated_experiences.append(ExperienceInput(**exp).model_dump())
            
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
        
    return PlacementResponse(
        placements=final_state.get("placements", []),
        errors=final_state.get("errors", [])
    )
