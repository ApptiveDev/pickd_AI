from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from pydantic import ValidationError
import json
import uuid
from typing import Optional

from app.schemas.resume_dto import ExperienceInput, PlacementResponse
from app.services.resume_service import create_workflow

from app.schemas.job_dto import UrlAnalysisRequest, JobPostingCreate
from app.services.job_analysis_service import analyze_job_url
from app.services.pdf_analysis_service import analyze_job_pdf

router = APIRouter()

workflow = create_workflow()

@router.post("/analyze/url", response_model=JobPostingCreate)
async def analyze_url(request: UrlAnalysisRequest):
    """
    URL을 입력받아 Firecrawl로 마크다운을 추출하고,
    LLM을 통해 11개 필드로 구성된 구조화된 데이터를 반환합니다.
    """
    try:
        result = analyze_job_url(request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/pdf", response_model=JobPostingCreate)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드받아 Azure Document Intelligence로 분석하고,
    LLM을 통해 11개 필드로 구성된 구조화된 데이터를 반환합니다.
    """
    try:
        file_content = await file.read()
        result = analyze_job_pdf(file_content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-and-place", response_model=PlacementResponse)
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
