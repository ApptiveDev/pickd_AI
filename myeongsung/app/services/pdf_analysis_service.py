import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas.job_dto import JobPostingCreate

def analyze_job_pdf(file_content: bytes) -> JobPostingCreate:
    """
    Upstage Document Parse를 이용해 PDF에서 텍스트 및 표를 추출하고,
    LLM을 통해 구조화된 채용 공고 데이터(JobPostingCreate)로 변환합니다.
    """
    # 1. Upstage API 설정
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")

    # 2. PDF 분석 실행 (Upstage Layout Analysis API 호출)
    try:
        url = "https://api.upstage.ai/v1/document-ai/layout-analysis"
        headers = {"Authorization": f"Bearer {api_key}"}
        # 한글 파일명 등으로 인한 latin-1 인코딩 에러 방지를 위해 파일명을 'document.pdf'로 고정
        files = {"document": ("document.pdf", file_content, "application/pdf")}
        
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise ValueError(f"Upstage API 호출 중 오류가 발생했습니다: {str(e)}")

    # 3. 추출된 텍스트 및 표 정보 정리 (페이지 정보 및 요소 ID 포함)
    extracted_content = []
    element_map = {} # ID로 요소 정보를 찾기 위한 맵
    
    for idx, element in enumerate(result.get("elements", [])):
        page_num = element.get("page")
        category = element.get("category")
        content = element.get("content", {}).get("text", "")
        
        # 요소 정보 저장 (나중에 bbox 매핑용)
        element_map[idx] = element
        
        # 페이지 및 ID 표시 추가 (LLM이 출처를 식별할 수 있도록)
        prefix = f"[ID:{idx}, Page:{page_num}] " if page_num else f"[ID:{idx}] "
        
        if category == "table":
            html = element.get("content", {}).get("html", "")
            if html:
                extracted_content.append(f"\n{prefix}[Table]\n{html}\n")
            else:
                extracted_content.append(f"\n{prefix}[Table Text]\n{content}\n")
        else:
            extracted_content.append(f"{prefix}{content}")

    full_content = "\n".join(extracted_content)

    if not full_content.strip():
        raise ValueError("PDF에서 유의미한 텍스트를 추출하지 못했습니다.")

    # 4. OpenAI 기반 구조화 데이터 추출
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채용 공고 분석 전문가입니다. 주어진 PDF 파싱 내용(ID 및 페이지 번호 포함)을 면밀히 분석하여 정보를 추출하세요. 특히 'citations' 필드에는 해당 정보의 근거가 된 'element_id'와 'page', 'content'를 정확히 입력하여 NotebookLM과 같은 출처 기능을 제공해야 합니다."),
        ("user", "다음 PDF 분석 내용을 바탕으로 채용 정보와 출처(Citations)를 추출해주세요:\n\n{content}")
    ])
    
    chain = prompt | llm.with_structured_output(JobPostingCreate)
    
    try:
        # LangSmith에 'pipe-analy'이라는 이름으로 추적되도록 config 추가
        structured_result = chain.invoke(
            {"content": full_content},
            config={"run_name": "pipe-analy"}
        )
        
        # 5. 출처(Citations) 보완 (bbox 매핑 및 페이지 이동 링크 추가)
        if structured_result.citations:
            for citation in structured_result.citations:
                # bbox 매핑
                if citation.element_id is not None and citation.element_id in element_map:
                    el = element_map[citation.element_id]
                    
                    # Upstage 좌표 필드는 'coordinates' 또는 'bounding_box'일 수 있음
                    raw_coords = el.get("coordinates") or el.get("bounding_box") or []
                    
                    if raw_coords:
                        xs, ys = [], []
                        # 1. [{"x": 1, "y": 2}, ...] 형식인 경우
                        if isinstance(raw_coords[0], dict):
                            xs = [p.get("x") for p in raw_coords if "x" in p]
                            ys = [p.get("y") for p in raw_coords if "y" in p]
                        # 2. [x1, y1, x2, y2, ...] 단순 리스트인 경우
                        elif isinstance(raw_coords[0], (int, float)):
                            xs = [v for i, v in enumerate(raw_coords) if i % 2 == 0]
                            ys = [v for i, v in enumerate(raw_coords) if i % 2 != 0]
                        
                        if xs and ys:
                            # [x1, y1, x2, y2] 형태로 정규화하여 반환
                            citation.bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                
                # 페이지 이동 링크
                if citation.page > 0:
                    citation.source_url = f"#page={citation.page}"

        
        return structured_result
    except Exception as e:
        raise ValueError(f"LLM 데이터 추출 중 오류가 발생했습니다: {str(e)}")


