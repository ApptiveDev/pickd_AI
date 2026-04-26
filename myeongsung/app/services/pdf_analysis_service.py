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

    # 3. 추출된 텍스트 및 표 정보 정리 (페이지 정보 포함)
    extracted_content = []
    
    for element in result.get("elements", []):
        page_num = element.get("page")
        category = element.get("category")
        content = element.get("content", {}).get("text", "")
        
        # 페이지 번호 표시 추가 (LLM이 출처를 식별할 수 있도록)
        prefix = f"[Page {page_num}] " if page_num else ""
        
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
        ("system", "당신은 채용 공고 분석 전문가입니다. 주어진 PDF 파싱 내용(페이지 번호 포함)을 면밀히 분석하여, 지정된 스키마에 맞게 정보를 추출하세요. 특히 'citations' 필드에는 각 주요 정보의 근거가 된 페이지 번호와 원문 텍스트 일부를 반드시 포함하여 NotebookLM과 같은 출처 기능을 제공해야 합니다."),
        ("user", "다음 PDF 분석 내용을 바탕으로 채용 정보와 출처(Citations)를 추출해주세요:\n\n{content}")
    ])
    
    chain = prompt | llm.with_structured_output(JobPostingCreate)
    
    try:
        # LangSmith에 'pdf_parsing_with_citations'라는 이름으로 추적되도록 config 추가
        structured_result = chain.invoke(
            {"content": full_content},
            config={"run_name": "pdf_parsing_with_citations"}
        )
        return structured_result
    except Exception as e:
        raise ValueError(f"LLM 데이터 추출 중 오류가 발생했습니다: {str(e)}")

