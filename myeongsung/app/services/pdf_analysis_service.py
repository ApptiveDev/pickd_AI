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
        files = {"document": file_content}
        
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise ValueError(f"Upstage API 호출 중 오류가 발생했습니다: {str(e)}")

    # 3. 추출된 텍스트 및 표 정보 정리 (Markdown 형식으로 추출된 텍스트 활용)
    # Upstage 응답에서 'elements'를 순회하며 텍스트와 표 정보를 수집합니다.
    extracted_content = []
    
    for element in result.get("elements", []):
        category = element.get("category")
        content = element.get("content", {}).get("text", "")
        
        if category == "table":
            # 표의 경우 html 형식이 있다면 활용하고 없으면 텍스트 활용
            html = element.get("content", {}).get("html", "")
            if html:
                extracted_content.append(f"\n[Table]\n{html}\n")
            else:
                extracted_content.append(f"\n[Table Text]\n{content}\n")
        else:
            extracted_content.append(content)

    full_content = "\n".join(extracted_content)

    if not full_content.strip():
        raise ValueError("PDF에서 유의미한 텍스트를 추출하지 못했습니다.")


    # 4. OpenAI 기반 구조화 데이터 추출
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채용 공고 분석 전문가입니다. 주어진 PDF 파싱 텍스트와 표 정보를 면밀히 분석하여, 지정된 스키마에 맞게 필수 정보와 우대/개인화 정보를 추출하세요. 해당하는 정보가 명확하지 않다면 null 또는 기본값을 사용하세요."),
        ("user", "다음 PDF 분석 내용을 바탕으로 채용 정보를 추출해주세요:\n\n{content}")
    ])
    
    chain = prompt | llm.with_structured_output(JobPostingCreate)
    
    try:
        # LangSmith에 'pdf_parsing'이라는 이름으로 추적되도록 config 추가
        structured_result = chain.invoke(
            {"content": full_content},
            config={"run_name": "pdf_parsing"}
        )
        return structured_result
    except Exception as e:
        raise ValueError(f"LLM 데이터 추출 중 오류가 발생했습니다: {str(e)}")
