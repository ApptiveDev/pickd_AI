import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.schemas.job_dto import JobPostingCreate

import requests

def analyze_job_url(url: str) -> JobPostingCreate:
    """
    Firecrawl을 이용해 URL에서 Markdown을 추출하고,
    LLM을 통해 구조화된 채용 공고 데이터(JobPostingCreate)로 변환합니다.
    """
    # 1. Firecrawl로 URL 스크래핑
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY 환경변수가 설정되지 않았습니다.")
        
    try:
        response = requests.post(
            'https://api.firecrawl.dev/v2/scrape',
            headers={
                'Authorization': f'Bearer {firecrawl_api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'url': url,
                'formats': ['markdown']
            },
            timeout=30
        )
        response.raise_for_status()
        scrape_result = response.json()
        
        # v2 API 응답 형식: {"success": true, "data": {"markdown": "..."}}
        if not scrape_result.get("success"):
            raise ValueError(f"Firecrawl API 실패: {scrape_result.get('error', '알 수 없는 오류')}")
            
        markdown_content = scrape_result.get("data", {}).get("markdown", "")
    except Exception as e:
        raise ValueError(f"Firecrawl API 호출 중 오류가 발생했습니다: {str(e)}")
    
    if not markdown_content:
        raise ValueError("해당 URL에서 마크다운 텍스트를 추출하지 못했습니다.")

    # 2. OpenAI 기반 구조화 데이터 추출
    # .env 파일에 OPENAI_API_KEY 가 설정되어 있어야 합니다.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채용 공고 분석 전문가입니다. 주어진 채용 공고의 마크다운 텍스트를 면밀히 분석하여, 지정된 스키마에 맞게 필수 정보와 우대/개인화 정보를 추출하세요. 해당하는 정보가 명확하지 않다면 null 또는 기본값을 사용하세요."),
        ("user", "다음 채용 공고를 분석해주세요:\n\n{markdown}")
    ])
    
    # with_structured_output를 사용하여 Pydantic DTO 형식으로 완벽하게 추출
    chain = prompt | llm.with_structured_output(JobPostingCreate)
    
    try:
        result = chain.invoke({"markdown": markdown_content})
        return result
    except Exception as e:
        raise ValueError(f"LLM 데이터 추출 중 오류가 발생했습니다: {str(e)}")
