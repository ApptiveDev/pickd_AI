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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채용 공고 분석 전문가입니다. 주어진 채용 공고의 마크다운 텍스트를 면밀히 분석하여, 지정된 스키마에 맞게 정보를 추출하세요. 특히 'citations' 필드에는 각 주요 정보의 근거가 된 원문 텍스트 일부를 포함해야 합니다. URL 분석이므로 페이지 번호(page)는 모두 0으로 설정하세요."),
        ("user", "다음 채용 공고를 분석하고 출처(Citations)를 포함하여 결과를 추출해주세요:\n\n{markdown}")
    ])
    
    # with_structured_output를 사용하여 Pydantic DTO 형식(citations 포함)으로 추출
    chain = prompt | llm.with_structured_output(JobPostingCreate)
    
    try:
        # LangSmith에 'firecrawl'이라는 이름으로 추적되도록 config 추가
        result = chain.invoke(
            {"markdown": markdown_content},
            config={"run_name": "firecrawl"}
        )
        
        # 3. 출처(Citations)에 웹 하이라이트 링크(Text Fragment) 추가
        from urllib.parse import quote
        
        if result.citations:
            for citation in result.citations:
                # 브라우저의 'Scroll to Text Fragment' 기능 활용 (#:~:text=문구)
                # 문구가 너무 길면 인코딩 문제가 생길 수 있으므로 적절히 처리
                safe_text = quote(citation.content.replace("\n", " ").strip())
                citation.source_url = f"{url}#:~:text={safe_text}"
                
        return result

    except Exception as e:
        raise ValueError(f"LLM 데이터 추출 중 오류가 발생했습니다: {str(e)}")
