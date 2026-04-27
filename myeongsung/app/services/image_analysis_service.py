import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.schemas.job_dto import JobPostingCreate

from typing import List

def analyze_job_image(image_bytes_list: List[bytes]) -> JobPostingCreate:
    """
    여러 장의 이미지 파일을 Gemini 1.5 Flash를 통해 통합 분석하고 구조화된 데이터를 추출합니다.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

    # Gemini 1.5 Flash 모델 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0
    )

    # 모든 이미지를 base64로 인코딩하여 메시지 구성
    image_parts = []
    for image_bytes in image_bytes_list:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # 시스템 프롬프트 추가
    text_part = {
        "type": "text",
        "text": (
            "당신은 채용 공고 분석 전문가입니다. 제공된 모든 이미지들을 순서대로 분석하여 하나의 통합된 채용 정보를 추출하세요.\n"
            "다음 11개 필드를 포함한 JSON 형식으로 답변해야 하며, 'citations' 필드에는 각 정보의 근거가 된 원문 텍스트를 포함하세요.\n"
            "이미지 분석이므로 citations의 page는 이미지 순서(1부터 시작)로 설정하고, bbox는 알 수 없다면 null로 두세요."
        )
    }

    # 전체 메시지 구성 (텍스트 설명 + 이미지들)
    message = HumanMessage(content=[text_part] + image_parts)

    # 구조화된 출력(Structured Output) 사용
    structured_llm = llm.with_structured_output(JobPostingCreate)

    try:
        # LangSmith 추적을 위한 run_name 설정
        result = structured_llm.invoke([message], config={"run_name": "pipe-analy"})
        return result
    except Exception as e:
        raise ValueError(f"Gemini 이미지 통합 분석 중 오류가 발생했습니다: {str(e)}")

