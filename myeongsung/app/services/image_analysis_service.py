import os
import io
from typing import List
from google import genai
from google.genai import types
from PIL import Image
from app.schemas.job_dto import JobPostingCreate

def analyze_job_image(image_bytes_list: List[bytes]) -> JobPostingCreate:
    """
    구글의 최신 google-genai SDK와 Gemini 2.0 Flash-Lite 모델을 사용하여
    여러 장의 이미지를 통합 분석하고 구조화된 데이터를 추출합니다.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

    # 1. GenAI 클라이언트 초기화
    client = genai.Client(api_key=google_api_key)

    # 2. 이미지 및 텍스트 프롬프트 구성
    contents = []
    
    # 시스템 지시사항 및 텍스트 프롬프트
    prompt_text = (
        "당신은 채용 공고 분석 전문가입니다. 제공된 모든 이미지들을 순서대로 분석하여 하나의 통합된 채용 정보를 추출하세요.\n"
        "다음 필드들을 반드시 포함해야 합니다: company_name, job_title, qualifications, industry, application_period, "
        "essay_question_count, work_location, preferred_qualifications, extra_points, evaluation_criteria, salary.\n"
        "또한 'citations' 필드에는 각 정보의 근거가 된 원문 텍스트를 포함하고, page는 이미지 순서(1부터 시작)로 기록하세요."
    )
    contents.append(prompt_text)

    # 이미지 바이트들을 SDK 형식으로 변환하여 추가
    for image_bytes in image_bytes_list:
        # PIL을 사용해 이미지 유효성 확인 (선택 사항)
        img = Image.open(io.BytesIO(image_bytes))
        contents.append(img)

    # 3. Gemini 2.5 Flash-Lite 모델 호출 (구조화된 출력 설정)
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=JobPostingCreate.model_json_schema(), # Pydantic 스키마 전달
                temperature=0,
            )
        )

        
        # 4. 응답 데이터를 Pydantic 객체로 변환
        # response.text는 JSON 문자열이므로 파싱하여 반환
        import json
        result_dict = json.loads(response.text)
        return JobPostingCreate(**result_dict)

    except Exception as e:
        raise ValueError(f"Gemini 2.0 이미지 분석 중 오류가 발생했습니다: {str(e)}")
