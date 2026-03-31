import json
import re
import os
import requests
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# ------------------------------------------------------------------
# 1. Pydantic을 이용한 출력 스키마 정의
# ------------------------------------------------------------------
class CertificateBonus(BaseModel):
    certificate_name: str = Field(description="자격증 이름")
    score: str = Field(description="부여되는 가산점 점수 (예: 5점, 3%)")
    target_group: str = Field(description="해당 가산점이 적용되는 직군/계열")
    evidence_page: int = Field(description="정보가 발견된 실제 페이지 번호")
    original_text: str = Field(description="해당 점수를 증명하는 실제 공고문 속 핵심 문구")

class CertificateExtractionResult(BaseModel):
    bonuses: List[CertificateBonus]

# ------------------------------------------------------------------
# 2. STEP 1: Upstage 결과 전처리 함수
# ------------------------------------------------------------------
def preprocess_upstage_result(upstage_json: dict) -> str:
    """
    Upstage Layout Analyzer의 JSON 결과물을 순회하며 텍스트를 재구성합니다.
    페이지가 바뀔 때마다 [Page: N] 마커를 텍스트 사이에 삽입하며,
    표(Table)의 경우 HTML 구조를 유지하여 LLM이 행렬 관계를 파악할 수 있도록 합니다.
    """
    elements = upstage_json.get('elements', [])
    processed_text = []
    
    current_page = None
    
    for el in elements:
        # Upstage 응답 내 page 파라미터 추출
        page = el.get('page')
        
        # 엘리먼트의 페이지가 변경된 경우 [Page: N] 마커 삽입
        if page is not None and page != current_page:
            processed_text.append(f"\n[Page: {page}]\n")
            current_page = page
            
        category = el.get('category', 'text')
        
        # v1 / v2 응답 구조의 차이를 감안하여 content 유무 호환 처리
        content_obj = el.get('content', el)
        content_html = content_obj.get('html', '')
        content_text = content_obj.get('text', '')
        
        # 표(table) 엘리먼트일 경우 HTML 구조를 살려 Markdown/텍스트화
        if category == 'table' and content_html:
            processed_text.append(f"<table>\n{content_html}\n</table>\n")
        elif content_text:
            processed_text.append(content_text + "\n")
        elif content_html:
            processed_text.append(content_html + "\n")
            
    return "".join(processed_text)

# ------------------------------------------------------------------
# 3. STEP 2 & 3: LLM 분석 요청 및 출력 매핑 함수
# ------------------------------------------------------------------
def extract_certificate_bonuses(openai_key: str, processed_text: str) -> List[dict]:
    """
    전처리된 텍스트를 OpenAI(GPT) API에 전달하여 구조화된 JSON 데이터로 추출합니다.
    (OpenAI Python SDK 1.40.0+ 의 Structured Outputs(parse) 기능을 활용)
    """
    client = OpenAI(api_key=openai_key)
    
    system_prompt = """
    당신은 공기업 및 공공기관 채용 공고문 분석 전문가입니다.
    주어진 공고문 텍스트에서 '자격증 가산점' 관련 정보를 한 치의 오차도 없이 추출해야 합니다.
    
    [가이드라인 및 제약사항]
    1. 모든 가산점 항목은 반드시 제공된 텍스트 내 `[Page: N]` 마커를 근거로 추출하고, 해당 페이지 번호를 `evidence_page`에 기입하세요.
    2. 데이터 속 표(Table)로 된 가산점 정보를 분석할 때는 열(Column)과 행(Row)의 관계를 정확히 파악하여, 잘못된 점수가 다른 자격증/직군에 매칭되지 않도록 극도로 주의하세요.
    3. 만약 특정 자격증, 점수 규칙, 대상자 등의 정보가 두 페이지(예: Page 3 ~ Page 4)에 걸쳐 연결되어 있다면, `evidence_page`는 그 내용이 시작되는 페이지 번호로 설정하고, `original_text`에는 연결된 내용을 모두 통합하여 담으세요.
    4. 분석 후 Pydantic 스키마 정의에 맞추어 JSON 형식으로 반환하세요.
    """
    
    user_prompt = f"아래는 공고문 원본(Upstage 파싱 결과)입니다.\n\n{processed_text}"
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",  # GPT-4o 등 성능 우선 모델 사용
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=CertificateExtractionResult
    )
    
    # 파싱된 pydantic 모델 반환
    parsed_result = response.choices[0].message.parsed
    return [bonus.model_dump() for bonus in parsed_result.bonuses]

# ------------------------------------------------------------------
# 4. STEP 4: 비판적 예외 검증 (Assert 로직)
# ------------------------------------------------------------------
def validate_extraction(bonuses: List[dict], processed_text: str):
    """
    결과값 Validation:
    LLM이 환각(Hallucination)으로 존재하지 않는 페이지 번호를 조작하는지 막기 위한 검증 절차
    """
    # 원본 텍스트에 포함된 실제 페이지 번호들을 정수 Set으로 추출
    page_markers = set(int(p) for p in re.findall(r'\[Page:\s*(\d+)\]', processed_text))
    
    for bonus in bonuses:
        page_num = bonus['evidence_page']
        # Assert: LLM이 지목한 페이지 번호가 실제 파싱된 Page Marker에 존재하는지 검사
        assert page_num in page_markers, (
            f"검증 오류(Validation Error): 추출된 항목 '{bonus['certificate_name']}'의 "
            f"근거 페이지({page_num})가 원본 텍스트에 존재하지 않습니다."
        )
        
    print("[*] VLLM(Validation): 모든 추출 결과의 evidence_page 무결성 검증을 통과했습니다.")

# ------------------------------------------------------------------
# 5. 통합 파이프라인 (Main)
# ------------------------------------------------------------------
def process_job_posting(pdf_path: str, upstage_api_key: str, openai_api_key: str) -> List[dict]:
    # 1. 문서 메타데이터 로딩 (Upstage API 호출)
    print(f"[*] Upstage API를 통해 PDF 문서 분석 중... ({os.path.basename(pdf_path)})")
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {upstage_api_key}"}
    
    with open(pdf_path, "rb") as f:
        files = {"document": f}
        data = {"ocr": "force", "model": "document-parse"}
        response = requests.post(url, headers=headers, files=files, data=data)
        
    response.raise_for_status()
    upstage_data = response.json()
        
    # 2. 문서 전처리 (페이지 마커 및 표/텍스트 조합)
    print("[*] Upstage 결과물 전처리 및 페이지 마커 삽입 중...")
    processed_text = preprocess_upstage_result(upstage_data)
    
    # 3. LLM 분석 요청
    print("[*] GPT-4o를 이용한 자격증 가산점 정보 추출 중...")
    extracted_data = extract_certificate_bonuses(openai_api_key, processed_text)
    
    # 4. 환각/조작 검증
    validate_extraction(extracted_data, processed_text)
    
    return extracted_data

if __name__ == "__main__":
    # 기존 Document.py의 환경변수 및 파일(PDF) 연동
    UPSTAGE_API_KEY = ""
    OPENAI_API_KEY = ""
    TARGET_PDF = "/Users/myeongsung/Documents/upstage/한국전력공사.pdf"
    
    if os.path.exists(TARGET_PDF):
        try:
            final_result = process_job_posting(TARGET_PDF, UPSTAGE_API_KEY, OPENAI_API_KEY)
            print("\n[🎉 최종 추출 완료 🎉]")
            print(json.dumps(final_result, ensure_ascii=False, indent=2))
        except AssertionError as ae:
            print(f"[!] 검증 실패: {ae}")
        except Exception as e:
            print(f"[!] 오류 발생: {e}")
    else:
        print(f"[!] 테스트를 시작하려면 '{TARGET_PDF}' 파일이 필요합니다.")
