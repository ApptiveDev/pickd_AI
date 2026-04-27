import streamlit as st
import requests
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io
import os

st.set_page_config(page_title="Pickd AI - JD Analyzer Demo", layout="wide")

st.title("🚀 Pickd AI - 채용 공고 분석 데모")
st.markdown("PDF, URL, 이미지 공고를 분석하고 출처를 시각적으로 확인하세요.")

# --- 사이드바 설정 ---
st.sidebar.header("설정")
analysis_mode = st.sidebar.selectbox("분석 모드 선택", ["PDF 분석", "URL 분석", "이미지 분석"])
api_base_url = st.sidebar.text_input("API 서버 주소", "http://127.0.0.1:8001/api/v1")

# --- 유틸리티 함수 ---
def draw_bbox(image, bbox, label=None):
    """이미지 위에 bbox를 그립니다."""
    draw = ImageDraw.Draw(image)
    # bbox format: [x1, y1, x2, y2]
    draw.rectangle(bbox, outline="red", width=3)
    if label:
        draw.text((bbox[0], bbox[1]-10), label, fill="red")
    return image

def render_pdf_page(pdf_bytes, page_num):
    """PDF 특정 페이지를 이미지로 렌더링합니다."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)  # 0-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# --- 메인 로직 ---
if analysis_mode == "PDF 분석":
    uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file and st.button("PDF 분석 시작"):
        with st.spinner("Upstage & LLM 분석 중..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{api_base_url}/analyze/pdf", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state["analysis_result"] = result
                st.session_state["pdf_bytes"] = uploaded_file.getvalue()
                st.success("분석 완료!")
            else:
                st.error(f"오류 발생: {response.text}")

elif analysis_mode == "URL 분석":
    url = st.text_input("채용 공고 URL을 입력하세요", placeholder="https://www.saramin.co.kr/...")
    if url and st.button("URL 분석 시작"):
        with st.spinner("Firecrawl & LLM 분석 중..."):
            response = requests.post(f"{api_base_url}/analyze/url", json={"url": url})
            if response.status_code == 200:
                st.session_state["analysis_result"] = response.json()
                st.success("분석 완료!")
            else:
                st.error(f"오류 발생: {response.text}")

elif analysis_mode == "이미지 분석":
    uploaded_files = st.file_uploader("공고 이미지들을 업로드하세요 (다중 선택 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files and st.button("이미지 통합 분석 시작"):
        with st.spinner("Gemini 2.5 Flash-Lite 분석 중..."):
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            response = requests.post(f"{api_base_url}/analyze/image", files=files)
            if response.status_code == 200:
                st.session_state["analysis_result"] = response.json()
                st.session_state["image_bytes_list"] = [f.getvalue() for f in uploaded_files]
                st.success("분석 완료!")
            else:
                st.error(f"오류 발생: {response.text}")

# --- 결과 표시 및 시각화 ---
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 추출된 채용 정보")
        
        # 필드명 매핑 (한국어: 영어)
        field_mapping = {
            "company_name": "기업명",
            "job_title": "공고명",
            "qualifications": "지원 자격",
            "industry": "산업/분야",
            "application_period": "지원 기간",
            "essay_question_count": "자소서 문항 수",
            "work_location": "근무지",
            "preferred_qualifications": "우대 사항",
            "extra_points": "가산점 항목",
            "evaluation_criteria": "전형 절차",
            "salary": "급여/연봉"
        }
        
        for eng_name, kor_name in field_mapping.items():
            st.write(f"**{kor_name} ({eng_name})**: {result.get(eng_name, 'N/A')}")
            
        st.divider()
        st.subheader("🔗 출처(Citations) 목록")
        for i, citation in enumerate(result.get("citations", [])):
            # 출처 필드명도 매핑 적용
            display_field = field_mapping.get(citation['field'], citation['field'])
            with st.expander(f"[{i+1}] {display_field} ({citation['field']})"):

                st.write(f"**내용**: {citation['content']}")
                if citation.get("source_url"):
                    st.link_button("원본 위치 확인", citation["source_url"])
                
                # 시각화 버튼
                if st.button(f"위치 보기", key=f"btn_{i}"):
                    st.session_state["selected_citation"] = citation

    with col2:
        st.subheader("📍 원본 위치 시각화")
        if "selected_citation" in st.session_state:
            cit = st.session_state["selected_citation"]
            
            # PDF 시각화
            if analysis_mode == "PDF 분석" and "pdf_bytes" in st.session_state:
                page_img = render_pdf_page(st.session_state["pdf_bytes"], cit["page"])
                if cit.get("bbox"):
                    page_img = draw_bbox(page_img, cit["bbox"], cit["field"])
                st.image(page_img, caption=f"Page {cit['page']} 분석 결과", use_container_width=True)
            
            # 이미지 시각화
            elif analysis_mode == "이미지 분석" and "image_bytes_list" in st.session_state:
                img_idx = cit["page"] - 1
                if 0 <= img_idx < len(st.session_state["image_bytes_list"]):
                    img = Image.open(io.BytesIO(st.session_state["image_bytes_list"][img_idx]))
                    if cit.get("bbox"):
                        # Gemini bbox는 보통 정규화되어 있을 수 있으므로 처리 필요 (현재는 Upstage 스타일 좌표 가정)
                        img = draw_bbox(img, cit["bbox"], cit["field"])
                    st.image(img, caption=f"이미지 {cit['page']} 분석 결과", use_container_width=True)
            
            # URL은 링크로 대체
            else:
                st.info("URL 분석은 '원본 위치 확인' 버튼을 클릭해 브라우저에서 직접 확인하세요.")
        else:
            st.info("왼쪽 목록에서 '위치 보기' 버튼을 클릭하면 이곳에 해당 영역이 표시됩니다.")
