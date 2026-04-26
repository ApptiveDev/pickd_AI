import json
import requests
import io
import time

MOCK_JD_MARKDOWN = """
# AI 플랫폼 백엔드 엔지니어 경력 채용

## 직무 비전 및 기회 (Opportunities)
- 대규모 언어 모델(LLM)과 RAG를 결합한 사내 및 기업용 AI 에이전트 서비스 구축
- Vector DB (Qdrant, Pinecone 등) 설계 및 Hybrid Search 파이프라인 최적화
- Spring Boot, FastAPI, Node.js 등을 활용한 안정적인 멀티 마이크로서비스 아키텍처(MSA) 운영 경험 제공

## 직무 수행의 난관 (Threats)
- 비정형 도메인 데이터 검색 시 나타나는 오분류 문제와 RAG의 Hallucination(환각) 리스크 최소화 역량 필수
- 대용량 데이터 및 영상 분석 시 서버의 인프라 비용과 응답 지연(Latency)이 기하급수적으로 늘어나는 문제
"""

MOCK_EXPERIENCES = [
    {
        "id": 1,
        "title": "Fin-agent: 금융 특화 AI 에이전트 및 Hybrid Retrieval 구축",
        "content": "미래에셋 공모전에서 금융 질의응답 신뢰성을 높이기 위해 Sparse(의미) 65% + Dense(유사도) 35% 가중치를 적용한 Hybrid Retrieval 시스템을 구축함. 계열사 오분류 문제를 해결하기 위해 Reranker와 LLM 검증 단계를 추가하여 검색 정확도를 극대화함.",
        "tags": ["LangGraph", "RAG", "Hybrid Retrieval", "Python"],
        "priority": "상"
    },
    {
        "id": 2,
        "title": "Memoralaxy: GNN 기반 지식 추천 및 DAG 구조화",
        "content": "비정형 문서에서 지식을 추출하고 GraphSAGE 모델을 통해 추천 시스템을 구축함. 특히 DFS 알고리즘을 활용해 관계 데이터의 사이클을 제거하고 강제로 DAG(방향성 비순환 그래프) 구조를 확보하여 학습 경로의 논리적 선후 관계를 보장함.",
        "tags": ["GNN", "GraphSAGE", "Algorithm", "Python"],
        "priority": "상"
    },
    {
        "id": 3,
        "title": "YAR-YAR_BE: 비용 효율적 AI 릴스 분석 시스템",
        "content": "숏폼 영상 분석 시 발생하는 고비용 문제를 해결하기 위해 FFmpeg로 핵심 프레임(3~5장)만 추출하여 분석하는 로직을 설계함. Spring-FastAPI-MySQL 멀티 컨테이너 환경을 Docker Compose와 GitHub Actions로 CI/CD 구축함.",
        "tags": ["Spring Boot", "FastAPI", "Docker", "FFmpeg", "비용최적화"],
        "priority": "상"
    },
    {
        "id": 4,
        "title": "stopping: 판례 기반 스토킹 판별 RAG 시스템",
        "content": "Qdrant Vector DB에 실제 판례를 저장하고 사용자의 상황과 유사한 데이터를 증강(RAG)하여 GPT-3.5가 법적 근거를 제시하도록 설계함. Spring 서버와 Python 임베딩 로직을 연동하여 실시간 유사도 추출 기능을 구현함.",
        "tags": ["Vector DB", "Qdrant", "RAG", "Spring Boot"],
        "priority": "중"
    },
    {
        "id": 5,
        "title": "TOMO: 소셜 모임 서비스 인프라 및 CI/CD 운영",
        "content": "Nginx 리버스 프록시 및 HTTPS 환경을 구축하고 AWS/Oracle Cloud를 병행하여 VM 인스턴스를 운영함. GitHub Actions를 통해 자동 배포 파이프라인을 구축하고 실제 유저 피드백 기반 릴리즈를 경험함.",
        "tags": ["Infrastructure", "CI/CD", "Nginx", "AWS"],
        "priority": "중"
    },
    {
        "id": 6,
        "title": "Nomad: WebSocket 기반 실시간 채팅 및 AOP 예외처리",
        "content": "WebSocket을 이용한 실시간 채팅과 세션 로그인을 구현함. 특히 AOP를 도입하여 서비스 단의 예외를 전역적으로 핸들링하고 클라이언트에게 표준화된 에러 응답을 전달하는 구조를 설계함.",
        "tags": ["WebSocket", "AOP", "Spring Boot", "Java"],
        "priority": "하"
    }
]

MOCK_PROMPTS = [
    "문항 1: 지원 직무와 관련하여 본인이 직면했던 가장 어려운 기술적 도전은 무엇이며, 이를 어떻게 해결했는지 기술하십시오.",
    "문항 2: 기존의 방식에서 벗어나 효율성을 높이거나 비용을 절감했던 창의적인 문제 해결 경험이 있다면 기술하십시오.",
    "문항 3: 본인이 가진 기술적 약점은 무엇이며, 이를 보완하기 위해 어떤 노력을 기울이고 있는지 실제 프로젝트 사례를 들어 기술하십시오."
]

def run_test():
    url = "http://127.0.0.1:8000/analyze-and-place"
    
    print("\\n=== [실제 명성님의 데이터를 활용한 API 테스트 실행] ===")
    
    # 안정적인 테스트를 위해 사용자의 경험(RAG, AI 백엔드)과 정확히 매칭되는 가상의 JD PDF를 전송합니다.
    dummy_pdf = io.BytesIO(MOCK_JD_MARKDOWN.encode("utf-8"))
    files = {
        "jd_pdf": ("mock_ai_jd.pdf", dummy_pdf, "application/pdf")
    }
    
    data = {
        "experiences_json": json.dumps(MOCK_EXPERIENCES, ensure_ascii=False),
        "essay_prompts_json": json.dumps(MOCK_PROMPTS, ensure_ascii=False)
    }
    
    print("🚀 FastAPI 서버로 분석을 요청합니다...")
    start_time = time.time()
    
    try:
        response = requests.post(url, files=files, data=data)
    except Exception as e:
        print("네트워크 또는 FastAPI 서버 연결 오류:", e)
        return
        
    end_time = time.time()
    
    if response.status_code == 200:
        result_json = response.json()
        print(f"✅ 분석 완료! (소요 시간: {end_time - start_time:.2f}초)\\n")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))
        
        # 보기 좋게 요약 출력
        print("\\n=======================================================")
        for p in result_json.get("placements", []):
            print(f"- [매칭 문항] : {p['question'][:30]}...")
            print(f"  [채택 경험] : {p['experience_title']}")
            print(f"  [전략 방향] : {p['selected_strategy']}")
            print(f"  [가이드라인]: {p['writing_guide']}")
            print("-" * 55)
            
    else:
        print(f"❌ 분석 실패 (Status Code: {response.status_code})")
        print(response.text)

if __name__ == "__main__":
    run_test()
