from fastapi import FastAPI
from dotenv import load_dotenv

# 애플리케이션 시작 전 .env 환경변수를 자동으로 불러옵니다.
load_dotenv()

from app.api.router import router as analyze_router

app = FastAPI(
    title="Resume Strategist API",
    description="JD 분석(PDF/URL) 및 경험 배치를 수행하는 LangGraph 기반 AI 에이전트 API (Stateless Worker)",
    version="1.2.0"
)

# API 라우터 등록
app.include_router(analyze_router, prefix="/api/v1", tags=["analyze"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
