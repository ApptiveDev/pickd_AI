from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any

# ── 온보딩 데이터 DTO ──────────────────────────────────────
class UserBase(BaseModel):
    nickname: str = Field(..., description="닉네임")
    email: EmailStr = Field(..., description="이메일")
    region_code: str = Field(..., description="지역 (코드화)")
    
    school: str = Field(..., description="학교")
    major: str = Field(..., description="전공")
    education_status: str = Field(..., description="학력 상태")
    
    interested_job_groups: List[str] = Field(..., description="관심 직군")
    interested_industries: List[str] = Field(..., description="관심 산업")
    
    certificates: List[str] = Field(..., description="자격증 (코드 기반)")
    languages: List[str] = Field(..., description="어학 (코드 기반)")

class UserCreate(UserBase):
    id: str = Field(..., description="Google OAuth ID (PK)")

class UserResponse(UserBase):
    id: str

    class Config:
        from_attributes = True

class Citation(BaseModel):
    field: str = Field(..., description="근거가 되는 필드명")
    page: int = Field(..., description="근거가 발견된 PDF 페이지 번호")
    content: str = Field(..., description="근거가 된 원문 텍스트 일부")

# ── 공고 분석 데이터 DTO (11개 필드 + 출처) ──────────────────
class JobPostingBase(BaseModel):
    # 필수 정보 (6종)
    company_name: str = Field(..., description="기업명")
    job_title: str = Field(..., description="직무명")
    qualifications: str = Field(..., description="자격요건")
    industry: str = Field(..., description="산업")
    application_period: str = Field(..., description="지원 기간")
    essay_question_count: int = Field(..., description="자소서 문항 수")
    
    # 개인화 정보 (5종)
    work_location: Optional[str] = Field(None, description="근무지")
    preferred_qualifications: Optional[str] = Field(None, description="우대사항")
    extra_points: Optional[str] = Field(None, description="가산점")
    evaluation_criteria: Optional[str] = Field(None, description="전형 배점 (텍스트 또는 JSON 형태의 문자열)")
    salary: Optional[str] = Field(None, description="연봉")

    # 출처 정보 (NotebookLM 스타일)
    citations: List[Citation] = Field(default_factory=list, description="데이터 추출 근거 및 출처 정보")


class JobPostingCreate(JobPostingBase):
    pass

class JobPostingResponse(JobPostingBase):
    id: int

    class Config:
        from_attributes = True

class UrlAnalysisRequest(BaseModel):
    url: str = Field(..., description="분석할 채용 공고 URL")

