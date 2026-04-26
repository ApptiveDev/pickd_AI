from sqlalchemy import Column, String, Integer, JSON
from database import Base

class User(Base):
    __tablename__ = "users"

    # [Task] Google OAuth ID 기반 PK 설계 반영
    id = Column(String, primary_key=True, index=True, comment="Google OAuth ID")
    
    # [Task] 온보딩 데이터 DB 스키마 설계
    # 기본 정보
    nickname = Column(String, index=True, comment="닉네임")
    email = Column(String, unique=True, index=True, comment="이메일")
    region_code = Column(String, comment="지역 (코드화 저장)")
    
    # 학력
    school = Column(String, comment="학교")
    major = Column(String, comment="전공")
    education_status = Column(String, comment="학력 상태 (재학, 졸업 등)")
    
    # 관심사 (MVP 단계에서는 다중 선택을 JSON으로 저장)
    interested_job_groups = Column(JSON, comment="관심 직군 (다중 선택)")
    interested_industries = Column(JSON, comment="관심 산업 (다중 선택)")
    
    # 준비 상태
    certificates = Column(JSON, comment="자격증 (코드 기반)")
    languages = Column(JSON, comment="어학 (코드 기반)")

class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # 필수 정보 (6종)
    company_name = Column(String, index=True, comment="기업명")
    job_title = Column(String, comment="직무명")
    qualifications = Column(String, comment="자격요건")
    industry = Column(String, comment="산업")
    application_period = Column(String, comment="지원 기간")
    essay_question_count = Column(Integer, comment="자소서 문항 수")
    
    # 개인화 정보 (5종)
    work_location = Column(String, nullable=True, comment="근무지")
    preferred_qualifications = Column(String, nullable=True, comment="우대사항")
    extra_points = Column(String, nullable=True, comment="가산점")
    evaluation_criteria = Column(JSON, nullable=True, comment="전형 배점 (구조화)")
    salary = Column(String, nullable=True, comment="연봉")
