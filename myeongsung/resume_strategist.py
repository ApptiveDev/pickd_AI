import os
import json
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Literal, Optional

# .env 환경변수를 자동으로 불러옵니다.
load_dotenv()

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ==========================================
# [Fix 2] 간이 캐싱 메모리 (동일 URL 크롤링 회피)
# ==========================================
JD_URL_CACHE: Dict[str, str] = {}


# ==========================================
# 0. Custom Exceptions
# ==========================================
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
class RateLimitException(Exception):
    pass

# ==========================================
# 1. State 정의
# ==========================================
class AgentState(TypedDict):
    jd_markdown: str
    jd_url: Optional[str]
    experiences: List[Dict[str, Any]]
    prompts: List[str]
    jd_context: Dict[str, Any]
    placements: List[Dict[str, Any]]
    remaining_indices: List[int]
    errors: List[str]

# ==========================================
# 2. 구조화된 출력을 위한 Pydantic 모델
# ==========================================
class JDAnalysis(BaseModel):
    opportunities: str = Field(description="Opportunities (O)")
    threats: str = Field(description="Threats (T)")

class StrategyScore(BaseModel):
    SO: int = Field(ge=0, le=100)
    ST: int = Field(ge=0, le=100)
    WO: int = Field(ge=0, le=100)
    WT: int = Field(ge=0, le=100)

class ScoredExperience(BaseModel):
    id: int = Field(...)
    scores: StrategyScore = Field(...)
    primary_strategy: Literal["SO", "ST", "WO", "WT"] = Field(...)
    reasoning: str = Field(...)

class ExperienceScoringList(BaseModel):
    scored_experiences: List[ScoredExperience] = Field(...)


# ==========================================
# 3. LangGraph 노드 함수 구현
# ==========================================

def jd_ingestion_router(state: AgentState) -> Literal["Upstage_Parse_Node", "Cache_Hit_Node", "Web_Scraping_Node"]:
    """Node: 파싱 노드 라우팅 로직 (Cache 우선적 확인)"""
    if state.get("jd_markdown") and state.get("jd_markdown").strip():
        return "Upstage_Parse_Node"
    elif state.get("jd_url") and state.get("jd_url").strip():
        url = state["jd_url"]
        # [Fix 2] 이전에 들어온 URL인지 체크하여 Conditional Edge 분기
        if url in JD_URL_CACHE:
            return "Cache_Hit_Node"
        return "Web_Scraping_Node"
    else:
        return "Upstage_Parse_Node"

def upstage_parse_node(state: AgentState) -> AgentState:
    return state

def cache_hit_node(state: AgentState) -> AgentState:
    """Node: 이전에 저장된 JD 메모리를 재사용하여 통신 횟수를 절감함"""
    url = state.get("jd_url")
    if url in JD_URL_CACHE:
        print(f"[*] {url} 발견! 캐시에서 결과(마크다운)를 불러옵니다 (크롤링 건너뜀).")
        state["jd_markdown"] = JD_URL_CACHE[url]
    return state

# [Fix 3] 429 에러 밸생 시 지수 백오프 로직 (5초->10초 늘려가며 최대 3회 시도)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=20),
    retry=retry_if_exception_type(RateLimitException)
)
def fetch_html_with_retry(url: str) -> str:
    import requests
    from fake_useragent import UserAgent
    
    # [Fix 1] fake-useragent와 랜덤 Sleep 처리
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    delay = random.uniform(1.5, 3.5)
    print(f"[*] 봇 탐지 회피용 대기 진행... ({delay:.2f}초 Sleep)")
    time.sleep(delay)
    
    response = requests.get(url, headers=headers, timeout=15)
    
    if response.status_code == 429:
        print(f"⚠️ [HTTP 429 에러] 대상 서버에서 너무 많은 요청을 감지함! 지수 백오프(Exponential Backoff) 재시도 작동 준비...")
        raise RateLimitException(f"429 Too Many Requests: {url}")
        
    response.raise_for_status()
    return response.text

def web_scraping_node(state: AgentState) -> AgentState:
    url = state.get("jd_url")
    if not url:
        return state
        
    try:
        from bs4 import BeautifulSoup
        print(f"[*] {url} 라이브 웹 스크래핑 시도...")
        
        # 재시도/헤더/지연로직이 포함된 안전한 페치(fetch) 함수 호출
        html_text = fetch_html_with_retry(url)
        
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.extract()
            
        raw_text = soup.get_text(separator="\n", strip=True)
        if len(raw_text) > 30000:
            raw_text = raw_text[:30000]
            
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 웹 페이지에서 채용 정보만 추출하는 전문 데이터 엔지니어입니다. 제공된 텍스트에서 회사 소개, 주요 업무, 자격 요건, 우대 사항, 복지 혜택에 해당하는 내용만 마크다운 형식으로 요약하세요. 채용과 관련 없는 광고, 사이트 메뉴, 법적 고지 등은 반드시 제외하십시오."),
            ("user", "원문 텍스트:\n{raw_text}")
        ])
        
        clean_md = (prompt | llm).with_retry(stop_after_attempt=3).invoke({"raw_text": raw_text}).content
        state["jd_markdown"] = clean_md
        
        # [Fix 2] 분석 성공 시 캐시(딕셔너리)에 덮어쓰기 저장
        JD_URL_CACHE[url] = clean_md
        
    except Exception as e:
        state["errors"].append(f"[Web Scraping Error] {str(e)}")
        state["jd_markdown"] = f"# 스크래핑 및 LLM 정제 실패\n\nURL: {url}\n오류 내용: {str(e)}"
        
    return state


def jd_structural_analyzer(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채용 공고의 행간을 읽는 전략가입니다. 제공된 마크다운 텍스트에서 다음을 도출하세요.\n"
                   "Opportunities (O): 직무의 비전, 핵심 우대사항, 기업의 성장 동력.\n"
                   "Threats (T): 직무 수행의 난관, 기술적 복잡성, 업계의 페인 포인트.\n"
                   "결과는 JSON 구조로 저장하세요."),
        ("user", "JD Markdown:\n{jd_markdown}")
    ])
    
    chain = (prompt | llm.with_structured_output(JDAnalysis)).with_retry(stop_after_attempt=3)
    try:
        result = chain.invoke({"jd_markdown": state.get("jd_markdown", "")})
        state["jd_context"] = result.model_dump()
    except Exception as e:
        state["errors"].append(f"[JD Analysis API Error] {str(e)}")
        
    return state


def swot_strategy_scorer(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "JD의 O, T와 각 경험의 S(강점), W(약점)를 대조하여 4대 전략 점수를 매기세요.\n"
                   "SO: 강점으로 기회 선점 / ST: 강점으로 위협 돌파 / WO: 약점을 기회로 상쇄 / WT: 약점을 인정하고 보완.\n"
                   "가장 점수가 높은 '극단적 전략'을 해당 경험의 대표 전략으로 정의하세요."),
        ("user", "JD Context:\nOpportunities: {opportunities}\nThreats: {threats}\n\nExperiences:\n{experiences}")
    ])
    
    chain = (prompt | llm.with_structured_output(ExperienceScoringList)).with_retry(stop_after_attempt=3)
    try:
        experiences_json = json.dumps(state["experiences"], ensure_ascii=False)
        result = chain.invoke({
            "opportunities": state.get("jd_context", {}).get("opportunities", ""),
            "threats": state.get("jd_context", {}).get("threats", ""),
            "experiences": experiences_json
        })
        
        score_map = {item.id: item for item in result.scored_experiences}
        for exp in state["experiences"]:
            score_data = score_map.get(exp["id"])
            if score_data:
                exp["scores"] = score_data.scores.model_dump()
                exp["primary_strategy"] = score_data.primary_strategy
                exp["strategy_reasoning"] = score_data.reasoning
                
    except Exception as e:
        state["errors"].append(f"[SWOT Scoring API Error] {str(e)}")
        
    return state


# ==========================================
# [개선] 문항 의도 기반 전략 감지 헬퍼 함수
# - SO 기본값 편향 제거
# - 6가지 문항 유형 커버
# ==========================================
_STRATEGY_CHOICES = ["SO", "ST", "WO", "WT"]

def _detect_intent_strategy(prompt_text: str, llm) -> tuple[str | None, bool]:
    """자소서 문항 의도를 분석하여 최적 SWOT 전략을 반환합니다.
    
    Returns:
        (target_strategy, fallback_used)
        - target_strategy: 'SO' | 'ST' | 'WO' | 'WT' | None (None이면 폴백 필요)
        - fallback_used: LLM 감지 실패 여부
    """
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 자소서 문항 의도 분석 전문가입니다.\n"
         "아래 [문항 유형별 전략 매핑]을 기준으로 주어진 자소서 문항을 분석하여 "
         "가장 적합한 SWOT 전략 하나를 선택하세요.\n\n"
         "[문항 유형별 전략 매핑]\n"
         "• 성과/도전/목표달성/리더십 문항 → SO\n"
         "  예: '목표를 세우고 달성한 경험', '도전적인 사례', '리더로서의 경험', '성과를 낸 경험'\n\n"
         "• 위기대응/경쟁상황/압박/기술적 난관 문항 → ST\n"
         "  예: '어려운 상황에서 문제를 해결한 경험', '갈등·충돌을 해결한 경험', '실패 위기를 극복한 사례'\n\n"
         "• 약점보완/성장/개선/협업·팀워크 문항 → WO\n"
         "  예: '부족한 점을 보완한 경험', '피드백을 받아 성장한 경험', '팀원과 협력하여 성과를 낸 경험'\n\n"
         "• 실패/한계 인정/반성/포기 경험 문항 → WT\n"
         "  예: '가장 힘들었던 경험', '실패한 경험과 교훈', '포기했거나 한계를 직면한 경험'\n\n"
         "• 가치관/신념/직업의식/인생관 문항 → SO\n"
         "  예: '직업 가치관', '인생 좌우명', '가장 중요하게 여기는 것'\n\n"
         "• 지원동기/직무이해/입사 후 포부 문항 → ST\n"
         "  예: '지원 동기', '이 직무를 선택한 이유', '입사 후 목표 및 성장 계획'\n\n"
         "중요: 반드시 SO, ST, WO, WT 중 정확히 하나만 출력하세요. "
         "다른 텍스트, 설명, 구두점은 절대 포함하지 마세요."),
        ("user", "자소서 문항: {prompt_text}")
    ])

    try:
        raw = (intent_prompt | llm).with_retry(stop_after_attempt=3).invoke(
            {"prompt_text": prompt_text}
        ).content.strip().upper()
        # 불필요한 구두점·공백 제거 후 유효성 검사
        cleaned = raw.replace(".", "").replace(",", "").replace("'", "").replace('"', "").strip()
        if cleaned in _STRATEGY_CHOICES:
            return cleaned, False
        # LLM이 유효하지 않은 문자열 반환 → 폴백 필요
        return None, True
    except Exception:
        return None, True


def _score_based_fallback(
    experiences: list,
    remaining_indices: list,
    priority_weight: dict,
) -> str:
    """LLM 의도 감지 실패 시 경험 점수 합계 기반으로 최적 전략을 선택합니다.
    
    각 후보 경험의 우선순위를 반영한 전략별 점수 총합을 계산,
    가장 높은 전략을 반환합니다. 점수가 모두 0이면 무작위 선택.
    """
    strategy_totals: dict[str, float] = {s: 0.0 for s in _STRATEGY_CHOICES}

    for idx in remaining_indices:
        exp = experiences[idx]
        scores = exp.get("scores", {})
        # 우선순위 가중치: 상=3, 중=2, 하=1 (0이면 최소 1 보장)
        p_val = priority_weight.get(exp.get("priority", "하"), 0) + 1
        for strategy in _STRATEGY_CHOICES:
            strategy_totals[strategy] += scores.get(strategy, 0) * p_val

    if all(v == 0.0 for v in strategy_totals.values()):
        return random.choice(_STRATEGY_CHOICES)

    return max(strategy_totals, key=lambda s: strategy_totals[s])


def sequential_strategic_placer(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    experiences = state["experiences"]
    prompts = state["prompts"]
    
    remaining_indices = state.get("remaining_indices", [])
    if not remaining_indices:
        remaining_indices = list(range(len(experiences)))
        
    placements = []
    priority_weight = {"상": 2, "중": 1, "하": 0}
    
    # [Fix] 가이드라인과 매칭 논리를 엄격히 통제할 구조화된 출력 모델 설계
    class GuideOutput(BaseModel):
        reasoning: str = Field(description="전략 선정 이유. 반드시 '분석 결과 [선택한 전략명] 전략이 가장 적합합니다'라는 문장으로 시작할 것")
        writing_guide: str = Field(description="실제 자소서 작성 가이드라인 및 핵심 키워드 정리")
    
    for prompt_text in prompts:
        if not remaining_indices:
            placements.append({
                "question": prompt_text,
                "experience_title": "경험 부족",
                "selected_strategy": "N/A",
                "reasoning": "할당 가능한 여유 경험이 부족합니다.",
                "writing_guide": "N/A"
            })
            continue

        # [개선] 6가지 문항 유형 기반 의도 감지 (SO 기본값 편향 제거)
        target_strategy, fallback_used = _detect_intent_strategy(prompt_text, llm)
        if fallback_used or target_strategy is None:
            # LLM 감지 실패 시 경험 점수 합계 기반 폴백 (SO 하드코딩 제거)
            target_strategy = _score_based_fallback(experiences, remaining_indices, priority_weight)
            print(f"[*] 의도 감지 폴백 사용 → 점수 기반 전략: {target_strategy} (문항: '{prompt_text[:30]}...')") 
        else:
            print(f"[*] 의도 감지 성공 → 전략: {target_strategy} (문항: '{prompt_text[:30]}...')")
            
        best_exp_idx = -1
        max_score = -1
        best_priority_val = -1
        
        for idx in remaining_indices:
            exp = experiences[idx]
            score = exp.get("scores", {}).get(target_strategy, 0)
            p_val = priority_weight.get(exp.get("priority", "하"), 0)
            
            if max_score != -1 and abs(max_score - score) < 5:
                if p_val > best_priority_val:
                    max_score = score
                    best_exp_idx = idx
                    best_priority_val = p_val
            elif score > max_score:
                max_score = score
                best_exp_idx = idx
                best_priority_val = p_val

        if best_exp_idx != -1:
            best_exp = experiences[best_exp_idx]
            remaining_indices.remove(best_exp_idx)
            
            # [Fix] Self-Correction을 유도하는 프롬프트 적용
            guide_prompt = ChatPromptTemplate.from_messages([
                ("system", "자소서 작성 가이드라인과 매칭 논리를 작성해주세요.\n"
                           "1. reasoning 필드는 서두에 반드시 '분석 결과 [{target_strategy}] 전략이 가장 적합합니다'라는 문장을 강제로 포함하여 AI 스스로 자신의 논리적 일관성을 확인(Self-Correction)하세요. 결론 내린 전략 명칭은 반드시 {target_strategy} 와 100% 일치해야 합니다.\n"
                           "2. writing_guide 필드는 경험의 내용과 매칭 논리를 연결하여, 서술 시 강조해야 할 핵심 키워드 및 흐름을 분석하세요."),
                ("user", "문항: {prompt_text}\n전략: {target_strategy}\n경험 명: {exp_title}\n경험 내용: {exp_content}\n경험의 원래 평가이유: {reasoning}")
            ])
            try:
                guide_chain = (guide_prompt | llm.with_structured_output(GuideOutput)).with_retry(stop_after_attempt=3)
                guide_result = guide_chain.invoke({
                    "prompt_text": prompt_text,
                    "target_strategy": target_strategy,
                    "exp_title": best_exp["title"],
                    "exp_content": best_exp["content"],
                    "reasoning": best_exp.get("strategy_reasoning", "")
                })
                final_reasoning = guide_result.reasoning
                final_guide = guide_result.writing_guide
            except Exception:
                final_reasoning = f"분석 결과 [{target_strategy}] 전략이 가장 적합합니다. (세부 매칭 논리 생성 실패)"
                final_guide = "가이드 생성 실패"

            placements.append({
                "question": prompt_text,
                "experience_id": best_exp["id"],
                "experience_title": best_exp["title"],
                "selected_strategy": target_strategy,
                "reasoning": final_reasoning,
                "writing_guide": final_guide
            })
            
    state["placements"] = placements
    state["remaining_indices"] = remaining_indices
    return state


# ==========================================
# 4. LangGraph 파이프라인 컴파일
# ==========================================
def create_workflow() -> Any:
    workflow = StateGraph(AgentState)
    
    # 노드 부착
    workflow.add_node("Upstage_Parse_Node", upstage_parse_node)
    workflow.add_node("Cache_Hit_Node", cache_hit_node)
    workflow.add_node("Web_Scraping_Node", web_scraping_node)
    
    workflow.add_node("JD_Structural_Analyzer", jd_structural_analyzer)
    workflow.add_node("SWOT_Strategy_Scorer", swot_strategy_scorer)
    workflow.add_node("Sequential_Strategic_Placer", sequential_strategic_placer)
    
    # 라우팅
    workflow.add_conditional_edges(START, jd_ingestion_router)
    
    # 순차 플로우 연결
    workflow.add_edge("Upstage_Parse_Node", "JD_Structural_Analyzer")
    workflow.add_edge("Cache_Hit_Node", "JD_Structural_Analyzer")
    workflow.add_edge("Web_Scraping_Node", "JD_Structural_Analyzer")
    
    workflow.add_edge("JD_Structural_Analyzer", "SWOT_Strategy_Scorer")
    workflow.add_edge("SWOT_Strategy_Scorer", "Sequential_Strategic_Placer")
    workflow.add_edge("Sequential_Strategic_Placer", END)
    
    return workflow.compile()
