import os
import json
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Literal, Optional, Union

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
    user_persona: str                  # 지원자 성향/가치관 (동적 S/W 프레이밍용)
    jd_context: Dict[str, Any]
    placements: List[Dict[str, Any]]
    remaining_indices: List[int]       # 하위 호환 유지 (신규 로직에서는 미사용)
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
    user_persona = state.get("user_persona", "") or "별도 성향 정보 없음"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "JD의 O, T와 각 경험의 S(강점), W(약점)를 대조하여 4대 전략 점수를 매기세요.\n"
         "SO: 강점으로 기회 선점 / ST: 강점으로 위협 돌파 / WO: 약점을 기회로 상쇄 / WT: 약점을 인정하고 보완.\n"
         "가장 점수가 높은 '극단적 전략'을 해당 경험의 대표 전략으로 정의하세요.\n\n"
         "[동적 S/W 프레이밍 규칙]\n"
         "아래 User Persona를 S/W 분류의 해석 필터로 사용하세요.\n"
         "단, Persona와 경험 팩트(content)가 충돌하면 반드시 팩트를 우선합니다.\n"
         "예: Persona가 '끈기'이지만 경험에 중도 포기가 명시된 경우 → W로 분류, WT 점수를 높이세요.\n\n"
         "한국어 구사: 모든 텍스트 출력(reasoning 등)을 반드시 한국어로 작성하세요."),
        ("user",
         "User Persona: {user_persona}\n"
         "JD Context:\nOpportunities: {opportunities}\nThreats: {threats}\n\n"
         "Experiences:\n{experiences}")
    ])

    chain = (prompt | llm.with_structured_output(ExperienceScoringList)).with_retry(stop_after_attempt=3)
    try:
        experiences_json = json.dumps(state["experiences"], ensure_ascii=False)
        result = chain.invoke({
            "user_persona": user_persona,
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
    """자소서 문항별 최적 경험을 전략적으로 배치합니다.

    주요 변경사항:
    - 1:N 매핑 허용: 동일 경험을 여러 문항에 재사용 가능
    - user_persona 기반 동적 S/W 해석
    - 팩트-페르소나 충돌 시 팩트 우선 (환각 방지)
    - 3단 reasoning: [JD 타겟팅] / [동적 프레이밍] / [전략 도출]
    - 적합 경험 없을 때 N/A 반환 (억지 매핑 금지)
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    experiences = state["experiences"]
    prompts = state["prompts"]
    user_persona = state.get("user_persona", "") or "별도 성향 정보 없음"
    jd_context = state.get("jd_context", {})
    placements = []

    # ── 구조화 출력 모델 ──────────────────────────────────────────
    class StrategicPlacement(BaseModel):
        experience_id: Optional[Union[str, int]] = Field(
            None,
            description="매핑된 경험 ID. 경험 목록 중 하나의 id 값 (문자열 혹은 숫자 가능). 적합한 경험이 없으면 null."
        )
        selected_strategy: Literal["SO", "ST", "WO", "WT", "N/A"] = Field(
            ...,
            description="SWOT 전략. 적합한 경험이 없으면 N/A."
        )
        jd_targeting: str = Field(
            description="[JD 타겟팅] JD의 어떤 구체적 요구사항을 Opportunity(O) 또는 Threat(T)으로 설정했는지 명시."
        )
        dynamic_framing: str = Field(
            description="[동적 프레이밍] user_persona 기준으로 해당 경험이 왜 강점(S) 또는 약점(W)으로 해석되는지 명시. "
                        "페르소나와 팩트가 충돌하면 팩트를 우선하고 그 이유를 설명하세요."
        )
        strategy_derivation: str = Field(
            description="[전략 도출] 선택한 SWOT 전략이 이 문항과 JD 환경에서 왜 최선의 선택인지 전략적 가치를 논증."
        )
        writing_guide: str = Field(
            description="실제 자소서 작성 가이드라인: 경험 내용과 전략을 연결하여 강조할 키워드·서술 흐름 정리."
        )

    # ── 시스템 프롬프트 ───────────────────────────────────────────
    SYSTEM_PROMPT = (
        "당신은 개발자 채용을 위한 최고 수준의 이력서/자소서 전략 분석 에이전트입니다.\n"
        "JD 분석, 지원자 경험, 지원자 성향을 종합하여 각 자소서 문항에 가장 강력한 전략적 매핑을 제공하세요.\n\n"
        "# 핵심 지침\n"
        "1. [유연한 매핑 - 1:N 허용]\n"
        "   - 하나의 경험 ID를 여러 문항에 중복 사용할 수 있습니다.\n"
        "   - 경험 데이터(content, tags)에 없는 사실(팀워크, 갈등 조율 등)을 절대 지어내지 마세요.\n"
        "   - 어떤 경험도 이 문항에 적합하지 않다면 experience_id=null, selected_strategy=N/A를 반환하세요.\n\n"
        "2. [동적 S/W 프레이밍 - User Persona 기반]\n"
        "   - 경험을 S/W로 분류할 때 user_persona를 해석 필터로 사용하세요.\n"
        "   - 단, Persona와 경험 팩트(content)가 충돌하면 반드시 팩트를 우선하세요.\n"
        "   - 예: Persona='끈기'지만 content에 '중도 포기'가 명시 → W 분류, WT 전략 선택.\n\n"
        "3. [SWOT 전략 정의]\n"
        "   - SO: 강점(S) + Persona → JD 핵심 요구사항(O) 완벽 충족\n"
        "   - ST: 강점(S) + Persona → JD 실무 허들/위협(T) 돌파\n"
        "   - WO: 약점(W) 인정 + JD 환경(O)에서의 성장 어필\n"
        "   - WT: 약점(W)과 위협(T) 직시, 현실적 보완책 제시\n\n"
        "4. [reasoning 3단 논리 구조]\n"
        "   - jd_targeting: JD의 어떤 구체적 요구사항을 O/T로 설정했는가\n"
        "   - dynamic_framing: Persona 기준으로 이 경험이 왜 S/W인가 (팩트 충돌 시 팩트 우선)\n"
        "   - strategy_derivation: 선택한 전략이 왜 이 문항과 JD 환경에서 최선인가\n\n"
        "# 한국어 출력 규칙 (가장 중요)\n"
        "모든 텍스트 출력을 반드시 한국어로 작성하세요. "
        "영어 사용을 엄격히 금지합니다."
    )

    # 경험 목록을 요약 (점수 포함해서 LLM이 참고하도록)
    experiences_summary = json.dumps(
        [
            {
                "id": e["id"],
                "title": e["title"],
                "content": e["content"],
                "tags": e.get("tags", []),
                "priority": e.get("priority", ""),
                "swot_scores": e.get("scores", {}),
                "primary_strategy": e.get("primary_strategy", ""),
            }
            for e in experiences
        ],
        ensure_ascii=False,
        indent=2,
    )

    placement_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user",
         "## 자소서 문항\n{prompt_text}\n\n"
         "## JD 분석 결과\nOpportunities: {opportunities}\nThreats: {threats}\n\n"
         "## User Persona\n{user_persona}\n\n"
         "## 사용 가능한 경험 목록 (1:N 재사용 가능)\n{experiences}")
    ])
    chain = (
        placement_prompt | llm.with_structured_output(StrategicPlacement)
    ).with_retry(stop_after_attempt=3)

    for prompt_text in prompts:
        print(f"[*] 문항 분석 중: '{prompt_text[:40]}...'")
        try:
            result: StrategicPlacement = chain.invoke({
                "prompt_text": prompt_text,
                "opportunities": jd_context.get("opportunities", ""),
                "threats": jd_context.get("threats", ""),
                "user_persona": user_persona,
                "experiences": experiences_summary,
            })

            # experience_id → experience_title 조회 (타입 불일치 방지를 위해 str()로 변환 후 비교)
            exp_title = "N/A"
            if result.experience_id is not None:
                matched = next(
                    (e for e in experiences if str(e["id"]) == str(result.experience_id)), None
                )
                exp_title = matched["title"] if matched else f"Unknown (id={result.experience_id})"
                print(f"    → 전략: {result.selected_strategy} | 경험: '{exp_title}'")
            else:
                print(f"    → 전략: N/A | 적합한 경험 없음")

            placements.append({
                "essay_question":           prompt_text,
                "matched_experience_id":    result.experience_id,
                "matched_experience_title": exp_title,
                "strategy":                 result.selected_strategy,
                "jd_targeting":             result.jd_targeting,
                "dynamic_framing":          result.dynamic_framing,
                "strategy_derivation":      result.strategy_derivation,
                "writing_guide":            result.writing_guide,
            })

        except Exception as e:
            state["errors"].append(f"[Strategic Placement Error] {str(e)}")
            placements.append({
                "essay_question":           prompt_text,
                "matched_experience_id":    None,
                "matched_experience_title": "오류",
                "strategy":                 "N/A",
                "jd_targeting":             f"배치 처리 중 오류 발생: {str(e)}",
                "dynamic_framing":          "",
                "strategy_derivation":      "",
                "writing_guide":            "N/A",
            })

    state["placements"] = placements
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
