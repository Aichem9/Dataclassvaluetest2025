import os
import io
import re
import json
import base64
import pandas as pd
import streamlit as st

# ---- Optional LLM ----
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    pass

# ---- File parsers ----
from typing import List, Tuple

def read_pdf(file) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""

def read_docx(file) -> str:
    try:
        import docx
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def read_txt(file) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            return file.read().decode("cp949", errors="ignore")
        except Exception:
            return ""

def extract_text(uploaded_files) -> str:
    texts = []
    for f in uploaded_files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            texts.append(read_pdf(f))
        elif name.endswith(".docx"):
            texts.append(read_docx(f))
        elif name.endswith(".txt"):
            texts.append(read_txt(f))
        else:
            texts.append("")
    return "\n\n".join(texts).strip()

# ---- Framework (요약 버전) ----
FRAMEWORK = {
    "데이터 인식 및 이해": [
        "데이터 개념/정의 이해", "정형/비정형·질적/양적 구분", "데이터 생성·수집 맥락 이해",
        "일상 속 데이터 인식", "데이터-정보-지식 관계"
    ],
    "데이터 수집 및 관리": [
        "적합한 수집 방법(관찰·측정·설문 등)", "데이터 구조화·정제·품질(정확성·완전성·신뢰성)",
        "저장·관리·공유 절차", "2차/공공데이터 활용"
    ],
    "데이터 분석 및 해석": [
        "기술통계/패턴/이상치", "관계 분석(상관·인과)", "통계적 추론·가설검증",
        "도구 활용(스프레드시트·분석툴)", "근거 기반 해석"
    ],
    "데이터 활용 및 표현": [
        "목적 맞는 시각화 선택", "스토리텔링·논증", "청중 맞춤 표현",
        "데이터 기반 의사결정/문제 해결"
    ],
    "데이터 윤리 및 비판적 사고": [
        "개인정보·프라이버시", "출처·신뢰성 검증", "편향·왜곡·허위정보 판별",
        "알고리즘 편향/사회적 영향·책임"
    ],
}

RUBRIC_SCALE = ["보완 필요 (1점)", "보통 (2점)", "우수 (3점)"]

# ---- Keyword heuristics (API 없을 때) ----
HEURISTICS = {
    "데이터 인식 및 이해": [
        r"데이터의 개념|정의|메타데이터|정형|비정형|질적|양적|데이터 생태계|데이터 과학|데이터 경제"
    ],
    "데이터 수집 및 관리": [
        r"관찰|측정|설문|표본|정제|품질|신뢰성|정확성|저장|관리|공유|보안|공공 데이터|데이터베이스|크롤링"
    ],
    "데이터 분석 및 해석": [
        r"평균|중앙값|최빈값|분산|표준편차|상관|회귀|가설|추론|모델|예측|분석 도구|스프레드시트"
    ],
    "데이터 활용 및 표현": [
        r"시각화|그래프|차트|인포그래픽|대시보드|스토리텔링|발표|보고서|의사결정|정책"
    ],
    "데이터 윤리 및 비판적 사고": [
        r"개인정보|프라이버시|출처|신뢰성|편향|왜곡|허위 정보|공정성|거버넌스|윤리|책임"
    ],
}

def heuristic_score(text: str, pattern: str) -> int:
    if not text:
        return 1
    hits = len(re.findall(pattern, text, flags=re.IGNORECASE))
    if hits >= 6:
        return 3
    elif hits >= 2:
        return 2
    return 1

def run_heuristic_eval(text: str) -> Tuple[pd.DataFrame, str]:
    rows = []
    recs = []
    for cat, patterns in HEURISTICS.items():
        score = max(heuristic_score(text, p) for p in patterns)
        rows.append([cat, score])
        # 간단 추천
        if score < 3:
            if cat == "데이터 수집 및 관리":
                recs.append("데이터 품질(정확성·완전성·신뢰성)과 저장·보호 절차를 명시하세요.")
            elif cat == "데이터 분석 및 해석":
                recs.append("요약을 넘어 상관·가설검증 등 해석 근거를 포함하세요.")
            elif cat == "데이터 활용 및 표현":
                recs.append("시각화 결과를 스토리텔링으로 연결하고 의사결정을 명시하세요.")
            elif cat == "데이터 윤리 및 비판적 사고":
                recs.append("개인정보·편향·출처검증 등 윤리적 성찰 활동을 설계하세요.")
            elif cat == "데이터 인식 및 이해":
                recs.append("정형/비정형·질적/양적 구분과 데이터 생태계를 도식화하세요.")
    df = pd.DataFrame(rows, columns=["범주", "점수"]).set_index("범주")
    summary = "\n".join(sorted(set(recs)))
    return df, summary

# ---- LLM evaluation ----
SYSTEM_PROMPT = """당신은 한국의 교사 연수용 자료를 평가하는 교육과정 전문가입니다.
입력된 문서 텍스트를 바탕으로 데이터리터러시 5대 범주(인식, 수집, 분석, 활용, 윤리)에 대해
각 범주를 1~3점(1=보완 필요, 2=보통, 3=우수)으로 채점하고, 채점 근거 핵심문장/근거 키워드를 제시한 뒤,
종합 보완 사항을 5개 이내로 한국어로 간결히 제안하세요.
결과는 JSON으로만 반환하세요: 
{"rubric": [{"category": "...", "score": 1|2|3, "evidence": ["...","..."]}, ...],
 "recommendations": ["...","..."]}"""

def run_llm_eval(text: str, api_key: str) -> Tuple[pd.DataFrame, str]:
    client = OpenAI(api_key=api_key)
    msg = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": text[:15000]} # 토큰 과다 방지
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0.2
    )
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        # 파싱 실패 시 휴리스틱으로 대체
        return run_heuristic_eval(text)

    rows = []
    for item in data.get("rubric", []):
        rows.append([item.get("category",""), int(item.get("score",2)), " / ".join(item.get("evidence", [])[:3])])
    if not rows:
        return run_heuristic_eval(text)

    df = pd.DataFrame(rows, columns=["범주", "점수", "근거"]).set_index("범주")
    summary = "\n".join(data.get("recommendations", []))
    return df, summary

# ---- Helpers ----
def df_to_csv_download(df: pd.DataFrame, filename: str = "rubric.csv"):
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button("⬇️ 루브릭(CSV) 다운로드", data=csv, file_name=filename, mime="text/csv")

def text_download(content: str, filename: str = "report.md"):
    b = content.encode("utf-8")
    st.download_button("⬇️ 종합 보완 사항(MD) 다운로드", data=b, file_name=filename, mime="text/markdown")

def make_markdown_report(df: pd.DataFrame, recs: str, source_info: str) -> str:
    total = int(df["점수"].sum())
    md = [f"# 데이터리터러시 수업 분석 보고서",
          "",
          f"- 총점: **{total} / 15**",
          "",
          "## 루브릭 결과",
          df.to_markdown(),
          "",
          "## 종합 보완 사항",
          recs or "- (없음)",
          "",
          "## 분석 정보",
          source_info]
    return "\n".join(md)

# ---- UI ----
st.set_page_config(page_title="데이터리터러시 수업 분석 도우미", layout="wide")
st.title("🧪 데이터리터러시 수업 분석 도우미 (Streamlit)")

with st.sidebar:
    st.header("설정")
    st.markdown("- 파일 업로드 후 **분석 시작**을 클릭하세요.\n- OpenAI 키가 있으면 LLM 기반 정밀 평가가 수행됩니다.")
    api_key = st.text_input("OpenAI API Key (선택)", type="password", help="입력 시 LLM 평가 사용, 미입력 시 휴리스틱 평가")
    st.markdown("---")
    st.markdown("**지원 파일**: PDF, DOCX, TXT")

uploaded = st.file_uploader("수업자료 업로드 (여러 개 가능)", type=["pdf","docx","txt"], accept_multiple_files=True)

if uploaded:
    with st.expander("업로드 파일 목록", expanded=False):
        for f in uploaded:
            st.write("•", f.name)

start = st.button("🚀 분석 시작", use_container_width=True)

if start:
    text = extract_text(uploaded) if uploaded else ""
    if not text:
        st.error("텍스트를 추출할 수 없습니다. PDF 스캔본이라면 OCR 후 업로드해 주세요.")
        st.stop()

    # 표시용 소스 정보
    src_names = ", ".join([f.name for f in uploaded]) if uploaded else "입력 텍스트"
    st.info(f"분석 대상: {src_names}")

    # 평가 실행
    if api_key and OPENAI_AVAILABLE:
        df, recs = run_llm_eval(text, api_key)
        st.success("LLM 기반 정밀 평가 완료")
    else:
        df, recs = run_heuristic_eval(text)
        st.warning("OpenAI 키 미입력 → 휴리스틱 평가 수행")

    # 점수 표
    left, right = st.columns([2,1])
    with left:
        st.subheader("📊 루브릭")
        st.dataframe(df, use_container_width=True)
    with right:
        st.metric(label="총점 (15점 만점)", value=int(df["점수"].sum()))

    # 세부 근거 보기 (LLM 모드일 때 근거 컬럼 존재)
    if "근거" in df.columns:
        with st.expander("근거/키워드 보기", expanded=False):
            st.table(df[["근거"]])

    st.subheader("🧭 종합 보완 사항")
    st.write(recs if recs else "- (없음)")

    # 다운로드
    md_report = make_markdown_report(df, recs, f"소스: {src_names}")
    df_to_csv_download(df)
    text_download(md_report, filename="수업분석_보고서.md")

    st.download_button(
        "⬇️ 전체 보고서(Markdown) 다운로드",
        data=md_report.encode("utf-8"),
        file_name="데이터리터러시_분석_보고서.md",
        mime="text/markdown"
    )

# 하단 정보
st.markdown("---")
st.caption("※ 본 도구는 교육적 참고용입니다. 개인정보·저작권·윤리 가이드를 준수하세요.")
