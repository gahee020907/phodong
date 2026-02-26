"""
==============================================================================
🧸 포동 PHODONG — 통합 웹 앱 (Streamlit Cloud 배포용)
==============================================================================
흐름:
  [설정] 이름·나이·장르·목적 입력
    ↓
  [카메라] 사물 찍기 → Gemini Vision으로 캐릭터 생성 (최대 4개)
    ↓
  [동화] 최종 동화 자동 생성 + 표시
==============================================================================
"""

import os, json, re, base64, io, time, logging
from dataclasses import dataclass, field
from typing import Optional, List

import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="포동 PHODONG",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phodong")

# ── 상수 ─────────────────────────────────────────────────────────────────────
MAX_SCENES   = 4
GEMINI_MODEL = "gemini-2.5-flash"

GENRE_OPTIONS   = ["판타지", "전래동화", "일상", "모험", "SF", "자연", "우정", "가족"]
PURPOSE_OPTIONS = ["자신감", "안전", "감정조절", "협동", "창의력", "배려", "도전", "호기심"]

# ── API 키 ────────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_API_KEY", "")

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class StoryConfig:
    child_name:   str = "민준"
    partner_name: str = "친구"
    age:          int = 7
    genre:        str = "판타지"
    purpose:      str = "자신감"

@dataclass
class StoryCard:
    character_name:   str = ""
    character_type:   str = ""
    personality:      str = ""
    magic_power:      str = ""
    dialogue:         str = ""
    story_narration:  str = ""
    image_b64:        str = ""  # base64 인코딩된 이미지

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&family=Gowun+Dodum&display=swap');

    html, body, [class*="css"] {
        font-family: 'Gowun Dodum', sans-serif;
    }

    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #FFF5F7 0%, #FFFFFF 50%, #F0F7FF 100%);
    }

    /* 헤더 */
    .phodong-header {
        text-align: center;
        padding: 40px 20px 20px;
    }
    .phodong-header h1 {
        font-family: 'Jua', sans-serif;
        font-size: 3rem;
        background: linear-gradient(135deg, #FF9EAA, #FF7B8E, #A0C4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .phodong-header p {
        color: #aaa;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    /* 스텝 인디케이터 */
    .step-bar {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin: 20px 0 30px;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: 'Jua', sans-serif;
        font-size: 0.95rem;
        color: #ccc;
    }
    .step-item.active { color: #FF9EAA; }
    .step-item.done   { color: #A0C4FF; }
    .step-dot {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: #eee;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: bold;
    }
    .step-item.active .step-dot { background: #FF9EAA; color: white; }
    .step-item.done   .step-dot { background: #A0C4FF; color: white; }
    .step-line { width: 40px; height: 2px; background: #eee; margin-top: 14px; }

    /* 카드 */
    .phodong-card {
        background: white;
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 10px 40px rgba(255,158,170,0.10);
        border: 2px solid #FFE3F1;
        margin-bottom: 20px;
    }

    /* 설정 섹션 레이블 */
    .section-label {
        font-family: 'Jua', sans-serif;
        font-size: 1.1rem;
        color: #FF9EAA;
        margin-bottom: 12px;
    }

    /* 옵션 버튼 */
    .stButton > button {
        border-radius: 18px !important;
        font-family: 'Jua' !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
    }

    /* 캐릭터 카드 */
    .char-card {
        background: linear-gradient(135deg, #FFF5F7, #F0F7FF);
        border-radius: 20px;
        padding: 20px 24px;
        border: 2px solid #FFE3F1;
        margin-bottom: 12px;
        animation: fadeInUp 0.4s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .char-name {
        font-family: 'Jua', sans-serif;
        font-size: 1.3rem;
        color: #FF7B8E;
        margin-bottom: 6px;
    }
    .char-dialogue {
        font-size: 1rem;
        color: #5D4037;
        background: #FFFBE6;
        border-radius: 12px;
        padding: 10px 14px;
        border-left: 4px solid #FFD580;
        margin-top: 8px;
        line-height: 1.7;
    }

    /* 씬 카운터 */
    .scene-counter {
        font-family: 'Jua', sans-serif;
        font-size: 1.1rem;
        color: #A0C4FF;
        text-align: center;
        margin: 10px 0 20px;
    }

    /* 동화 본문 */
    .story-body {
        background: linear-gradient(180deg, #FFFFFE 0%, #FFF9F5 100%);
        padding: 45px 55px;
        border-radius: 20px;
        border: 3px solid #FFE3F1;
        font-family: 'Gowun Dodum', sans-serif;
        font-size: 1.2rem;
        line-height: 2.2;
        color: #4A4A4A;
        position: relative;
        white-space: pre-line;
    }
    .story-title {
        font-family: 'Jua', sans-serif;
        font-size: 2rem;
        background: linear-gradient(135deg, #FF9EAA, #FF7B8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    .the-end {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 2px dashed #FFE3F1;
        font-family: 'Jua', sans-serif;
        font-size: 1.3rem;
        color: #FFD580;
    }

    /* 카메라 안내 */
    .camera-guide {
        text-align: center;
        padding: 16px;
        background: linear-gradient(135deg, #F0F7FF, #FFFFFF);
        border-radius: 16px;
        border: 2px dashed #A0C4FF;
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }

    /* 배지 */
    .badge-row {
        display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-family: 'Jua', sans-serif;
    }
    .badge-pink   { background: #FFE3F1; color: #D63384; }
    .badge-blue   { background: #E3F2FD; color: #1E429F; }
    .badge-yellow { background: #FFF5C4; color: #B7791F; }

    /* 숨기기 */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ── SVG 아이콘 ────────────────────────────────────────────────────────────────
BEAR_SVG = """
<svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="55" r="35" fill="#D6B898"/>
  <circle cx="35" cy="25" r="12" fill="#D6B898"/>
  <circle cx="65" cy="25" r="12" fill="#D6B898"/>
  <circle cx="35" cy="25" r="6" fill="#EAC7A8"/>
  <circle cx="65" cy="25" r="6" fill="#EAC7A8"/>
  <ellipse cx="50" cy="60" rx="14" ry="10" fill="#FFF0F5"/>
  <circle cx="50" cy="56" r="4" fill="#5D4037"/>
  <circle cx="42" cy="48" r="3" fill="#333"/>
  <circle cx="58" cy="48" r="3" fill="#333"/>
  <path d="M50 60V65" stroke="#5D4037" stroke-width="2" stroke-linecap="round"/>
  <path d="M46 65C46 65 48 68 50 68C52 68 54 65 54 65" stroke="#5D4037" stroke-width="2" stroke-linecap="round"/>
</svg>
"""


# ── 연령별 언어 지침 ──────────────────────────────────────────────────────────
def age_language_guide(age: int) -> str:
    guides = {
        5: ("친숙하고 일상적인 단어를 사용하세요. 문장은 자연스럽게 이어지되 간결하게 써주세요. "
            "의성어·의태어를 적극 활용하고 (예: 반짝반짝, 살금살금), 어려운 한자어나 추상적 개념은 피하세요. "
            "이야기 흐름(배경→사건→해결)을 유지하되 각 장면을 2~3문장으로 자연스럽게 묘사하세요."),
        6: ("친숙한 어휘와 짧은 복문을 사용하세요 (예: ~했어요, 그래서 ~). "
            "감정 표현(기뻐요, 무서워요)과 의성어·의태어를 활용하고 각 장면을 3~4문장으로 묘사하세요."),
        7: ("원인과 결과 표현을 사용하세요 (왜냐하면, 그래서, 하지만). "
            "인물의 감정과 의도를 구체적으로 묘사하고 각 장면을 4~5문장으로 풍부하게 서술하세요."),
        8: ("비유 표현(마치 ~처럼)과 다양한 어휘를 활용하세요. "
            "인물의 심리와 사건의 인과관계를 상세히 묘사하여 풍부한 이야기를 만들어주세요."),
    }
    return guides.get(age, guides[7])


# ── Gemini 캐릭터 생성 ────────────────────────────────────────────────────────
def generate_character(image: Image.Image, config: StoryConfig, seen_types: list) -> Optional[dict]:
    api_key = get_api_key()
    if not api_key:
        st.error("API 키가 설정되지 않았습니다.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    seen_str = ", ".join(seen_types) if seen_types else "없음"

    prompt = f"""
당신은 {config.age}세 아이를 위한 창의적인 동화 작가입니다.
카메라 속 사물을 '살아있는 캐릭터'로 만들어 주인공({config.child_name})에게 말을 걸어주세요.

[연령별 언어 수준 — 반드시 준수]
{age_language_guide(config.age)}

[캐릭터 설정]
1. 이름: {config.genre} 장르에 어울리는 기발하고 재미있는 이름
2. 능력: 이 사물이 가진 특별한 마법 능력이나 기능
3. 성격: 사물의 생김새나 용도에 어울리는 성격

[대사]
주인공({config.child_name})이나 짝꿍({config.partner_name})에게 건네는 말. {config.purpose}와 관련된 조언 포함.

[주의]
이미 등장한 사물: {seen_str}
위 사물과 동일하거나 매우 유사한 사물이면 "has_interesting_object": false 로 설정하세요.

결과는 반드시 아래 JSON 형식으로만 응답하세요:
{{
    "has_interesting_object": true,
    "character_name": "캐릭터 이름",
    "character_type": "원래 사물 이름",
    "magic_power": "마법 능력",
    "personality": "성격",
    "dialogue": "주인공에게 하는 대사",
    "story_narration": "상황 설명"
}}
사물이 없거나 중복이면 "has_interesting_object": false 로 설정하세요.
"""
    try:
        response = model.generate_content([prompt, image])
        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        return data if data.get("has_interesting_object") else None
    except Exception as e:
        logger.error(f"캐릭터 생성 오류: {e}")
        return None


# ── Gemini 동화 생성 ──────────────────────────────────────────────────────────
def generate_story(cards: List[StoryCard], config: StoryConfig) -> str:
    api_key = get_api_key()
    if not api_key:
        return "API 키 오류"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    scenes = "\n".join([
        f"- {c.character_name}({c.character_type}): \"{c.dialogue}\" / {c.story_narration}"
        for c in cards
    ])

    prompt = f"""
전문 동화 작가로서 '{config.child_name}'와 '{config.partner_name}'의 한국어 동화를 작성하세요.

[독자 정보]
- 대상 연령: {config.age}세
- 장르: {config.genre}
- 교육 목적: {config.purpose}

[언어 수준 — 반드시 준수]
{age_language_guide(config.age)}

[조건]
1. 첫 줄은 동화 제목만 작성하세요.
2. 아이에게 읽어주는 따뜻한 해요체를 사용하세요.
3. 아래 장면을 모두 포함하여 자연스럽게 연결하세요.
4. 교육 목적({config.purpose})이 설교적이지 않고 이야기 흐름 안에 자연스럽게 녹아들게 하세요.
5. 마지막 줄은 "끝." 으로 마무리하세요.

[장면 목록]
{scenes}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"동화 생성 오류: {e}"


# ── 이미지 → base64 ───────────────────────────────────────────────────────────
def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── 세션 초기화 ───────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "step":         "config",   # config → camera → story
        "config":       None,
        "cards":        [],
        "seen_types":   [],
        "story_text":   "",
        "processing":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── 헤더 + 스텝바 ─────────────────────────────────────────────────────────────
def render_header():
    bear = BEAR_SVG
    st.markdown(
        f'<div class="phodong-header">{bear}<h1>포동 PHODONG</h1><p>사물이 살아있는 나만의 동화</p></div>',
        unsafe_allow_html=True
    )

def render_stepbar(current: str):
    steps = [("config", "설정"), ("camera", "촬영"), ("story", "동화")]
    keys = [s[0] for s in steps]
    current_idx = keys.index(current) if current in keys else 0
    items = []
    for i, (key, label) in enumerate(steps):
        if key == current:
            cls = "active"
        elif i < current_idx:
            cls = "done"
        else:
            cls = ""
        line = "<div class='step-line'></div>" if i < len(steps)-1 else ""
        items.append(
            f"<div class='step-item {cls}'><div class='step-dot'>{i+1}</div> {label}</div>{line}"
        )
    st.markdown("<div class='step-bar'>" + "".join(items) + "</div>", unsafe_allow_html=True)


# ── STEP 1: 설정 화면 ─────────────────────────────────────────────────────────
def render_config():
    st.markdown('<div class="phodong-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">👤 아이 정보</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        child_name = st.text_input("주인공 이름", value="", placeholder="예: 민준")
    with col2:
        partner_name = st.text_input("짝꿍 이름", value="", placeholder="예: 뽀로로")
    with col3:
        age = st.selectbox("나이", options=[5, 6, 7, 8], index=2)

    st.markdown('<p class="section-label" style="margin-top:20px">📚 장르</p>', unsafe_allow_html=True)
    genre_cols = st.columns(len(GENRE_OPTIONS))
    selected_genre = st.session_state.get("sel_genre", GENRE_OPTIONS[0])
    for i, g in enumerate(GENRE_OPTIONS):
        with genre_cols[i]:
            if st.button(g, key=f"genre_{g}",
                         type="primary" if selected_genre == g else "secondary",
                         use_container_width=True):
                st.session_state["sel_genre"] = g
                st.rerun()

    st.markdown('<p class="section-label" style="margin-top:20px">🎯 이야기 목적</p>', unsafe_allow_html=True)
    purpose_cols = st.columns(len(PURPOSE_OPTIONS))
    selected_purpose = st.session_state.get("sel_purpose", PURPOSE_OPTIONS[0])
    for i, p in enumerate(PURPOSE_OPTIONS):
        with purpose_cols[i]:
            if st.button(p, key=f"purpose_{p}",
                         type="primary" if selected_purpose == p else "secondary",
                         use_container_width=True):
                st.session_state["sel_purpose"] = p
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("✨ 모험 시작하기!", type="primary", use_container_width=True):
            if not child_name.strip():
                st.warning("주인공 이름을 입력해주세요.")
                return
            st.session_state["config"] = StoryConfig(
                child_name=child_name.strip(),
                partner_name=partner_name.strip() or "친구",
                age=age,
                genre=st.session_state.get("sel_genre", GENRE_OPTIONS[0]),
                purpose=st.session_state.get("sel_purpose", PURPOSE_OPTIONS[0]),
            )
            st.session_state["step"]   = "camera"
            st.session_state["cards"]  = []
            st.session_state["seen_types"] = []
            st.rerun()


# ── STEP 2: 카메라 화면 ───────────────────────────────────────────────────────
def render_camera():
    config: StoryConfig = st.session_state["config"]
    cards:  List[StoryCard] = st.session_state["cards"]
    n = len(cards)

    # 씬 카운터
    st.markdown(
        f'<div class="scene-counter">📸 {n} / {MAX_SCENES} 장면 완성</div>',
        unsafe_allow_html=True
    )

    # 진행바
    st.progress(n / MAX_SCENES)

    # 완료 시 동화로 이동
    if n >= MAX_SCENES:
        st.success(f"🎉 {MAX_SCENES}개 장면 완성! 동화를 만들고 있어요...")
        time.sleep(1)
        with st.spinner("✨ 동화 생성 중..."):
            st.session_state["story_text"] = generate_story(cards, config)
        st.session_state["step"] = "story"
        st.rerun()
        return

    # 카메라 안내
    st.markdown(f"""
    <div class="camera-guide">
        📷 사물을 카메라에 비추고 <b>촬영 버튼</b>을 눌러주세요<br>
        <span style="color:#A0C4FF">{config.child_name}의 동화 친구를 찾고 있어요!</span>
    </div>
    """, unsafe_allow_html=True)

    # 카메라 입력
    cam_col, result_col = st.columns([1, 1])

    with cam_col:
        img_file = st.camera_input("", label_visibility="collapsed")

        if img_file and not st.session_state.get("processing"):
            st.session_state["processing"] = True
            image = Image.open(img_file).convert("RGB")
            image.thumbnail((800, 800))

            with st.spinner("🔍 사물을 분석하고 있어요..."):
                data = generate_character(image, config, st.session_state["seen_types"])

            st.session_state["processing"] = False

            if data:
                card = StoryCard(
                    character_name=data.get("character_name", ""),
                    character_type=data.get("character_type", ""),
                    personality=data.get("personality", ""),
                    magic_power=data.get("magic_power", ""),
                    dialogue=data.get("dialogue", ""),
                    story_narration=data.get("story_narration", ""),
                    image_b64=image_to_b64(image),
                )
                st.session_state["cards"].append(card)
                st.session_state["seen_types"].append(data.get("character_type", ""))
                st.rerun()
            else:
                st.warning("사물을 인식하지 못했어요. 다시 찍어주세요!")

    # 발견된 캐릭터 목록
    with result_col:
        if cards:
            st.markdown('<p class="section-label">🌟 발견된 동화 친구들</p>', unsafe_allow_html=True)
            for card in cards:
                img_col, text_col = st.columns([1, 2])
                with img_col:
                    if card.image_b64:
                        st.image(b64_to_image(card.image_b64), use_container_width=True)
                with text_col:
                    st.markdown(f"""
                    <div class="char-card">
                        <div class="char-name">✨ {card.character_name}</div>
                        <div class="badge-row">
                            <span class="badge badge-pink">{card.character_type}</span>
                            <span class="badge badge-blue">{card.magic_power[:15]}...</span>
                        </div>
                        <div class="char-dialogue">"{card.dialogue}"</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#ccc; padding:40px 20px;">
                <div style="font-size:3rem">🔍</div>
                <p>아직 동화 친구를 발견하지 못했어요</p>
            </div>
            """, unsafe_allow_html=True)

    # 처음으로 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← 처음으로", type="secondary"):
        st.session_state["step"] = "config"
        st.rerun()


# ── STEP 3: 동화 화면 ─────────────────────────────────────────────────────────
def render_story():
    config: StoryConfig = st.session_state["config"]
    cards:  List[StoryCard] = st.session_state["cards"]
    story:  str = st.session_state.get("story_text", "")

    if not story:
        with st.spinner("✨ 동화 생성 중..."):
            story = generate_story(cards, config)
            st.session_state["story_text"] = story

    # 제목/본문 분리
    lines = story.strip().split("\n")
    title = lines[0].strip() if lines else "나만의 동화"
    body  = "\n".join(lines[1:]).strip() if len(lines) > 1 else story

    # 헤더
    st.markdown(f"""
    <div class="phodong-card" style="text-align:center; margin-bottom:16px;">
        <div style="font-family:'Jua',sans-serif; font-size:0.9rem; color:#aaa; margin-bottom:6px;">
            {config.age}세 · {config.genre} · {config.purpose}
        </div>
        <div class="story-title">{title}</div>
        <div style="color:#aaa; font-size:0.95rem">
            주인공: {config.child_name} & {config.partner_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 동화 본문
    st.markdown(f"""
    <div class="story-body">
        <div class="story-text">{body}</div>
        <div class="the-end">🌟 끝 🌟</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 등장인물 요약
    with st.expander("📖 등장 캐릭터 보기"):
        for card in cards:
            c1, c2 = st.columns([1, 3])
            with c1:
                if card.image_b64:
                    st.image(b64_to_image(card.image_b64), use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class="char-card">
                    <div class="char-name">{card.character_name}</div>
                    <div class="badge-row">
                        <span class="badge badge-pink">{card.character_type}</span>
                        <span class="badge badge-yellow">{card.personality[:20]}</span>
                    </div>
                    <div class="char-dialogue">"{card.dialogue}"</div>
                </div>
                """, unsafe_allow_html=True)

    # 다운로드
    st.download_button(
        label="📥 동화 저장하기",
        data=f"{title}\n\n{body}",
        file_name=f"포동_{config.child_name}의동화.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 새 동화 만들기", type="primary", use_container_width=True):
        for key in ["step", "config", "cards", "seen_types", "story_text",
                    "processing", "sel_genre", "sel_purpose"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session()
    render_header()
    render_stepbar(st.session_state["step"])

    step = st.session_state["step"]
    if step == "config":
        render_config()
    elif step == "camera":
        render_camera()
    elif step == "story":
        render_story()


if __name__ == "__main__":
    main()
