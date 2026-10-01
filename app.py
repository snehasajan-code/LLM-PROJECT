import inspect
import html
import textwrap
import streamlit as st

from translator import translate_text
from ambiguity import detect_ambiguity
from verifier import verify_translation
from candidate_engine import translate_and_rank


# ============================================================
# OPTIONAL MODULES
# ============================================================

try:
    from evaluator import evaluate_translation
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False


try:
    from repair_engine import repair_translation
    REPAIR_AVAILABLE = True
except ImportError:
    REPAIR_AVAILABLE = False


try:
    from voice_engine import (
        process_voice_input,
        text_to_speech
    )
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Multilingual Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FUTURISTIC UI
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL BACKGROUND
   ========================================================= */

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(72, 100, 255, 0.18),
            transparent 27%
        ),

        radial-gradient(
            circle at 90% 15%,
            rgba(175, 70, 255, 0.15),
            transparent 28%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(0, 220, 190, 0.08),
            transparent 32%
        ),

        linear-gradient(
            135deg,
            #02040e 0%,
            #070a1c 45%,
            #030511 100%
        );

    color: #f5f7ff;

    overflow-x: hidden;
}


/* =========================================================
   STREAMLIT HEADER
   ========================================================= */

header[data-testid="stHeader"] {

    background:
        rgba(3, 5, 17, 0.72) !important;

    backdrop-filter:
        blur(18px);

    border-bottom:
        1px solid rgba(120, 140, 255, 0.08);
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.main .block-container {

    position: relative;

    z-index: 5;

    max-width: 1450px;

    padding-top: 2rem;

    padding-bottom: 4rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(3, 6, 22, 0.98),
            rgba(2, 4, 15, 0.99)
        ) !important;

    border-right:
        1px solid rgba(120, 140, 255, 0.17);

    box-shadow:
        10px 0 50px rgba(0, 0, 0, 0.38);

    z-index: 20;
}


section[data-testid="stSidebar"] * {
    color: #edf0ff;
}


/* =========================================================
   ANIMATED MULTILINGUAL ATMOSPHERE
   ========================================================= */

/*
   The moving language field is created entirely with CSS pseudo-elements.
   This avoids rendering raw <span> tags in Streamlit and keeps the
   animation behind the application UI.
*/

.stApp::before,
.stApp::after {

    position: fixed;

    left: -15vw;

    width: 130vw;

    height: 100vh;

    pointer-events: none;

    z-index: 0;

    user-select: none;

    white-space: pre;

    font-weight: 700;

    line-height: 2.5;

    letter-spacing: 1.4vw;

    text-shadow:
        0 0 12px rgba(35, 205, 255, 0.48),
        0 0 30px rgba(130, 100, 255, 0.24);

    overflow: hidden;
}


/* First multilingual layer */

.stApp::before {

    content:
        "A     क       ക        ஆ      文       గ       ಗ       A       અ\\A"
        "   अ       T       മ       क       અ       தமிழ்       A       ગ\\A"
        "文       അ       B       హ       क       A       മ       त       文\\A"
        "     ഗ       अ       ಕನ್ನಡ       E       അ       క       A       क";

    top: -4vh;

    color: rgba(95, 193, 255, 0.19);

    font-size: clamp(27px, 2.8vw, 48px);

    line-height: 2.8;

    filter: drop-shadow(0 0 10px rgba(36, 172, 255, 0.32));

    transform: rotate(-7deg) translate3d(0, 0, 0);

    animation:
        languageDriftOne
        24s
        ease-in-out
        infinite alternate;

    background:
        radial-gradient(
            circle at 18% 30%,
            rgba(76, 108, 255, 0.13),
            transparent 25%
        ),
        radial-gradient(
            circle at 82% 68%,
            rgba(182, 78, 255, 0.10),
            transparent 24%
        );
}


/* Second multilingual layer */

.stApp::after {

    content:
        "   अ       ആ       M       અ       క       E       ഗ       A\\A"
        "ಕ       A       क       文       മ       T       அ       અ       क\\A"
        "   E       హ       A       തമിഴ்       क       ನ       अ       A";

    top: 22vh;

    color: rgba(208, 128, 255, 0.17);

    font-size: clamp(24px, 2.35vw, 41px);

    line-height: 3.1;

    filter: drop-shadow(0 0 12px rgba(184, 75, 255, 0.35));

    transform: rotate(8deg) translate3d(0, 0, 0);

    animation:
        languageDriftTwo
        30s
        ease-in-out
        infinite alternate;

    background:
        radial-gradient(
            circle at 72% 20%,
            rgba(255, 103, 210, 0.10),
            transparent 22%
        ),
        radial-gradient(
            circle at 28% 80%,
            rgba(50, 220, 190, 0.08),
            transparent 25%
        );
}


/* Slow floating motion */

@keyframes languageDriftOne {

    0% {

        transform:
            rotate(-7deg)
            translate3d(-3vw, -2vh, 0)
            scale(1);

        opacity: 0.45;
    }

    25% {

        transform:
            rotate(-5deg)
            translate3d(4vw, 5vh, 0)
            scale(1.03);

        opacity: 0.72;
    }

    50% {

        transform:
            rotate(-8deg)
            translate3d(-1vw, 10vh, 0)
            scale(0.98);

        opacity: 0.52;
    }

    75% {

        transform:
            rotate(-4deg)
            translate3d(6vw, 4vh, 0)
            scale(1.05);

        opacity: 0.68;
    }

    100% {

        transform:
            rotate(-7deg)
            translate3d(-4vw, -1vh, 0)
            scale(1);

        opacity: 0.45;
    }
}


@keyframes languageDriftTwo {

    0% {

        transform:
            rotate(8deg)
            translate3d(5vw, 5vh, 0)
            scale(1);

        opacity: 0.30;
    }

    30% {

        transform:
            rotate(5deg)
            translate3d(-4vw, 12vh, 0)
            scale(1.04);

        opacity: 0.55;
    }

    60% {

        transform:
            rotate(9deg)
            translate3d(7vw, 18vh, 0)
            scale(0.97);

        opacity: 0.38;
    }

    100% {

        transform:
            rotate(5deg)
            translate3d(-2vw, 8vh, 0)
            scale(1.02);

        opacity: 0.48;
    }
}


/* Extra ambient light that moves across the screen */

.stApp {

    position: relative;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(72, 100, 255, 0.18),
            transparent 27%
        ),

        radial-gradient(
            circle at 90% 15%,
            rgba(175, 70, 255, 0.15),
            transparent 28%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(0, 220, 190, 0.08),
            transparent 32%
        ),

        linear-gradient(
            135deg,
            #02040e 0%,
            #070a1c 45%,
            #030511 100%
        );

    background-size:
        140% 140%,
        130% 130%,
        150% 150%,
        100% 100%;

    animation:
        ambientBackground
        18s
        ease-in-out
        infinite alternate;
}


@keyframes ambientBackground {

    0% {

        background-position:
            0% 0%,
            100% 0%,
            50% 100%,
            0% 0%;
    }

    50% {

        background-position:
            18% 12%,
            82% 22%,
            40% 85%,
            0% 0%;
    }

    100% {

        background-position:
            35% 25%,
            65% 35%,
            60% 70%,
            0% 0%;
    }
}


/* Keep application content above the animated atmosphere */

/* Streamlit's actual main content container */
section[data-testid="stMain"] {
    position: relative !important;
    z-index: 10 !important;
}

/* Soft wireframe globe, echoing the reference without competing with the UI */
section[data-testid="stMain"]::before {
    content: "";
    position: fixed;
    z-index: 0;
    pointer-events: none;
    width: min(38vw, 500px);
    height: min(38vw, 500px);
    min-width: 300px;
    min-height: 300px;
    right: 3vw;
    top: 20vh;
    border: 1px solid rgba(49, 185, 255, 0.28);
    border-radius: 50%;
    opacity: 0.40;
    background:
        radial-gradient(circle at 34% 28%, rgba(18, 139, 255, 0.25), transparent 56%),
        repeating-radial-gradient(ellipse at center, transparent 0 28px, rgba(35, 187, 255, 0.16) 29px 30px, transparent 31px 45px),
        repeating-linear-gradient(0deg, transparent 0 35px, rgba(46, 166, 255, 0.12) 36px 37px, transparent 38px 52px),
        repeating-linear-gradient(90deg, transparent 0 35px, rgba(46, 166, 255, 0.12) 36px 37px, transparent 38px 52px),
        radial-gradient(circle, rgba(4, 25, 62, 0.2), rgba(2, 8, 24, 0.5) 72%);
    box-shadow: inset -24px -8px 54px rgba(1, 7, 28, 0.75), 0 0 45px rgba(13, 139, 255, 0.16);
    mask-image: radial-gradient(circle, #000 62%, transparent 71%);
}

section[data-testid="stMain"] .block-container {
    position: relative !important;
    z-index: 11 !important;
}

/* App view wrapper must create a clean stacking context */
div[data-testid="stAppViewContainer"] {
    position: relative !important;
    z-index: 1 !important;
}

section[data-testid="stSidebar"] {
    position: relative !important;
    z-index: 20 !important;
}

header[data-testid="stHeader"] {
    position: relative !important;
    z-index: 30 !important;
}


/* =========================================================
   HERO
   ========================================================= */


.hero {

    position: relative;

    text-align: center;

    padding:
        48px
        30px
        50px
        30px;

    margin-bottom: 28px;

    border-radius: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(18, 25, 60, 0.78),
            rgba(7, 11, 31, 0.64)
        );

    border:
        1px solid rgba(135, 150, 255, 0.20);

    box-shadow:
        0 25px 80px rgba(0, 0, 0, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);

    backdrop-filter:
        blur(20px);

    overflow: hidden;
}


/* Animated glow */

.hero::before {

    content: "";

    position: absolute;

    width: 600px;

    height: 600px;

    top: -450px;

    left: 50%;

    transform: translateX(-50%);

    background:
        radial-gradient(
            circle,
            rgba(90, 120, 255, 0.20),
            transparent 65%
        );

    animation:
        heroGlow 6s ease-in-out infinite;
}


@keyframes heroGlow {

    0%, 100% {

        opacity: 0.45;

        transform:
            translateX(-50%)
            scale(1);
    }

    50% {

        opacity: 0.85;

        transform:
            translateX(-50%)
            scale(1.15);
    }
}


/* =========================================================
   HERO TITLE
   ========================================================= */

.hero-title {

    position: relative;

    z-index: 2;

    font-size: 54px;

    font-weight: 900;

    letter-spacing: -2px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #9ed5ff,
            #c6a2ff,
            #ffb8eb,
            #ffffff
        );

    background-size: 300% auto;

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation:
        titleGradient
        6s
        ease
        infinite;
}


@keyframes titleGradient {

    0% {
        background-position: 0% center;
    }

    50% {
        background-position: 100% center;
    }

    100% {
        background-position: 0% center;
    }
}


/* =========================================================
   HERO SUBTITLE
   ========================================================= */

.hero-subtitle {

    position: relative;

    z-index: 2;

    color: #adb8df;

    font-size: 16px;

    margin-top: 12px;

    letter-spacing: 0.4px;
}


/* =========================================================
   BADGE
   ========================================================= */

.badge {

    position: absolute;

    z-index: 2;

    display: inline-block;

    top: 22px;

    right: 24px;

    padding:
        9px
        18px;

    border-radius: 999px;

    background:
        rgba(100, 120, 255, 0.08);

    border:
        1px solid rgba(74, 198, 255, 0.45);

    color: #e1f6ff;

    font-size: 13px;

    box-shadow:
        0 0 25px rgba(100, 120, 255, 0.08);

    animation:
        badgePulse
        3s
        ease-in-out
        infinite;
}


@keyframes badgePulse {

    0%, 100% {

        box-shadow:
            0 0 20px
            rgba(100, 120, 255, 0.08);
    }

    50% {

        box-shadow:
            0 0 38px
            rgba(130, 100, 255, 0.22);
    }
}


/* =========================================================
   GLASS PANELS
   ========================================================= */

.glass-panel {

    position: relative;

    background:
        linear-gradient(
            135deg,
            rgba(18, 25, 58, 0.78),
            rgba(7, 11, 31, 0.72)
        );

    border:
        1px solid rgba(130, 145, 255, 0.18);

    border-radius: 22px;

    padding: 28px;

    margin-bottom: 22px;

    backdrop-filter:
        blur(18px);

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.32),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);

    transition:
        transform 0.3s ease,
        border-color 0.3s ease,
        box-shadow 0.3s ease;
}


.glass-panel:hover {

    transform:
        translateY(-2px);

    border-color:
        rgba(140, 160, 255, 0.30);

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.40),
        0 0 35px rgba(90, 110, 255, 0.07);
}


/* =========================================================
   SECTION LABEL
   ========================================================= */

.step-label {

    color: #7f8ac0;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 2px;

    margin-bottom: 6px;
}


/* =========================================================
   TRANSLATION RESULT
   ========================================================= */

.translation-box {

    position: relative;

    background:
        linear-gradient(
            135deg,
            rgba(13, 52, 59, 0.80),
            rgba(10, 34, 47, 0.66)
        );

    border:
        1px solid rgba(80, 230, 190, 0.27);

    border-radius: 18px;

    padding: 25px;

    font-size: 22px;

    line-height: 1.8;

    color: #eafff9;

    box-shadow:
        0 0 35px rgba(50, 210, 180, 0.06);

    overflow: hidden;
}


.translation-box::after {

    content: "";

    position: absolute;

    top: 0;

    left: -100%;

    width: 50%;

    height: 100%;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.07),
            transparent
        );

    animation:
        resultShine
        5s
        ease-in-out
        infinite;
}


@keyframes resultShine {

    0% {
        left: -100%;
    }

    60% {
        left: 120%;
    }

    100% {
        left: 120%;
    }
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    position: relative;

    border-radius: 13px !important;

    border:
        1px solid rgba(130, 150, 255, 0.30) !important;

    background:
        linear-gradient(
            135deg,
            #08b9e8,
            #3563ff 48%,
            #c21bea
        ) !important;

    color: white !important;

    font-weight: 700 !important;

    min-height: 46px;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


.stButton > button:hover {

    transform:
        translateY(-3px)
        scale(1.01);

    box-shadow:
        0 10px 30px
        rgba(80, 100, 255, 0.28);

    border-color:
        rgba(190, 200, 255, 0.55) !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

textarea,
input {

    background:
        rgba(5, 9, 27, 0.80) !important;

    color:
        #f4f6ff !important;

    border-radius:
        14px !important;

    border:
        1px solid
        rgba(120, 140, 255, 0.20) !important;
}


textarea:focus,
input:focus {

    border:
        1px solid
        rgba(120, 160, 255, 0.55) !important;

    box-shadow:
        0 0 25px
        rgba(80, 120, 255, 0.12) !important;
}


/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {

    background:
        rgba(18, 25, 55, 0.58);

    border:
        1px solid
        rgba(120, 140, 255, 0.15);

    border-radius:
        15px;

    padding:
        13px;

    transition:
        transform 0.25s ease;
}


div[data-testid="stMetric"]:hover {

    transform:
        translateY(-3px);
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {

    font-size: 15px !important;

    font-weight: 650 !important;

    color:
        #7f89b5 !important;
}


button[data-baseweb="tab"][aria-selected="true"] {

    color:
        #dbe2ff !important;

    text-shadow:
        0 0 15px
        rgba(120, 150, 255, 0.5);
}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {

    border-radius:
        14px !important;

    backdrop-filter:
        blur(10px);
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {

    border-color:
        rgba(120, 140, 255, 0.12) !important;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #050713;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            #465bc0,
            #7448a9
        );

    border-radius: 10px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .hero-title {
        font-size: 38px;
    }

    .hero {
        padding: 35px 20px;
    }

    .badge {
        position: relative;
        top: auto;
        right: auto;
        margin-top: 18px;
        max-width: 100%;
        white-space: normal;
    }

    .stApp::before,
    .stApp::after {
        font-size: 20px;
        letter-spacing: 3vw;
        line-height: 3;
    }

    section[data-testid="stMain"]::before {
        width: 280px;
        height: 280px;
        min-width: 0;
        min-height: 0;
        right: -110px;
        top: 17vh;
        opacity: 0.24;
    }
}

@media (prefers-reduced-motion: reduce) {
    .stApp::before,
    .stApp::after,
    .hero::before,
    .hero-title,
    .badge,
    .translation-box::after {
        animation: none !important;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "ambiguity_result": None,

    "translation_result": None,

    "verification": None,

    "evaluation": None,

    "repair_result": None,

    "selected_interpretation": None,

    "selected_constraint": None,

    "translated_for_text": None,

    "voice_result": None,

    "voice_translation_result": None,

    "voice_verification": None,

    "voice_evaluation": None,

    "voice_repair_result": None,

    "voice_audio_bytes": None,

    "voice_tts_path": None,

    "current_source_text": None,

    "current_source_language": None,

    "current_target_language": None,

    "current_context": None,

    "current_tone": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def reset_translation_results():

    st.session_state.translation_result = None

    st.session_state.verification = None

    st.session_state.evaluation = None

    st.session_state.repair_result = None

    st.session_state.selected_interpretation = None

    st.session_state.selected_constraint = None


# ============================================================
# GENERIC FUNCTION CALLER
# ============================================================

def call_supported_function(function, arguments):

    try:

        signature = inspect.signature(function)

        parameters = signature.parameters

        accepted = {}

        for name, value in arguments.items():

            if name in parameters:

                accepted[name] = value

        return function(**accepted)

    except Exception:

        return function(**arguments)


# ============================================================
# VERIFICATION
# ============================================================

def run_verification(
    original_text,
    translated_text,
    source_language,
    target_language,
    selected_meaning=None,
    semantic_constraint=None,
    context="",
    language_confidence=None
):

    arguments = {

        "original_text": original_text,

        "translated_text": translated_text,

        "source_language": source_language,

        "target_language": target_language,

        "selected_meaning": selected_meaning,

        "semantic_constraint": semantic_constraint,

        "human_clarification": selected_meaning,

        "clarification": selected_meaning,

        "context": context,

        "language_confidence": language_confidence
    }

    return call_supported_function(
        verify_translation,
        arguments
    )


# ============================================================
# QUALITY EVALUATION
# ============================================================

def run_quality_evaluation(
    original_text,
    translated_text,
    source_language,
    target_language,
    context,
    tone,
    ambiguity_result=None,
    verification=None,
    selected_meaning=None
):

    if not EVALUATOR_AVAILABLE:

        return {

            "available": False,

            "meaning": "N/A",

            "grammar_naturalness": "N/A",

            "ambiguity_resolution": "N/A",

            "context": "N/A",

            "tone": "N/A",

            "overall": "N/A",

            "strengths": [],

            "issues": [
                "evaluator.py is not available."
            ],

            "explanation": ""
        }


    arguments = {

        "original_text": original_text,

        "translated_text": translated_text,

        "source_language": source_language,

        "target_language": target_language,

        "context": context,

        "tone": tone,

        "ambiguity_result": ambiguity_result,

        "verification": verification,

        "selected_meaning": selected_meaning,

        "human_clarification": selected_meaning,

        "clarification": selected_meaning
    }


    try:

        result = call_supported_function(
            evaluate_translation,
            arguments
        )

        if not isinstance(result, dict):

            return {

                "available": False,

                "meaning": "N/A",

                "grammar_naturalness": "N/A",

                "ambiguity_resolution": "N/A",

                "context": "N/A",

                "tone": "N/A",

                "overall": "N/A",

                "strengths": [],

                "issues": [
                    "Evaluator returned an invalid response."
                ],

                "explanation": ""
            }


        result["available"] = True

        return result


    except Exception as e:

        return {

            "available": False,

            "meaning": "N/A",

            "grammar_naturalness": "N/A",

            "ambiguity_resolution": "N/A",

            "context": "N/A",

            "tone": "N/A",

            "overall": "N/A",

            "strengths": [],

            "issues": [
                f"Evaluation failed: {e}"
            ],

            "explanation": ""
        }


# ============================================================
# ADAPTIVE REPAIR
# ============================================================

def run_adaptive_repair(
    original_text,
    translated_text,
    verification,
    source_language,
    target_language,
    context,
    tone,
    selected_meaning=None,
    semantic_constraint=None
):

    if not REPAIR_AVAILABLE:

        return {

            "attempted": False,

            "repaired": False,

            "reason":
                "repair_engine.py is not available."
        }


    try:

        return repair_translation(

            original_text=original_text,

            translated_text=translated_text,

            verification=verification,

            source_language=source_language,

            target_language=target_language,

            context=context,

            tone=tone,

            selected_meaning=selected_meaning,

            semantic_constraint=semantic_constraint
        )


    except Exception as e:

        return {

            "attempted": True,

            "repaired": False,

            "reason":
                f"Adaptive repair failed: {e}"
        }


# ============================================================
# DISPLAY VERIFICATION
# ============================================================

def display_verification(verification):

    if not verification:

        return


    st.markdown(
        "### 🔄 Back-Translation Verification"
    )


    status = verification.get(
        "status",
        "REVIEW"
    )


    if status == "PASS":

        st.success(
            "✅ Verification PASSED — "
            "No major semantic problem detected."
        )

    else:

        st.warning(
            "⚠️ Verification requires review."
        )


    back_translation = verification.get(
        "back_translation",
        ""
    )


    if back_translation:

        st.markdown(
            "**Back-Translation:**"
        )

        st.info(
            back_translation
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        value = verification.get(
            "meaning_preserved"
        )

        if value is True:
            display = "✅ Yes"
        elif value is False:
            display = "❌ No"
        else:
            display = "N/A"

        st.metric(
            "Meaning Preserved",
            display
        )


    with col2:

        value = verification.get(
            "selected_meaning_preserved"
        )

        if value is True:
            display = "✅ Yes"
        elif value is False:
            display = "❌ No"
        else:
            display = "N/A"

        st.metric(
            "Selected Meaning",
            display
        )


    with col3:

        value = verification.get(
            "numbers_preserved"
        )

        if value is True:
            display = "✅ Yes"
        elif value is False:
            display = "❌ No"
        else:
            display = "N/A"

        st.metric(
            "Numbers Preserved",
            display
        )


    reason = verification.get(
        "reason",
        ""
    )


    if reason:

        st.markdown(
            "**Verification Analysis:**"
        )

        st.info(
            reason
        )

    risk = verification.get("risk_index") or {}
    if risk:
        st.markdown("**Translation Risk Index**")
        st.metric("Risk", f"{risk.get('score', 'N/A')}/100 — {risk.get('level', 'N/A')}")
        st.caption(risk.get("interpretation", "Design-based index; not a probability."))
        for item in risk.get("reasons", []):
            st.write(f"• {item}")

    candidates = verification.get("candidate_evidence") or []
    if candidates:
        with st.expander("Candidate evidence ranking"):
            for index, item in enumerate(candidates, start=1):
                st.write(
                    f"{index}. Evidence score {item.get('evidence_score', 'N/A')} "
                    f"· Risk {item.get('risk_index', 'N/A')}/100 · "
                    f"{item.get('translation', '')}"
                )


# ============================================================
# DISPLAY EVALUATION
# ============================================================

def display_evaluation(evaluation):

    if not evaluation:

        return


    st.markdown(
        "### 📊 Translation Quality Evaluation"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        meaning = evaluation.get(
            "meaning",
            "N/A"
        )

        context_score = evaluation.get(
            "context",
            "N/A"
        )

        st.metric("Meaning", str(meaning))
        st.metric("Context", str(context_score))


    with col2:

        grammar = evaluation.get(
            "grammar_naturalness",
            "N/A"
        )

        tone = evaluation.get(
            "tone",
            "N/A"
        )

        st.metric("Grammar", str(grammar))
        st.metric("Tone", str(tone))


    with col3:

        ambiguity = evaluation.get(
            "ambiguity_resolution",
            "N/A"
        )

        overall = evaluation.get(
            "overall",
            "N/A"
        )

        st.metric("Ambiguity Resolution", str(ambiguity))
        st.metric("Overall", str(overall))


    strengths = evaluation.get(
        "strengths",
        []
    )


    if strengths:

        st.markdown(
            "**👍 Strengths**"
        )

        for item in strengths:

            st.write(
                f"• {item}"
            )


    issues = evaluation.get(
        "issues",
        []
    )


    if issues:

        st.markdown(
            "**⚠️ Issues**"
        )

        for item in issues:

            st.write(
                f"• {item}"
            )


    explanation = evaluation.get(
        "explanation",
        ""
    )


    if explanation:

        st.info(
            explanation
        )

    st.caption(evaluation.get(
        "assessment_type",
        "Evidence summary; not a calibrated human quality score."
    ))


# ============================================================
# TRANSLATION PIPELINE
# ============================================================

def process_text_translation(
    source_text,
    source_language,
    target_language,
    context,
    tone,
    selected_meaning=None,
    semantic_constraint=None,
    ambiguity_result=None,
    language_confidence=None
):

    with st.spinner("🤖 Translating and verifying the best NLLB result..."):
        ranked = translate_and_rank(
            source_text, source_language, target_language,
            verify_translation,
            clarification=selected_meaning,
            semantic_constraint=semantic_constraint,
            context=context,
            tone=tone,
            # One beam candidate avoids three forward translations plus three
            # back-translations and LLM checks on every normal request.
            candidate_count=1,
        )
        translation = ranked["translation"]
        verification = (ranked.get("selected") or {}).get("verification")
        if not verification:
            verification = run_verification(
                source_text, translation, source_language, target_language,
                selected_meaning, semantic_constraint, context, language_confidence,
            )
        if language_confidence is not None:
            from risk_engine import assess_risk
            verification["risk_index"] = assess_risk(
                verification, bool(selected_meaning), language_confidence
            )
        verification["candidate_evidence"] = [
            {"translation": item["translation"],
             "evidence_score": item["evidence_score"],
             "risk_index": item["risk"]["score"]}
            for item in ranked.get("candidates", [])
        ]


    evaluation = run_quality_evaluation(

        original_text=source_text,

        translated_text=translation,

        source_language=source_language,

        target_language=target_language,

        context=context,

        tone=tone,

        ambiguity_result=ambiguity_result,

        verification=verification,

        selected_meaning=selected_meaning
    )


    return (
        translation,
        verification,
        evaluation
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    '🌐 AI Multilingual Translator'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Context-Aware • Ambiguity-Resolved • '
    'Human-in-the-Loop • Self-Verified Translation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="badge">'
    '✦ Powered by<br>Ollama • Qwen • NLLB-200'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ Translation Settings"
)

st.sidebar.caption(
    "Configure the translation context and style."
)


LANGUAGES = [
    "English",
    "Malayalam",
    "Hindi",
    "Tamil",
    "Telugu",
    "Kannada"
]


source_language = st.sidebar.selectbox(
    "🌐 Source Language",
    LANGUAGES,
    index=0
)


target_language = st.sidebar.selectbox(
    "🎯 Target Language",
    [
        "Malayalam",
        "English",
        "Hindi",
        "Tamil",
        "Telugu",
        "Kannada"
    ],
    index=0
)


context = st.sidebar.selectbox(
    "📚 Context",
    [
        "General",
        "Academic",
        "Business",
        "Medical",
        "Technical",
        "Travel",
        "Social",
        "Conversation"
    ]
)


tone = st.sidebar.selectbox(
    "🎭 Tone",
    [
        "Neutral",
        "Formal",
        "Informal",
        "Professional",
        "Friendly",
        "Polite",
        "Technical"
    ]
)


st.sidebar.divider()


st.sidebar.markdown(
    "### 🚀 System Pipeline"
)

st.sidebar.write(
    "1. 🔎 Ambiguity Detection"
)

st.sidebar.write(
    "2. 👤 Human Clarification"
)

st.sidebar.write(
    "3. 🌐 Context-Aware Translation"
)

st.sidebar.write(
    "4. 🔄 Back-Translation Verification"
)

st.sidebar.write(
    "5. 📊 Quality Evaluation"
)

st.sidebar.write(
    "6. 🛠️ Adaptive Repair"
)

st.sidebar.write(
    "7. 🎙️ Voice Translation"
)


# ============================================================
# TABS
# ============================================================

text_tab, voice_tab = st.tabs(
    [
        "📝 Text Translation",
        "🎙️ Voice Translation"
    ]
)


# ============================================================
# TEXT TRANSLATION
# ============================================================

with text_tab:

    st.markdown(
        '<div class="glass-panel">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="step-label">STEP 01 • INPUT</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "### 📝 Enter Text"
    )


    st.caption(
        "Type or paste the sentence you want to translate."
    )


    text = st.text_area(
        "Text to translate",
        height=170,
        placeholder=(
            "Example: I saw her duck, the bird."
        ),
        max_chars=5000,
        label_visibility="collapsed"
    )


    st.caption(
        f"{len(text)} / 5000 characters"
    )


    if st.button(
        "🚀 Analyze & Translate",
        use_container_width=True,
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        elif source_language == target_language:

            st.warning(
                "Please select different source "
                "and target languages."
            )

        else:

            reset_translation_results()

            st.session_state.translated_for_text = text

            st.session_state.current_source_text = text

            st.session_state.current_source_language = (
                source_language
            )

            st.session_state.current_target_language = (
                target_language
            )

            st.session_state.current_context = context

            st.session_state.current_tone = tone


            with st.spinner(
                "🧠 Detecting ambiguity..."
            ):

                try:

                    result = detect_ambiguity(

                        text=text,

                        source_language=source_language,

                        target_language=target_language,

                        context=context
                    )


                    st.session_state.ambiguity_result = result


                except Exception as e:

                    st.error(
                        f"Ambiguity detection failed: {e}"
                    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # AMBIGUITY
    # ========================================================

    ambiguity_result = (
        st.session_state.ambiguity_result
    )


    if ambiguity_result:

        if ambiguity_result.get(
            "ambiguous",
            False
        ):

            st.markdown(
                "### 🧠 Ambiguity Detection"
            )


            st.warning(
                "⚠️ Possible ambiguity detected."
            )


            reason = ambiguity_result.get(
                "reason",
                ""
            )


            if reason:

                st.info(
                    reason
                )


            interpretations = (
                ambiguity_result.get(
                    "interpretations",
                    []
                )
            )


            constraints = (
                ambiguity_result.get(
                    "semantic_constraints",
                    []
                )
            )


            if len(interpretations) >= 2:

                if len(constraints) < len(
                    interpretations
                ):

                    constraints += (
                        [""] *
                        (
                            len(interpretations)
                            - len(constraints)
                        )
                    )


                selected = st.radio(
                    "Select the meaning you intended:",
                    interpretations
                )


                selected_index = (
                    interpretations.index(
                        selected
                    )
                )


                selected_constraint = (
                    constraints[selected_index]
                )


                st.session_state.selected_interpretation = (
                    selected
                )

                st.session_state.selected_constraint = (
                    selected_constraint
                )


                st.success(
                    f"Selected meaning: {selected}"
                )


                if selected_constraint:

                    st.caption(
                        "Semantic constraint"
                    )

                    st.code(
                        selected_constraint
                    )


                if st.button(
                    "🔄 Translate Using Selected Meaning",
                    use_container_width=True
                ):

                    try:

                        (
                            translation,
                            verification,
                            evaluation
                        ) = process_text_translation(

                            source_text=text,

                            source_language=source_language,

                            target_language=target_language,

                            context=context,

                            tone=tone,

                            selected_meaning=selected,

                            semantic_constraint=(
                                selected_constraint
                            ),

                            ambiguity_result=(
                                ambiguity_result
                            )
                        )


                        st.session_state.translation_result = (
                            translation
                        )

                        st.session_state.verification = (
                            verification
                        )

                        st.session_state.evaluation = (
                            evaluation
                        )

                        st.session_state.repair_result = None


                    except Exception as e:

                        st.error(
                            f"Translation failed: {e}"
                        )


        else:

            if (
                st.session_state.translation_result
                is None
            ):

                try:

                    (
                        translation,
                        verification,
                        evaluation
                    ) = process_text_translation(

                        source_text=text,

                        source_language=source_language,

                        target_language=target_language,

                        context=context,

                        tone=tone,

                        selected_meaning=None,

                        semantic_constraint=None,

                        ambiguity_result=(
                            ambiguity_result
                        )
                    )


                    st.session_state.translation_result = (
                        translation
                    )

                    st.session_state.verification = (
                        verification
                    )

                    st.session_state.evaluation = (
                        evaluation
                    )


                except Exception as e:

                    st.error(
                        f"Translation failed: {e}"
                    )


    # ========================================================
    # TRANSLATION RESULT
    # ========================================================

    translation = (
        st.session_state.translation_result
    )


    if translation:

        st.markdown(
            "---"
        )


        st.markdown(
            '<div class="step-label">STEP 02 • RESULT</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            "### ✨ Translation Result"
        )


        safe_translation = html.escape(
            str(translation)
        )


        st.markdown(
            textwrap.dedent(f"""
                <div class="translation-box">
                    {safe_translation}
                </div>
            """),
            unsafe_allow_html=True
        )


        st.write("")


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Source",
                source_language
            )


        with col2:

            st.metric(
                "Target",
                target_language
            )


        with col3:

            st.metric(
                "Context",
                context
            )


        with col4:

            st.metric(
                "Tone",
                tone
            )


        # ====================================================
        # VERIFICATION
        # ====================================================

        verification = (
            st.session_state.verification
        )


        if verification:

            st.markdown(
                "---"
            )

            display_verification(
                verification
            )


        # ====================================================
        # ADAPTIVE REPAIR
        # ====================================================

        if (
            verification
            and
            verification.get("status") == "REVIEW"
        ):

            st.markdown(
                "---"
            )


            st.markdown(
                "### 🛠️ Adaptive Translation Repair"
            )


            if not REPAIR_AVAILABLE:

                st.warning(
                    "repair_engine.py is not available."
                )

            else:

                st.info(
                    "Verification detected a possible "
                    "translation problem. The repair engine "
                    "can attempt one correction."
                )


                if st.session_state.repair_result is None:

                    if st.button(
                        "🛠️ Run Adaptive Repair",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Diagnosing and repairing..."
                        ):

                            repair_result = (
                                run_adaptive_repair(

                                    original_text=text,

                                    translated_text=translation,

                                    verification=verification,

                                    source_language=(
                                        source_language
                                    ),

                                    target_language=(
                                        target_language
                                    ),

                                    context=context,

                                    tone=tone,

                                    selected_meaning=(
                                        st.session_state
                                        .selected_interpretation
                                    ),

                                    semantic_constraint=(
                                        st.session_state
                                        .selected_constraint
                                    )
                                )
                            )


                        st.session_state.repair_result = (
                            repair_result
                        )

                        st.rerun()


                repair_result = (
                    st.session_state.repair_result
                )


                if repair_result:

                    problem = repair_result.get(
                        "problem",
                        ""
                    )


                    instruction = (
                        repair_result.get(
                            "repair_instruction",
                            ""
                        )
                    )


                    corrected_source = (
                        repair_result.get(
                            "corrected_source_sentence",
                            ""
                        )
                    )


                    if problem:

                        st.warning(
                            f"**Detected Problem:** {problem}"
                        )


                    if instruction:

                        st.info(
                            f"**Repair Instruction:** "
                            f"{instruction}"
                        )


                    if corrected_source:

                        st.markdown(
                            "**Corrected Source:**"
                        )

                        st.code(
                            corrected_source
                        )


                    if (
                        repair_result.get("accepted") is True
                        and (repair_result.get("repaired_verification") or {}).get("status") == "PASS"
                    ):

                        repaired_translation = (
                            repair_result.get(
                                "repaired_translation",
                                ""
                            )
                        )


                        if repaired_translation:

                            col1, col2 = st.columns(2)


                            with col1:

                                st.markdown(
                                    "**Before Repair**"
                                )

                                st.error(
                                    translation
                                )


                            with col2:

                                st.markdown(
                                    "**After Repair**"
                                )

                                st.success(
                                    repaired_translation
                                )


                            repaired_verification = repair_result.get("repaired_verification")
                            if repaired_verification:
                                st.session_state.translation_result = repaired_translation
                                st.session_state.verification = repaired_verification
                                st.session_state.evaluation = run_quality_evaluation(
                                    text, repaired_translation, source_language,
                                    target_language, context, tone,
                                    verification=repaired_verification,
                                    selected_meaning=st.session_state.selected_interpretation,
                                )
                                if repaired_verification.get("status") == "PASS":
                                    st.success("✅ Repaired translation passed verification.")
                                else:
                                    st.warning("⚠️ Improved repair candidate still requires human review.")
                            else:
                                st.warning("Repair evidence was incomplete; original translation retained.")


                    else:

                        st.warning(
                            repair_result.get(
                                "reason",
                                "Repair could not be completed; original translation retained."
                            )
                        )
                        for candidate in repair_result.get("candidate_verifications", []):
                            evidence = candidate.get("verification", {})
                            st.write(
                                f"Candidate: {candidate.get('translation', '')}"
                            )
                            st.caption(
                                f"Evidence {candidate.get('evidence_score', 'N/A')} "
                                f"vs original {repair_result.get('original_evidence_score', 'N/A')} · "
                                f"Verification {evidence.get('status', 'REVIEW')}"
                            )


        # ====================================================
        # EVALUATION
        # ====================================================

        evaluation = (
            st.session_state.evaluation
        )


        if evaluation:

            st.markdown(
                "---"
            )

            display_evaluation(
                evaluation
            )


# ============================================================
# VOICE TRANSLATION
# ============================================================

with voice_tab:

    st.markdown(
        '<div class="glass-panel">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="step-label">'
        'MULTIMODAL MODE'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "### 🎙️ Voice-to-Voice Translation"
    )


    st.caption(
        "Speak → Detect → Transcribe → Translate → "
        "Verify → Repair → Speak"
    )


    if not VOICE_AVAILABLE:

        st.error(
            "Voice engine is unavailable."
        )

        st.info(
            "Make sure voice_engine.py exists and "
            "faster-whisper + edge-tts are installed."
        )


    else:

        # ====================================================
        # RECORD
        # ====================================================

        st.markdown(
            "#### 🎤 Record Your Speech"
        )


        audio_input = st.audio_input(
            "Record your voice",
            sample_rate=16000,
            key="voice_recorder"
        )


        if audio_input:

            audio_bytes = (
                audio_input.getvalue()
            )


            st.session_state.voice_audio_bytes = (
                audio_bytes
            )


            st.audio(
                audio_bytes,
                format="audio/wav"
            )


            if st.button(
                "🧠 Detect Language & Transcribe",
                use_container_width=True
            ):

                with st.spinner(
                    "🎙️ Analyzing speech..."
                ):

                    try:

                        result = process_voice_input(
                            audio_bytes
                        )


                        if result.get(
                            "success",
                            False
                        ):

                            st.session_state.voice_result = (
                                result
                            )

                            st.session_state.voice_translation_result = (
                                None
                            )

                            st.session_state.voice_verification = (
                                None
                            )

                            st.session_state.voice_evaluation = (
                                None
                            )

                            st.session_state.voice_repair_result = (
                                None
                            )

                            st.session_state.voice_tts_path = (
                                None
                            )

                        else:

                            st.error(
                                result.get(
                                    "error",
                                    "Speech processing failed."
                                )
                            )


                    except Exception as e:

                        st.error(
                            f"Voice processing failed: {e}"
                        )


        # ====================================================
        # VOICE RESULT
        # ====================================================

        voice_result = (
            st.session_state.voice_result
        )


        if voice_result:

            st.markdown(
                "---"
            )


            st.markdown(
                "### 🌐 Automatic Language Detection"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Detected Language",
                    voice_result.get(
                        "language",
                        "Unknown"
                    )
                )


            with col2:

                confidence = voice_result.get(
                    "confidence",
                    0
                )


                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )


            with col3:

                mix = voice_result.get(
                    "language_mix",
                    {}
                )


                st.metric(
                    "Language Pattern",
                    "Mixed"
                    if mix.get("mixed")
                    else "Single"
                )


            # =================================================
            # TRANSCRIPTION
            # =================================================

            st.markdown(
                "### 📝 Speech Transcription"
            )


            transcript = st.text_area(
                "Detected speech",
                value=voice_result.get(
                    "text",
                    ""
                ),
                height=130,
                key="voice_transcript"
            )


            if transcript.strip():

                st.markdown(
                    "### 🎯 Translation Settings"
                )


                detected_language = voice_result.get(
                    "language",
                    "English"
                )


                voice_target = st.selectbox(
                    "Target Language",
                    LANGUAGES,
                    index=(
                        LANGUAGES.index("Malayalam")
                        if detected_language == "English"
                        else 0
                    ),
                    key="voice_target"
                )


                voice_context = st.selectbox(
                    "Context",
                    [
                        "General",
                        "Academic",
                        "Business",
                        "Medical",
                        "Technical",
                        "Travel",
                        "Social",
                        "Conversation"
                    ],
                    key="voice_context"
                )


                voice_tone = st.selectbox(
                    "Tone",
                    [
                        "Neutral",
                        "Formal",
                        "Informal",
                        "Professional",
                        "Friendly",
                        "Polite",
                        "Technical"
                    ],
                    key="voice_tone"
                )


                if st.button(
                    "🚀 Translate Voice",
                    use_container_width=True,
                    type="primary"
                ):

                    if detected_language == voice_target:

                        st.warning(
                            "Please select a different "
                            "target language."
                        )

                    else:

                        with st.spinner(
                            "🧠 Running translation pipeline..."
                        ):

                            try:

                                voice_ambiguity = (
                                    detect_ambiguity(

                                        text=transcript,

                                        source_language=(
                                            detected_language
                                        ),

                                        target_language=(
                                            voice_target
                                        ),

                                        context=voice_context
                                    )
                                )


                                if voice_ambiguity.get(
                                    "ambiguous",
                                    False
                                ):

                                    voice_result[
                                        "ambiguity_result"
                                    ] = voice_ambiguity

                                    st.warning(
                                        "⚠️ Ambiguity detected. "
                                        "Select the intended meaning below."
                                    )

                                else:

                                    (
                                        voice_translation,
                                        voice_verification,
                                        voice_evaluation
                                    ) = (
                                        process_text_translation(

                                            source_text=transcript,

                                            source_language=(
                                                detected_language
                                            ),

                                            target_language=(
                                                voice_target
                                            ),

                                            context=voice_context,

                                            tone=voice_tone,
                                            language_confidence=voice_result.get("confidence")
                                        )
                                    )


                                    st.session_state.voice_translation_result = (
                                        voice_translation
                                    )

                                    st.session_state.voice_verification = (
                                        voice_verification
                                    )

                                    st.session_state.voice_evaluation = (
                                        voice_evaluation
                                    )

                            except Exception as e:

                                st.error(
                                    f"Voice translation failed: {e}"
                                )


                # =================================================
                # VOICE AMBIGUITY
                # =================================================

                voice_ambiguity = (
                    voice_result.get(
                        "ambiguity_result"
                    )
                )


                if voice_ambiguity and voice_ambiguity.get(
                    "ambiguous",
                    False
                ):

                    st.markdown(
                        "### 🧠 Ambiguity Resolution"
                    )


                    interpretations = (
                        voice_ambiguity.get(
                            "interpretations",
                            []
                        )
                    )


                    constraints = (
                        voice_ambiguity.get(
                            "semantic_constraints",
                            []
                        )
                    )


                    if len(interpretations) >= 2:

                        while len(constraints) < len(
                            interpretations
                        ):

                            constraints.append("")


                        selected_voice_meaning = st.radio(
                            "Select intended meaning",
                            interpretations,
                            key="voice_meaning"
                        )


                        selected_index = (
                            interpretations.index(
                                selected_voice_meaning
                            )
                        )


                        selected_voice_constraint = (
                            constraints[selected_index]
                        )


                        if st.button(
                            "🔄 Translate Selected Meaning",
                            use_container_width=True
                        ):

                            voice_result["selected_meaning"] = selected_voice_meaning
                            voice_result["selected_constraint"] = selected_voice_constraint

                            try:

                                (
                                    voice_translation,
                                    voice_verification,
                                    voice_evaluation
                                ) = (
                                    process_text_translation(

                                        source_text=transcript,

                                        source_language=(
                                            detected_language
                                        ),

                                        target_language=(
                                            voice_target
                                        ),

                                        context=voice_context,

                                        tone=voice_tone,

                                        selected_meaning=(
                                            selected_voice_meaning
                                        ),

                                        semantic_constraint=(
                                            selected_voice_constraint
                                        ),

                                        ambiguity_result=(
                                            voice_ambiguity
                                        ),
                                        language_confidence=voice_result.get("confidence")
                                    )
                                )


                                st.session_state.voice_translation_result = (
                                    voice_translation
                                )

                                st.session_state.voice_verification = (
                                    voice_verification
                                )

                                st.session_state.voice_evaluation = (
                                    voice_evaluation
                                )


                            except Exception as e:

                                st.error(
                                    f"Translation failed: {e}"
                                )


                # =================================================
                # FINAL VOICE TRANSLATION
                # =================================================

                voice_translation = (
                    st.session_state.voice_translation_result
                )


                if voice_translation:

                    st.markdown(
                        "---"
                    )


                    st.markdown(
                        "### ✨ Translated Text"
                    )


                    safe_voice_translation = (
                        html.escape(
                            str(voice_translation)
                        )
                    )


                    st.markdown(
                        textwrap.dedent(f"""
                            <div class="translation-box">
                                {safe_voice_translation}
                            </div>
                        """),
                        unsafe_allow_html=True
                    )


                    # =============================================
                    # VERIFICATION
                    # =============================================

                    voice_verification = (
                        st.session_state.voice_verification
                    )


                    if voice_verification:

                        st.markdown(
                            "---"
                        )

                        display_verification(
                            voice_verification
                        )


                    # =============================================
                    # REPAIR
                    # =============================================

                    if (
                        voice_verification
                        and
                        voice_verification.get(
                            "status"
                        ) == "REVIEW"
                    ):

                        st.markdown(
                            "### 🛠️ Adaptive Repair"
                        )


                        if st.session_state.voice_repair_result is None:

                            if REPAIR_AVAILABLE:

                                if st.button(
                                    "🛠️ Repair Voice Translation",
                                    use_container_width=True
                                ):

                                    with st.spinner(
                                        "Repairing translation..."
                                    ):

                                        repair_result = (
                                            run_adaptive_repair(

                                                original_text=(
                                                    transcript
                                                ),

                                                translated_text=(
                                                    voice_translation
                                                ),

                                                verification=(
                                                    voice_verification
                                                ),

                                                source_language=(
                                                    detected_language
                                                ),

                                                target_language=(
                                                    voice_target
                                                ),

                                                context=(
                                                    voice_context
                                                ),

                                                tone=(
                                                    voice_tone
                                                )
                                            )
                                        )


                                    st.session_state.voice_repair_result = (
                                        repair_result
                                    )

                                    st.rerun()


                        voice_repair = (
                            st.session_state.voice_repair_result
                        )


                        if voice_repair:

                            if (
                                voice_repair.get("accepted") is True
                                and (voice_repair.get("repaired_verification") or {}).get("status") == "PASS"
                            ):

                                repaired = (
                                    voice_repair.get(
                                        "repaired_translation",
                                        ""
                                    )
                                )


                                if repaired:

                                    st.markdown(
                                        "**Before Repair**"
                                    )

                                    st.error(
                                        voice_translation
                                    )


                                    st.markdown(
                                        "**After Repair**"
                                    )

                                    st.success(
                                        repaired
                                    )


                                    repaired_verification = voice_repair.get("repaired_verification")
                                    if repaired_verification:
                                        st.session_state.voice_translation_result = repaired
                                        st.session_state.voice_verification = repaired_verification
                                        st.session_state.voice_evaluation = run_quality_evaluation(
                                            transcript, repaired, detected_language,
                                            voice_target, voice_context, voice_tone,
                                            verification=repaired_verification,
                                            selected_meaning=voice_result.get("selected_meaning"),
                                        )
                                    else:
                                        st.warning("Repair evidence was incomplete; original translation retained.")

                            else:

                                st.warning(
                                    voice_repair.get(
                                        "reason",
                                        "Repair unsuccessful."
                                    )
                                )
                                for candidate in voice_repair.get("candidate_verifications", []):
                                    st.write(f"Candidate: {candidate.get('translation', '')}")
                                    st.caption(
                                        f"Evidence {candidate.get('evidence_score', 'N/A')} "
                                        f"vs original {voice_repair.get('original_evidence_score', 'N/A')}"
                                    )


                    # =============================================
                    # EVALUATION
                    # =============================================

                    voice_evaluation = (
                        st.session_state.voice_evaluation
                    )


                    if voice_evaluation:

                        st.markdown(
                            "---"
                        )

                        display_evaluation(
                            voice_evaluation
                        )


                    # =============================================
                    # TEXT TO SPEECH
                    # =============================================

                    st.markdown(
                        "---"
                    )


                    st.markdown(
                        "### 🔊 Text-to-Speech"
                    )


                    if st.button(
                        "🔊 Generate Translated Speech",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Generating translated audio..."
                        ):

                            try:

                                tts_result = (
                                    text_to_speech(

                                        voice_translation,

                                        voice_target
                                    )
                                )


                                if tts_result.get(
                                    "success",
                                    False
                                ):

                                    st.session_state.voice_tts_path = (
                                        tts_result.get(
                                            "audio_path"
                                        )
                                    )

                                else:

                                    st.error(
                                        tts_result.get(
                                            "error",
                                            "TTS failed."
                                        )
                                    )


                            except Exception as e:

                                st.error(
                                    f"TTS failed: {e}"
                                )


                    # =============================================
                    # AUDIO
                    # =============================================

                    tts_path = (
                        st.session_state.voice_tts_path
                    )


                    if tts_path:

                        try:

                            with open(
                                tts_path,
                                "rb"
                            ) as audio_file:

                                audio_data = (
                                    audio_file.read()
                                )


                            st.markdown(
                                "### 🎧 Final Voice Output"
                            )


                            st.audio(
                                audio_data,
                                format="audio/mp3"
                            )


                            st.success(
                                "🎧 Voice-to-voice translation completed."
                            )


                        except Exception as e:

                            st.error(
                                f"Could not load audio: {e}"
                            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# FEATURE PIPELINE
# ============================================================

st.markdown(
    "---"
)


st.markdown(
    "### 🚀 Intelligent Translation Pipeline"
)


pipeline_cols = st.columns(7)


pipeline_items = [

    ("🔎", "Detect"),

    ("👤", "Clarify"),

    ("🌐", "Translate"),

    ("🔄", "Verify"),

    ("📊", "Evaluate"),

    ("🛠️", "Repair"),

    ("🔊", "Speak")
]


for column, item in zip(
    pipeline_cols,
    pipeline_items
):

    with column:

        card_html = textwrap.dedent(f"""
            <div style="
                text-align:center;
                padding:14px 5px;
                border-radius:14px;
                background:rgba(18,25,55,0.55);
                border:1px solid rgba(120,140,255,0.15);
                transition:transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            ">
                <div style="font-size:25px; margin-bottom:5px;">{item[0]}</div>
                <div style="color:#aeb8dd; font-size:12px;">{item[1]}</div>
            </div>
        """)

        st.markdown(
            card_html,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)


st.markdown(
    """
    <div style="
        text-align:center;
        color:#69749e;
        font-size:12px;
        padding:20px;
    ">
        AI Multilingual Translator
        <br>
        Context-Aware • Ambiguity-Resolved •
        Human-in-the-Loop • Self-Verified •
        Adaptive Repair • Multimodal Voice Translation
    </div>
    """,
    unsafe_allow_html=True
)
