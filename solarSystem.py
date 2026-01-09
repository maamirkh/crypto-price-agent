# streamlit_solar_system_push_to_talk.py

import os
import asyncio
import threading
import tempfile
import time
import hashlib
import base64
import traceback
from io import BytesIO
from dotenv import load_dotenv

import streamlit as st
import streamlit.components.v1 as components
import httpx
from gtts import gTTS
import speech_recognition as sr
from langdetect import detect

# audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except Exception:
    AUDIO_RECORDER_AVAILABLE = False

# agents sdk
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# ---------- ENV ----------
load_dotenv()
def get_gemini_key():
    return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

GEMINI_API_KEY = get_gemini_key()
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY missing")
    st.stop()

# ---------- Gemini Setup ----------
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(model=model, tracing_disabled=True)

# ---------- Wikipedia Tool ----------
@function_tool
async def wikipedia_search(query: str) -> str:
    """
    Fetch summary from Wikipedia for solar system related queries.
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "%20")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        if resp.status_code != 200:
            return "Wikipedia se data nahi mil saka."
        data = resp.json()
        return data.get("extract", "No summary available.")

# ---------- Agent ----------
agent = Agent(
    name="Solar System Education Agent",
    instructions="""
You are an educational assistant specialized in the Solar System.
You must answer questions about planets, stars, moons, space, orbits, gravity, and astronomy.
If factual knowledge is required, use the wikipedia_search tool.
Explain concepts in simple language suitable for students.
Always answer in the same language as the user input (English or Urdu/Roman Urdu).
""",
    model=model,
    tools=[wikipedia_search],
)

# ---------- Utilities ----------
_SESSION_LOCK = threading.Lock()

def append_history(role, text):
    with _SESSION_LOCK:
        st.session_state.setdefault("history", []).append((role, text))

def run_agent_sync(user_input: str) -> str:
    async def _call():
        result = await Runner.run(agent, user_input, run_config=config)
        for item in reversed(result.new_items):
            try:
                return item.raw_item.content[0].text
            except Exception:
                pass
        return "No response."

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_call())
    finally:
        loop.close()

# ---------- Language Detection ----------
def detect_user_language(text: str) -> str:
    """
    Detects English or Urdu (including Roman Urdu)
    """
    try:
        lang = detect(text)
        if lang == "en":
            # Roman Urdu detection (simple heuristic)
            urdu_keywords = ["hai", "ka", "ki", "ke", "mera", "hum", "tum", "yeh", "kya"]
            if any(word in text.lower() for word in urdu_keywords):
                return "ur"
            return "en"
        elif lang == "ur":
            return "ur"
    except Exception:
        pass
    return "en"

# ---------- TTS Formatter ----------
def format_for_tts(text: str) -> str:
    """
    Convert bullets to "First..., Second..." and add pauses for headings
    """
    lines = text.split("\n")
    tts_lines = []
    bullet_counter = 1
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("*"):
            tts_lines.append(f"{bullet_counter}. {stripped[1:].strip()}")
            bullet_counter += 1
        elif stripped.endswith(":"):
            # Pause after headings
            tts_lines.append(stripped)
            tts_lines.append("...")  # small pause
        else:
            tts_lines.append(stripped)
    return " ".join(tts_lines)

# ---------- TTS ----------
def speak(text, user_text):
    lang_code = detect_user_language(user_text)
    tts_text = format_for_tts(text)
    try:
        tts = gTTS(text=tts_text, lang=lang_code)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = buf.read()
        b64 = base64.b64encode(audio).decode()
        html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        components.html(html, height=80)
        st.audio(audio, format="audio/mp3")
    except Exception:
        traceback.print_exc()

# ---------- UI ----------
st.set_page_config(page_title="Solar System Voice Agent", layout="centered")
st.title("🌌 Solar System Voice Agent")

if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = False

if not st.session_state.audio_enabled:
    if st.button("Enable Audio"):
        st.session_state.audio_enabled = True
        st.rerun()
    st.stop()

# ---------- Text Input ----------
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Ask about planets, stars, space...")
    submit = st.form_submit_button("Send")

if submit and user_input:
    append_history("You", user_input)
    with st.spinner("Thinking..."):
        response = run_agent_sync(user_input)
    append_history("Agent", response)
    speak(response, user_input)

# ---------- Voice Input ----------
st.markdown("## 🎙️ Voice Input")

if AUDIO_RECORDER_AVAILABLE:
    audio_bytes = audio_recorder("Record")
    if audio_bytes:
        h = hashlib.sha256(audio_bytes).hexdigest()
        if st.session_state.get("last_audio") != h:
            st.session_state.last_audio = h
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(audio_bytes)
            tmp.close()

            r = sr.Recognizer()
            with sr.AudioFile(tmp.name) as src:
                audio = r.record(src)
            text = r.recognize_google(audio)
            append_history("You", text)
            with st.spinner("Thinking..."):
                response = run_agent_sync(text)
            append_history("Agent", response)
            speak(response, text)

# ---------- Chat History ----------
st.markdown("### Conversation")
for role, msg in st.session_state.get("history", []):
    st.markdown(f"**{role}:** {msg}")


# # streamlit_solar_system_push_to_talk_tts.py

# import os
# import asyncio
# import threading
# import tempfile
# import time
# import hashlib
# import base64
# import traceback
# from io import BytesIO
# from dotenv import load_dotenv

# import streamlit as st
# import streamlit.components.v1 as components
# import httpx
# from gtts import gTTS
# import speech_recognition as sr

# # audio recorder
# try:
#     from audio_recorder_streamlit import audio_recorder
#     AUDIO_RECORDER_AVAILABLE = True
# except Exception:
#     AUDIO_RECORDER_AVAILABLE = False

# # agents sdk
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
# from agents.run import RunConfig

# # ---------- ENV ----------
# load_dotenv()

# def get_gemini_key():
#     return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# GEMINI_API_KEY = get_gemini_key()
# if not GEMINI_API_KEY:
#     st.error("GEMINI_API_KEY missing")
#     st.stop()

# # ---------- Gemini Setup ----------
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",
#     openai_client=external_client
# )

# config = RunConfig(model=model, tracing_disabled=True)

# # ---------- Wikipedia Tool ----------
# @function_tool
# async def wikipedia_search(query: str) -> str:
#     """
#     Fetch summary from Wikipedia for solar system related queries.
#     """
#     url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "%20")
#     async with httpx.AsyncClient() as client:
#         resp = await client.get(url, timeout=10)
#         if resp.status_code != 200:
#             return "Wikipedia se data nahi mil saka."
#         data = resp.json()
#         return data.get("extract", "No summary available.")

# # ---------- Agent ----------
# agent = Agent(
#     name="Solar System Education Agent",
#     instructions="""
# You are an educational assistant specialized in the Solar System.
# You must answer questions about planets, stars, moons, space, orbits, gravity, and astronomy.
# If factual knowledge is required, use the wikipedia_search tool.
# Explain concepts in simple language suitable for students.
# """,
#     model=model,
#     tools=[wikipedia_search],
# )

# # ---------- TTS Formatter ----------
# ORDINALS = [
#     "First", "Second", "Third", "Fourth", "Fifth",
#     "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"
# ]

# def format_for_tts(text: str) -> str:
#     """
#     Convert markdown/headings/bullets into TTS-friendly format
#     - Headings: pause after
#     - Bullets: First, Second, ...
#     """
#     lines = text.splitlines()
#     cleaned_lines = []
#     bullet_count = 0

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue

#         # Headings (bold or ending colon)
#         heading_match = None
#         if line.startswith("**") and "**" in line[2:]:
#             heading_match = line[2:].split("**", 1)[0]
#         elif line.endswith(":"):
#             heading_match = line[:-1]

#         if heading_match:
#             cleaned_lines.append(f"{heading_match}.")  # pause with dot
#             cleaned_lines.append(" ")  # extra pause
#             bullet_count = 0
#             continue

#         # Bullets
#         if line.startswith(("-", "*", "•")):
#             content = line[1:].strip()
#             if bullet_count < len(ORDINALS):
#                 prefix = ORDINALS[bullet_count]
#             else:
#                 prefix = f"Point {bullet_count+1}"
#             cleaned_lines.append(f"{prefix}: {content}.")
#             bullet_count += 1
#             continue

#         # Normal text, remove markdown
#         line = line.replace("*", "")
#         line = line.replace("`", "")
#         cleaned_lines.append(line)

#     final_text = " ".join(cleaned_lines)
#     final_text = final_text.replace("\n", " ").replace("  ", " ")
#     return final_text.strip()

# # ---------- Utilities ----------
# _SESSION_LOCK = threading.Lock()

# def append_history(role, text):
#     with _SESSION_LOCK:
#         st.session_state.setdefault("history", []).append((role, text))

# def run_agent_sync(user_input: str) -> str:
#     async def _call():
#         result = await Runner.run(agent, user_input, run_config=config)
#         for item in reversed(result.new_items):
#             try:
#                 return item.raw_item.content[0].text
#             except Exception:
#                 pass
#         return "No response."

#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         return loop.run_until_complete(_call())
#     finally:
#         loop.close()

# def speak(text):
#     try:
#         tts_text = format_for_tts(text)
#         tts = gTTS(text=tts_text, lang="en")
#         buf = BytesIO()
#         tts.write_to_fp(buf)
#         buf.seek(0)
#         audio = buf.read()
#         b64 = base64.b64encode(audio).decode()
#         html = f"""
#         <audio controls autoplay>
#             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#         </audio>
#         """
#         components.html(html, height=80)
#         st.audio(audio, format="audio/mp3")
#     except Exception:
#         traceback.print_exc()

# # ---------- UI ----------
# st.set_page_config(page_title="Solar System Voice Agent", layout="centered")
# st.title("🌌 Solar System Voice Agent")

# if "audio_enabled" not in st.session_state:
#     st.session_state.audio_enabled = False

# if not st.session_state.audio_enabled:
#     if st.button("Enable Audio"):
#         st.session_state.audio_enabled = True
#         st.rerun()
#     st.stop()

# # ---------- Text Input ----------
# with st.form("input_form", clear_on_submit=True):
#     user_input = st.text_input("Ask about planets, stars, space...")
#     submit = st.form_submit_button("Send")

# if submit and user_input:
#     append_history("You", user_input)
#     with st.spinner("Thinking..."):
#         response = run_agent_sync(user_input)
#     append_history("Agent", response)
#     speak(response)

# # ---------- Voice Input ----------
# st.markdown("## 🎙️ Voice Input")

# if AUDIO_RECORDER_AVAILABLE:
#     audio_bytes = audio_recorder("Record")
#     if audio_bytes:
#         h = hashlib.sha256(audio_bytes).hexdigest()
#         if st.session_state.get("last_audio") != h:
#             st.session_state.last_audio = h
#             tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#             tmp.write(audio_bytes)
#             tmp.close()

#             r = sr.Recognizer()
#             with sr.AudioFile(tmp.name) as src:
#                 audio = r.record(src)
#             text = r.recognize_google(audio)

#             append_history("You", text)
#             with st.spinner("Thinking..."):
#                 response = run_agent_sync(text)
#             append_history("Agent", response)
#             speak(response)

# # ---------- Chat History ----------
# st.markdown("### Conversation")
# for role, msg in st.session_state.get("history", []):
#     st.markdown(f"**{role}:** {msg}")

