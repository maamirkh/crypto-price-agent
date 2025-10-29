# streamlit_crypto_push_to_talk.py
"""
Updated Streamlit app:
- Push-to-talk (browser recorder) + text input
- Enter key submit (st.form) and Send button (right side)
- Input clears automatically after submit
- Agent response text shows immediately and TTS plays with slower rate
- audio_recorder_streamlit provides Record/Stop UI (if installed)
"""

import os
import asyncio
import threading
import tempfile
import time
from dotenv import load_dotenv

import httpx
import streamlit as st
import hashlib
from io import BytesIO
from gtts import gTTS
import base64
import traceback
import streamlit.components.v1 as components

# audio recorder component
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except Exception:
    AUDIO_RECORDER_AVAILABLE = False

# speech recognition
import speech_recognition as sr

# agents package
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# ---------- load env / secrets ----------
load_dotenv()

def get_gemini_key():
    k = os.getenv("GEMINI_API_KEY")
    if k:
        return k
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return None

GEMINI_API_KEY = get_gemini_key()
if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY nahin mila. Add karein:\n"
        "1) environment variable GEMINI_API_KEY (ya .env), ya\n"
        "2) .streamlit/secrets.toml (GEMINI_API_KEY = \"...\")"
    )
    st.stop()

# ---------- Setup Gemini client & model ----------
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# ---------- Tools ----------
@function_tool
async def get_crypto_price(symbols: str) -> str:
    url_base = "https://api.binance.com/api/v3/ticker/price"
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    results = []
    async with httpx.AsyncClient() as client:
        for symbol in symbol_list:
            try:
                url = f"{url_base}?symbol={symbol}"
                resp = await client.get(url, timeout=10.0)
                data = resp.json()
                if resp.status_code == 200 and "price" in data:
                    price = float(data["price"])
                    results.append(f"{symbol}: ${price:,.8f}")
                else:
                    results.append(f"{symbol}: ❌ Invalid or not found")
            except Exception as e:
                results.append(f"{symbol}: ❌ Error - {str(e)}")
    return "Prices:\n" + "\n".join(results)

@function_tool
async def list_all_binance_symbols() -> str:
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()
        symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
        total = len(symbols)
        sample = ", ".join(symbols[:20])
        return f"Total trading pairs: {total}\nSample: {sample}, ..."
    except Exception as e:
        return f"Error fetching symbols: {str(e)}"

# ---------- Agent ----------
agent = Agent(
    name="Crypto Smart Agent",
    instructions=(
        "You are a helpful assistant that can fetch crypto prices from Binance. "
        "You can fetch one or multiple symbols, and list all trading pairs."
        "You can reply in any user language"
    ),
    model=model,
    tools=[get_crypto_price, list_all_binance_symbols],
)

# ---------- Utilities ----------
_SESSION_LOCK = threading.Lock()

def append_history(role: str, text: str):
    with _SESSION_LOCK:
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append((role, text))

def run_agent_sync(user_input: str) -> str:
    async def _call():
        result = await Runner.run(agent, user_input, run_config=config)
        for item in reversed(result.new_items):
            try:
                return item.raw_item.content[0].text
            except Exception:
                continue
        return "Sorry, no response."

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(_call())
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
    return response

def speak_text_background(text: str, delay: float = 0.12, lang: str = "en"):
    """
    Robust gTTS playback for Streamlit:
    - Generates MP3 in-memory
    - Tries to embed via HTML <audio> with base64 src (gives play controls + often works around autoplay)
    - Falls back to st.audio if needed
    Notes: many browsers block autoplay until user interacts with page.
    """
    try:
        # small UI-friendly delay
        time.sleep(delay)
        # generate mp3 into BytesIO
        tts = gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.read()

        # base64 encode for HTML embedding
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        # Use components.html so browser receives raw HTML (shows controls)
        try:
            components.html(html, height=80)
            # also show fallback st.audio (useful if components blocked)
            try:
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                print("st.audio fallback error:", e)
        except Exception as e_html:
            print("components.html error:", e_html)
            # fallback to st.audio only
            try:
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e2:
                print("st.audio error after components failed:", e2)
        finally:
            try:
                buf.close()
            except Exception:
                pass

    except Exception as outer_e:
        # Print full traceback to terminal so you can paste it here if still failing
        print("TTS generation/playback failed:", outer_e)
        traceback.print_exc()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Crypto Agent (Push-to-talk)", layout="centered")
st.title("Crypto Agent — Push-to-talk + Text")

# 🔊 Enable audio button (Streamlit Cloud friendly)
# Place this here (right after the title) so first user interaction enables browser audio autoplay.
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = False

if not st.session_state.audio_enabled:
    if st.button("Enable audio / Start demo"):
        st.session_state.audio_enabled = True
        st.success("Audio enabled! You can now interact with voice.")
        # force a rerun so the rest of the app shows immediately after click
        st.rerun()

    st.info("Click 'Enable audio / Start demo' to activate voice features (required for browsers).")
    st.stop()
st.write("Always use pairs, for example btcusdt, ethusdt, dogeusdt etc...")

if "history" not in st.session_state:
    st.session_state.history = []

# Input form: Enter will submit, and clear_on_submit ensures it clears
with st.form(key="input_form", clear_on_submit=True):
    col_in, col_btn = st.columns([8, 1])
    with col_in:
        user_input = st.text_input("Type message (press Enter or Send):", key="user_input_form")
    with col_btn:
        submit = st.form_submit_button("Send")

    # When submitted handle synchronously (blocking with spinner)
    if submit and user_input and user_input.strip():
        val = user_input.strip()
        append_history("You", val)
        # run agent and show spinner so response text appears immediately
        with st.spinner("Agent is thinking..."):
            try:
                response = run_agent_sync(val)
            except Exception as e:
                response = f"Agent error: {e}"
        append_history("Agent", response)
        # speak slowly (previously slow_factor used) — keep call minimal
        speak_text_background(response, delay=0.12)
        # NOTE: removed st.rerun() here to avoid interrupting audio playback

# Voice area below
st.markdown("## Voice (push-to-talk)")
st.write("Record, then Stop. (Browser recorder)")

if not AUDIO_RECORDER_AVAILABLE:
    st.warning("Install `audio-recorder-streamlit` for browser Record/Stop UI. Fallback: upload WAV.")
    upload = st.file_uploader("WAV upload (fallback)", type=["wav"])
    if upload:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpf.write(upload.getbuffer())
        tmpf.close()
        r = sr.Recognizer()
        try:
            with sr.AudioFile(tmpf.name) as source:
                audio = r.record(source)
            recognized = r.recognize_google(audio)
            append_history("You", recognized)
            with st.spinner("Agent is thinking..."):
                response = run_agent_sync(recognized)
            append_history("Agent", response)
            speak_text_background(response, delay=0.12)
            # NOTE: removed st.rerun() here to avoid interrupting audio playback
        except Exception as e:
            st.error("Voice recognition failed: " + str(e))
else:
    audio_bytes = audio_recorder(text="Record (click to start/stop)", recording_color="#ffcc00")

if audio_bytes:
    # compute a simple hash to detect repeated same audio
    current_hash = hashlib.sha256(audio_bytes).hexdigest()

    # if we have already processed this exact audio, skip processing
    if st.session_state.get("last_audio_hash") == current_hash:
        # do nothing (prevents duplicate replies)
        st.info("Audio already processed — record again to get a new response.")
    else:
        # mark as processed
        st.session_state["last_audio_hash"] = current_hash

        # save to temp file and transcribe
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpf.write(audio_bytes)
        tmpf.close()

        r = sr.Recognizer()
        try:
            with sr.AudioFile(tmpf.name) as source:
                audio = r.record(source)
            recognized = r.recognize_google(audio)

            # append user text and synchronously call agent
            append_history("You", recognized)
            with st.spinner("Agent is thinking..."):
                response = run_agent_sync(recognized)
            append_history("Agent", response)

            # speak in background
            speak_text_background(response, delay=0.12)

            # NOTE: removed st.rerun() here to avoid interrupting audio playback

        except Exception as e:
            st.error("Voice recognition failed: " + str(e))

# Conversation display
st.write("### Conversation")
for who, txt in st.session_state.history:
    if who == "You":
        st.markdown(f"**You:** {txt}")
    else:
        st.markdown(f"**Agent:** {txt}")

st.write("---")
st.markdown("Tip: Type message and press Enter or Send. For voice, use the browser recorder (Record then Stop).")




# # streamlit_crypto_push_to_talk.py
# """
# Updated Streamlit app:
# - Push-to-talk (browser recorder) + text input
# - Enter key submit (st.form) and Send button (right side)
# - Input clears automatically after submit
# - Agent response text shows immediately and TTS plays with slower rate
# - audio_recorder_streamlit provides Record/Stop UI (if installed)
# """

# import os
# import asyncio
# import threading
# import tempfile
# import time
# from dotenv import load_dotenv

# import httpx
# import streamlit as st
# import hashlib

# # audio recorder component
# try:
#     from audio_recorder_streamlit import audio_recorder
#     AUDIO_RECORDER_AVAILABLE = True
# except Exception:
#     AUDIO_RECORDER_AVAILABLE = False

# # speech recognition and tts
# import speech_recognition as sr
# import pyttsx3

# # agents package
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
# from agents.run import RunConfig

# # ---------- load env / secrets ----------
# load_dotenv()

# def get_gemini_key():
#     k = os.getenv("GEMINI_API_KEY")
#     if k:
#         return k
#     try:
#         if "GEMINI_API_KEY" in st.secrets:
#             return st.secrets["GEMINI_API_KEY"]
#     except Exception:
#         pass
#     return None

# GEMINI_API_KEY = get_gemini_key()
# if not GEMINI_API_KEY:
#     st.error(
#         "GEMINI_API_KEY nahin mila. Add karein:\n"
#         "1) environment variable GEMINI_API_KEY (ya .env), ya\n"
#         "2) .streamlit/secrets.toml (GEMINI_API_KEY = \"...\")"
#     )
#     st.stop()

# # ---------- Setup Gemini client & model ----------
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# # ---------- Tools ----------
# @function_tool
# async def get_crypto_price(symbols: str) -> str:
#     url_base = "https://api.binance.com/api/v3/ticker/price"
#     symbol_list = [s.strip().upper() for s in symbols.split(",")]
#     results = []
#     async with httpx.AsyncClient() as client:
#         for symbol in symbol_list:
#             try:
#                 url = f"{url_base}?symbol={symbol}"
#                 resp = await client.get(url, timeout=10.0)
#                 data = resp.json()
#                 if resp.status_code == 200 and "price" in data:
#                     price = float(data["price"])
#                     results.append(f"{symbol}: ${price:,.8f}")
#                 else:
#                     results.append(f"{symbol}: ❌ Invalid or not found")
#             except Exception as e:
#                 results.append(f"{symbol}: ❌ Error - {str(e)}")
#     return "Prices:\n" + "\n".join(results)

# @function_tool
# async def list_all_binance_symbols() -> str:
#     url = "https://api.binance.com/api/v3/exchangeInfo"
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url, timeout=10.0)
#             data = response.json()
#         symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
#         total = len(symbols)
#         sample = ", ".join(symbols[:20])
#         return f"Total trading pairs: {total}\nSample: {sample}, ..."
#     except Exception as e:
#         return f"Error fetching symbols: {str(e)}"

# # ---------- Agent ----------
# agent = Agent(
#     name="Crypto Smart Agent",
#     instructions=(
#         "You are a helpful assistant that can fetch crypto prices from Binance. "
#         "You can fetch one or multiple symbols, and list all trading pairs."
#         "You can reply in any user language"
#     ),
#     model=model,
#     tools=[get_crypto_price, list_all_binance_symbols],
# )

# # ---------- Utilities ----------
# _SESSION_LOCK = threading.Lock()

# def append_history(role: str, text: str):
#     with _SESSION_LOCK:
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         st.session_state.history.append((role, text))

# def run_agent_sync(user_input: str) -> str:
#     async def _call():
#         result = await Runner.run(agent, user_input, run_config=config)
#         for item in reversed(result.new_items):
#             try:
#                 return item.raw_item.content[0].text
#             except Exception:
#                 continue
#         return "Sorry, no response."

#     loop = asyncio.new_event_loop()
#     try:
#         asyncio.set_event_loop(loop)
#         response = loop.run_until_complete(_call())
#     finally:
#         try:
#             loop.run_until_complete(loop.shutdown_asyncgens())
#         except Exception:
#             pass
#         loop.close()
#     return response

# def speak_text_background(text: str, delay: float = 0.12, slow_factor: float = 0.8):
#     """
#     Background TTS. slow_factor <1 slows voice (0.8 = 80% speed).
#     """
#     def _worker(t, d, sf):
#         try:
#             time.sleep(d)
#             engine = pyttsx3.init()
#             try:
#                 rate = engine.getProperty("rate")
#                 engine.setProperty("rate", int(rate * sf))
#             except Exception:
#                 pass
#             engine.say(t)
#             engine.runAndWait()
#             try:
#                 engine.stop()
#             except Exception:
#                 pass
#         except Exception as e:
#             print("TTS error:", e)
#     th = threading.Thread(target=_worker, args=(text, delay, slow_factor), daemon=True)
#     th.start()

# # ---------- Streamlit UI ----------
# st.set_page_config(page_title="Crypto Agent (Push-to-talk)", layout="centered")
# st.title("Crypto Agent — Push-to-talk + Text")
# st.write("Always use pairs, for example btcusdt, ethusdt, dogeusdt etc...")

# if "history" not in st.session_state:
#     st.session_state.history = []

# # Input form: Enter will submit, and clear_on_submit ensures it clears
# with st.form(key="input_form", clear_on_submit=True):
#     col_in, col_btn = st.columns([8, 1])
#     with col_in:
#         user_input = st.text_input("Type message (press Enter or Send):", key="user_input_form")
#     with col_btn:
#         submit = st.form_submit_button("Send")

#     # When submitted handle synchronously (blocking with spinner)
#     if submit and user_input and user_input.strip():
#         val = user_input.strip()
#         append_history("You", val)
#         # run agent and show spinner so response text appears immediately
#         with st.spinner("Agent is thinking..."):
#             try:
#                 response = run_agent_sync(val)
#             except Exception as e:
#                 response = f"Agent error: {e}"
#         append_history("Agent", response)
#         # speak slowly (slow_factor=0.8)
#         speak_text_background(response, delay=0.12, slow_factor=0.7)
#         # rerun to show updated history (form will clear automatically)
#         try:
#             st.rerun()
#         except Exception:
#             pass

# # Voice area below
# st.markdown("## Voice (push-to-talk)")
# st.write("Record, then Stop. (Browser recorder)")

# if not AUDIO_RECORDER_AVAILABLE:
#     st.warning("Install `audio-recorder-streamlit` for browser Record/Stop UI. Fallback: upload WAV.")
#     upload = st.file_uploader("WAV upload (fallback)", type=["wav"])
#     if upload:
#         tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         tmpf.write(upload.getbuffer())
#         tmpf.close()
#         r = sr.Recognizer()
#         try:
#             with sr.AudioFile(tmpf.name) as source:
#                 audio = r.record(source)
#             recognized = r.recognize_google(audio)
#             append_history("You", recognized)
#             with st.spinner("Agent is thinking..."):
#                 response = run_agent_sync(recognized)
#             append_history("Agent", response)
#             speak_text_background(response, delay=0.12, slow_factor=0.8)
#             st.rerun()
#         except Exception as e:
#             st.error("Voice recognition failed: " + str(e))
# else:
#     audio_bytes = audio_recorder(text="Record (click to start/stop)", recording_color="#ffcc00")

# if audio_bytes:
#     # compute a simple hash to detect repeated same audio
#     current_hash = hashlib.sha256(audio_bytes).hexdigest()

#     # if we have already processed this exact audio, skip processing
#     if st.session_state.get("last_audio_hash") == current_hash:
#         # do nothing (prevents duplicate replies)
#         st.info("Audio already processed — record again to get a new response.")
#     else:
#         # mark as processed
#         st.session_state["last_audio_hash"] = current_hash

#         # save to temp file and transcribe
#         tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         tmpf.write(audio_bytes)
#         tmpf.close()

#         r = sr.Recognizer()
#         try:
#             with sr.AudioFile(tmpf.name) as source:
#                 audio = r.record(source)
#             recognized = r.recognize_google(audio)

#             # append user text and synchronously call agent
#             append_history("You", recognized)
#             with st.spinner("Agent is thinking..."):
#                 response = run_agent_sync(recognized)
#             append_history("Agent", response)

#             # speak in background
#             speak_text_background(response, delay=0.12, slow_factor=0.8)

#             # rerun to update UI
#             try:
#                 st.rerun()
#             except Exception:
#                 pass

#         except Exception as e:
#             st.error("Voice recognition failed: " + str(e))

# # Conversation display
# st.write("### Conversation")
# for who, txt in st.session_state.history:
#     if who == "You":
#         st.markdown(f"**You:** {txt}")
#     else:
#         st.markdown(f"**Agent:** {txt}")

# st.write("---")
# st.markdown("Tip: Type message and press Enter or Send. For voice, use the browser recorder (Record then Stop).")




# import os
# import datetime
# import asyncio
# from dotenv import load_dotenv
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
# from agents.run import RunConfig
# import httpx

# import speech_recognition as sr
# import pyttsx3
# import threading
# import time

# # Load environment variables
# load_dotenv()

# # Load Gemini API key
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is missing in .env file.")

# # Setup Gemini-compatible OpenAI client
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# # Configure model
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# # Runner configuration
# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # -------------------------
# # Safe background TTS
# # -------------------------
# def speak_text_background(text: str, delay: float = 0.15):
#     """
#     Speak text using pyttsx3 in a background thread.
#     A short delay allows microphone resource to be released on some systems.
#     """
#     def _worker(t, d):
#         try:
#             # small delay to ensure mic released
#             time.sleep(d)
#             engine = pyttsx3.init()
#             # optional: adjust rate/volume/voice here
#             # rate = engine.getProperty('rate')
#             # engine.setProperty('rate', rate-20)
#             engine.say(t)
#             engine.runAndWait()
#             try:
#                 engine.stop()
#             except Exception:
#                 pass
#         except Exception as e:
#             # print to console for debugging; don't crash main loop
#             print("TTS error:", e)

#     th = threading.Thread(target=_worker, args=(text, delay), daemon=True)
#     th.start()

# # Backwards-compatible sync wrapper (if you prefer blocking)
# def speak_text(text: str):
#     speak_text_background(text)

# # -------------------------
# # Voice input
# # -------------------------
# async def get_voice_input() -> str:
#     """Listen from microphone and convert speech to text."""
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening… please speak now.")
#         r.adjust_for_ambient_noise(source)
#         audio = r.listen(source)
#     try:
#         text = r.recognize_google(audio)
#         print(f"You (voice): {text}")
#         return text
#     except Exception as e:
#         print("Sorry, could not understand your voice. Please type instead.")
#         return ""

# # ✅ Tool 1: Fetch single or multiple prices
# @function_tool
# async def get_crypto_price(symbols: str) -> str:
#     url_base = "https://api.binance.com/api/v3/ticker/price"
#     symbol_list = [s.strip().upper() for s in symbols.split(",")]
#     results = []

#     async with httpx.AsyncClient() as client:
#         for symbol in symbol_list:
#             try:
#                 url = f"{url_base}?symbol={symbol}"
#                 response = await client.get(url)
#                 data = response.json()

#                 if response.status_code == 200 and "price" in data:
#                     price = float(data["price"])
#                     results.append(f"{symbol}: ${price:,.8f}")
#                 else:
#                     results.append(f"{symbol}: ❌ Invalid or not found")

#             except Exception as e:
#                 results.append(f"{symbol}: ❌ Error - {str(e)}")

#     return "Prices:\n" + "\n".join(results)

# # ✅ Tool 2: List all active Binance trading pairs
# @function_tool
# async def list_all_binance_symbols() -> str:
#     url = "https://api.binance.com/api/v3/exchangeInfo"
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url)
#             data = response.json()

#         symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
#         total = len(symbols)
#         sample = ", ".join(symbols[:20])

#         return f"Total trading pairs: {total}\nSample: {sample}, ..."
    
#     except Exception as e:
#         return f"Error fetching symbols: {str(e)}"

# # ✅ Define the agent with tools
# agent = Agent(
#     name="Crypto Smart Agent",
#     instructions=(
#         "You are a helpful assistant that can fetch crypto prices from Binance. "
#         "You can fetch one or multiple symbols, and list all trading pairs."
#     ),
#     model=model,
#     tools=[get_crypto_price, list_all_binance_symbols],
# )

# # ✅ Main interaction loop with voice/text input and voice output
# async def main():
#     while True:
#         print("Type your message or say 'voice' to speak. Type 'quit' to exit.")
#         mode = input("Mode (type/voice/quit): ").strip().lower()
#         if mode == "quit":
#             break
        
#         if mode == "voice":
#             user_input = await get_voice_input()
#             if not user_input:
#                 # fallback to typing if voice not understood
#                 user_input = input("You (type): ")
#         else:
#             user_input = input("You: ")

#         if user_input.lower() == "quit":
#             break

#         # run the agent
#         try:
#             result = await Runner.run(agent, user_input, run_config=config)
#         except Exception as e:
#             print("Agent run error:", e)
#             continue

#         # Extract response text
#         response_text = ""
#         for item in reversed(result.new_items):
#             try:
#                 response_text = item.raw_item.content[0].text
#                 break
#             except Exception:
#                 continue

#         if response_text:
#             # always print text
#             print("\nAssistant:", response_text)
#             # speak in background (non-blocking) so both text and voice will appear
#             speak_text_background(response_text, delay=0.12)

# if __name__ == "__main__":
#     asyncio.run(main())




# import os
# import datetime
# import asyncio
# from dotenv import load_dotenv
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
# from agents.run import RunConfig
# import httpx

# import speech_recognition as sr
# import pyttsx3

# # Load environment variables
# load_dotenv()

# # Load Gemini API key
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is missing in .env file.")

# # Setup Gemini-compatible OpenAI client
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# # Configure model
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# # Runner configuration
# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # Initialise TTS engine (voice output)
# tts_engine = pyttsx3.init()

# def speak_text(text: str):
#     """Convert text to speech and play it."""
#     tts_engine.say(text)
#     tts_engine.runAndWait()

# async def get_voice_input() -> str:
#     """Listen from microphone and convert speech to text."""
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening… please speak now.")
#         r.adjust_for_ambient_noise(source)
#         audio = r.listen(source)
#     try:
#         text = r.recognize_google(audio)
#         print(f"You (voice): {text}")
#         return text
#     except Exception as e:
#         print("Sorry, could not understand your voice. Please type instead.")
#         return ""

# # ✅ Tool 1: Fetch single or multiple prices
# @function_tool
# async def get_crypto_price(symbols: str) -> str:
#     url_base = "https://api.binance.com/api/v3/ticker/price"
#     symbol_list = [s.strip().upper() for s in symbols.split(",")]
#     results = []

#     async with httpx.AsyncClient() as client:
#         for symbol in symbol_list:
#             try:
#                 url = f"{url_base}?symbol={symbol}"
#                 response = await client.get(url)
#                 data = response.json()

#                 if response.status_code == 200 and "price" in data:
#                     price = float(data["price"])
#                     results.append(f"{symbol}: ${price:,.8f}")
#                 else:
#                     results.append(f"{symbol}: ❌ Invalid or not found")

#             except Exception as e:
#                 results.append(f"{symbol}: ❌ Error - {str(e)}")

#     return "Prices:\n" + "\n".join(results)

# # ✅ Tool 2: List all active Binance trading pairs
# @function_tool
# async def list_all_binance_symbols() -> str:
#     url = "https://api.binance.com/api/v3/exchangeInfo"
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url)
#             data = response.json()

#         symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
#         total = len(symbols)
#         sample = ", ".join(symbols[:20])

#         return f"Total trading pairs: {total}\nSample: {sample}, ..."
    
#     except Exception as e:
#         return f"Error fetching symbols: {str(e)}"

# # ✅ Define the agent with tools
# agent = Agent(
#     name="Crypto Smart Agent",
#     instructions=(
#         "You are a helpful assistant that can fetch crypto prices from Binance. "
#         "You can fetch one or multiple symbols, and list all trading pairs."
#     ),
#     model=model,
#     tools=[get_crypto_price, list_all_binance_symbols],
# )

# # ✅ Main interaction loop with voice/text input and voice output
# async def main():
#     while True:
#         print("Type your message or say 'voice' to speak. Type 'quit' to exit.")
#         mode = input("Mode (type/voice/quit): ").strip().lower()
#         if mode == "quit":
#             break
        
#         if mode == "voice":
#             user_input = await get_voice_input()
#             if not user_input:
#                 # fallback to typing if voice not understood
#                 user_input = input("You (type): ")
#         else:
#             user_input = input("You: ")

#         if user_input.lower() == "quit":
#             break

#         result = await Runner.run(agent, user_input, run_config=config)

#         # Extract response text
#         response_text = ""
#         for item in reversed(result.new_items):
#             try:
#                 response_text = item.raw_item.content[0].text
#                 break
#             except Exception:
#                 continue

#         if response_text:
#             print("\nAssistant:", response_text)
#             speak_text(response_text)

# if __name__ == "__main__":
#     asyncio.run(main())



# import os
# import datetime
# import asyncio
# from dotenv import load_dotenv
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
# from agents.run import RunConfig
# import httpx

# # Load environment variables
# load_dotenv()

# # Load Gemini API key
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY is missing in .env file.")

# # Setup Gemini-compatible OpenAI client
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# # Configure model
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# # Runner configuration
# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # ✅ Tool 1: Fetch single or multiple prices
# @function_tool
# async def get_crypto_price(symbols: str) -> str:
#     """
#     Accepts one or multiple comma-separated symbols (e.g., BTCUSDT, ETHUSDT)
#     and fetches their current prices from Binance.
#     """
#     url_base = "https://api.binance.com/api/v3/ticker/price"
#     symbol_list = [s.strip().upper() for s in symbols.split(",")]
#     results = []

#     async with httpx.AsyncClient() as client:
#         for symbol in symbol_list:
#             try:
#                 url = f"{url_base}?symbol={symbol}"
#                 response = await client.get(url)
#                 data = response.json()

#                 if response.status_code == 200 and "price" in data:
#                     price = float(data["price"])
#                     results.append(f"{symbol}: ${price:,.8f}")
#                 else:
#                     results.append(f"{symbol}: ❌ Invalid or not found")

#             except Exception as e:
#                 results.append(f"{symbol}: ❌ Error - {str(e)}")

#     return "Prices:\n" + "\n".join(results)

# # ✅ Tool 2: List all active Binance trading pairs
# @function_tool
# async def list_all_binance_symbols() -> str:
#     """
#     Fetch all active trading symbols from Binance and return a sample list.
#     """
#     url = "https://api.binance.com/api/v3/exchangeInfo"
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url)
#             data = response.json()

#         symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
#         total = len(symbols)
#         sample = ", ".join(symbols[:20])

#         return f"Total trading pairs: {total}\nSample: {sample}, ..."
    
#     except Exception as e:
#         return f"Error fetching symbols: {str(e)}"

# # ✅ Define the agent with just two tools
# agent = Agent(
#     name="Crypto Smart Agent",
#     instructions=(
#         "You are a helpful assistant that can fetch crypto prices from Binance. "
#         "You can fetch one or multiple symbols, and list all trading pairs."
#     ),
#     model=model,
#     tools=[get_crypto_price, list_all_binance_symbols],
# )

# # ✅ Main interaction loop
# async def main():
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "quit":
#             break
#         result = await Runner.run(agent, user_input, run_config=config)

#         # Print latest model response
#         for item in reversed(result.new_items):
#             try:
#                 print("\nAssistant:", item.raw_item.content[0].text)
#                 break
#             except Exception:
#                 continue

# if __name__ == "__main__":
#     asyncio.run(main())


