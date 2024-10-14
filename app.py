import asyncio
import base64
import json
import threading
import time  # Added for sleep in audio stream
from asyncio import run_coroutine_threadsafe

import numpy as np
import pyaudio  # Updated import
import streamlit as st

from constants import HIDE_STREAMLIT_RUNNING_MAN_SCRIPT
from utils import SimpleRealtime, StreamingAudioRecorder

# Define the system message
SYSTEM_MESSAGE = (
    "You are a highly skilled AI assistant who specializes in translating Thai to English. "
    "Your job is to accurately and clearly interpret and translate all Thai speech or text "
    "into natural, fluent English. Please ensure that the translation remains polite and precise. "
    "Do not change the meaning of what is being said, and always prioritize clarity and correctness."
)

st.set_page_config(page_title="PreceptorAI Interpreter", layout="wide")

audio_buffer = np.array([], dtype=np.int16)
buffer_lock = threading.Lock()

# Initialize session state variables
if "audio_stream_started" not in st.session_state:
    st.session_state.audio_stream_started = False
if "connected" not in st.session_state:
    st.session_state.connected = False
    st.session_state.connection_error = None
if "client" not in st.session_state:
    st.session_state.client = None
if "recorder" not in st.session_state:
    st.session_state.recorder = StreamingAudioRecorder()
if "recording" not in st.session_state:
    st.session_state.recording = False


def audio_buffer_cb(pcm_audio_chunk):
    """
    Callback function to fill the audio buffer.
    """
    global audio_buffer
    with buffer_lock:
        audio_buffer = np.concatenate([audio_buffer, pcm_audio_chunk])


# Callback function for real-time playback using PyAudio
def sd_audio_cb(in_data, frame_count, time_info, status):
    global audio_buffer
    channels = 1
    with buffer_lock:
        if len(audio_buffer) >= frame_count:
            data = audio_buffer[:frame_count].reshape(-1, channels).tobytes()
            audio_buffer = audio_buffer[frame_count:]
        else:
            data = (np.zeros(frame_count * channels, dtype=np.int16)).tobytes()
    return (data, pyaudio.paContinue)


def start_audio_stream():
    p = pyaudio.PyAudio()

    def py_audio_callback(in_data, frame_count, time_info, status):
        global audio_buffer
        channels = 1
        with buffer_lock:
            if len(audio_buffer) >= frame_count:
                data = audio_buffer[:frame_count].reshape(-1, channels).tobytes()
                audio_buffer = audio_buffer[frame_count:]
            else:
                data = (np.zeros(frame_count * channels, dtype=np.int16)).tobytes()
        return (data, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
        stream_callback=py_audio_callback,
        frames_per_buffer=2000,
    )

    stream.start_stream()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


@st.cache_resource(show_spinner=False)
def create_loop():
    """
    Creates a globally cached event loop running in a separate thread.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


st.session_state.event_loop, worker_thread = create_loop()


def run_async(coroutine):
    """
    Helper for running an async function in the globally cached event loop.
    """
    future = run_coroutine_threadsafe(coroutine, st.session_state.event_loop)
    return future.result()


# Function to send session update to OpenAI WebSocket
async def send_session_update(ws):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "instructions": SYSTEM_MESSAGE,
        },
    }
    print("Sending session update:", session_update)
    await ws.send(json.dumps(session_update))


# Function to initialize and connect the client
def initialize_client():
    if st.session_state.client is None:
        st.session_state.client = SimpleRealtime(
            event_loop=st.session_state.event_loop,
            audio_buffer_cb=audio_buffer_cb,
            debug=True,
        )
        try:
            run_async(st.session_state.client.connect())
            if st.session_state.client.is_connected():
                st.session_state.connected = True
                # Assuming the client exposes a WebSocket attribute named 'ws'
                # You may need to adjust this based on your SimpleRealtime implementation
                if hasattr(st.session_state.client, "ws"):
                    run_async(send_session_update(st.session_state.client.ws))
                else:
                    print("WebSocket attribute 'ws' not found in client.")
            else:
                st.session_state.connected = False
                st.session_state.connection_error = "Unknown error during connection."
        except Exception as e:
            st.session_state.connected = False
            st.session_state.connection_error = str(e)


# Initialize the client and attempt to connect
initialize_client()


def start_recording():
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.recorder.start_recording()


def stop_recording():
    if st.session_state.recording:
        st.session_state.recording = False
        st.session_state.recorder.stop_recording()
        st.session_state.client.send("input_audio_buffer.commit")
        # Include the system_message parameter here
        st.session_state.client.send(
            "response.create", {"system_message": SYSTEM_MESSAGE}
        )


@st.fragment(run_every=1)
def audio_player():
    if not st.session_state.audio_stream_started:
        st.session_state.audio_stream_started = True
        # Start the audio stream in a separate thread to prevent blocking
        threading.Thread(target=start_audio_stream, daemon=True).start()


@st.fragment(run_every=1)
def audio_recorder():
    if st.session_state.recording:
        while not st.session_state.recorder.audio_queue.empty():
            chunk = st.session_state.recorder.get_audio_chunk()
            if chunk is not None:
                st.session_state.client.send(
                    "input_audio_buffer.append",
                    {"audio": base64.b64encode(chunk).decode()},
                )


@st.fragment(run_every=1)
def response_area():
    """
    Fragment to display the transcribed text of the audio input.
    """
    transcript = st.session_state.client.transcript
    st.markdown(
        "<h3>📝 Transcribed Text</h3>",
        unsafe_allow_html=True,
    )
    if transcript:
        st.write(
            f"""
            <div style="padding: 15px; background-color: #ecf0f1; border-radius: 10px; min-height: 150px;">
                {transcript}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write(
            """
            <div style="padding: 15px; background-color: #ecf0f1; border-radius: 10px; color: #7f8c8d; min-height: 150px;">
                The transcribed text will appear here after recording.
            </div>
            """,
            unsafe_allow_html=True,
        )


def st_app():
    """
    Main Streamlit app function with transcribed text display.
    """
    st.markdown(HIDE_STREAMLIT_RUNNING_MAN_SCRIPT, unsafe_allow_html=True)

    # Custom CSS for modern styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
        body {
            font-family: 'Montserrat', sans-serif;
            background-color: #f5f5f5;
        }
        /* Header */
        .main-header {
            text-align: center;
            font-size: 2.5em;
            font-weight: 600;
            margin-top: 20px;
            color: #2c3e50;
        }
        .sub-header {
            text-align: center;
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #3498db;
            color: white;
            border: none;
            font-size: 1em;
            font-weight: 600;
            padding: 10px;
            margin-top: 10px;
        }
        .stButton>button:disabled {
            background-color: #95a5a6;
            color: #ecf0f1;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        /* Status Indicator */
        .status-indicator {
            text-align: center;
            font-size: 1em;
            margin-top: 20px;
            color: #34495e;
        }
        /* Footer */
        .footer {
            text-align: center;
            color: #bdc3c7;
            font-size: 0.9em;
            margin-top: 50px;
            margin-bottom: 20px;
        }
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #3498db;
        }
        /* Other */
        .reportview-container .main footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section
    st.markdown(
        "<div class='main-header'>PreceptorAI Interpreter 🗣</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub-header'>Experience real-time audio processing with ease</div>",
        unsafe_allow_html=True,
    )
    st.write("---")

    # Main Content
    # Use columns to organize the layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display Transcribed Text
        response_area()

    with col2:
        st.subheader("Controls")
        if st.session_state.connected:
            start_disabled = st.session_state.recording
            stop_disabled = not st.session_state.recording

            # Arrange buttons in a horizontal layout
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.button(
                    "▶️ Start Recording",
                    on_click=start_recording,
                    disabled=start_disabled,
                    key="start_button",
                    help="Start recording audio input.",
                )
            with btn_col2:
                st.button(
                    "⏹️ Stop Recording",
                    on_click=stop_recording,
                    disabled=stop_disabled,
                    key="stop_button",
                    help="Stop recording and process the audio.",
                )

            # Recording Status Indicator
            if st.session_state.recording:
                st.markdown(
                    "<div class='status-indicator'><span style='color: green;'>● Recording in progress...</span></div>",
                    unsafe_allow_html=True,
                )
                st.progress(50)  # You can update the progress value as needed
            elif st.session_state.connected:
                st.markdown(
                    "<div class='status-indicator'><span style='color: blue;'>Ready to record.</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='status-indicator'><span style='color: red;'>Not connected.</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Connecting to the PreceptorAI Interpreter... Please wait.")

        # Connection Status
        st.subheader("Connection Status")
        if st.session_state.connected:
            st.success("✅ Connected to the PreceptorAI Interpreter.")
        else:
            if st.session_state.connection_error:
                st.error(f"❌ Connection error: {st.session_state.connection_error}")
            else:
                st.warning(
                    "Attempting to connect to the PreceptorAI Interpreter... Please wait."
                )

    # Footer
    st.write("---")
    st.markdown(
        "<div class='footer'>© 2024 PreceptorAI Interpreter. All rights reserved.</div>",
        unsafe_allow_html=True,
    )

    audio_player()
    audio_recorder()


if __name__ == "__main__":
    st_app()
