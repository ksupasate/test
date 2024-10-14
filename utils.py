import asyncio
import base64
import json
import os
import queue
from datetime import datetime

import numpy as np
import pyaudio  # Updated import
import tzlocal
import websockets
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SimpleRealtime:
    def __init__(self, event_loop=None, audio_buffer_cb=None, debug=False):
        self.url = "wss://api.openai.com/v1/realtime"
        self.debug = debug
        self.event_loop = event_loop
        self.logs = []
        self.transcript = ""
        self.ws = None
        self._message_handler_task = None
        self.audio_buffer_cb = audio_buffer_cb

    def is_connected(self):
        return self.ws is not None and self.ws.open

    def log_event(self, event_type, event):
        if self.debug:
            local_timezone = tzlocal.get_localzone()
            now = datetime.now(local_timezone).strftime("%H:%M:%S")
            msg = json.dumps(event)
            self.logs.append((now, event_type, msg))

        return True

    async def connect(self, model="gpt-4o-realtime-preview-2024-10-01"):
        if self.is_connected():
            raise Exception("Already connected")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.ws = await websockets.connect(
            f"{self.url}?model={model}", extra_headers=headers
        )

        # Start the message handler in the same loop as the websocket
        self._message_handler_task = self.event_loop.create_task(
            self._message_handler()
        )

        return True

    async def _message_handler(self):
        try:
            while True:
                if not self.ws:
                    await asyncio.sleep(0.05)
                    continue

                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                    data = json.loads(message)
                    self.receive(data)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"Message handler error: {e}")
            await self.disconnect()

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass
        self._message_handler_task = None
        return True

    def handle_audio(self, event):
        if event.get("type") == "response.audio_transcript.delta":
            self.transcript += event.get("delta")

        if event.get("type") == "response.audio.delta" and self.audio_buffer_cb:
            b64_audio_chunk = event.get("delta")
            decoded_audio_chunk = base64.b64decode(b64_audio_chunk)
            pcm_audio_chunk = np.frombuffer(decoded_audio_chunk, dtype=np.int16)
            self.audio_buffer_cb(pcm_audio_chunk)

    def receive(self, event):
        self.log_event("server", event)
        if "response.audio" in event.get("type"):
            self.handle_audio(event)
        return True

    def send(self, event_name, data=None):
        if not self.is_connected():
            raise Exception("RealtimeAPI is not connected")

        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")

        event = {"type": event_name, **data}

        self.log_event("client", event)

        self.event_loop.create_task(self.ws.send(json.dumps(event)))

        return True


class StreamingAudioRecorder:
    """
    Audio recorder using PyAudio.
    """

    def __init__(self, sample_rate=24000, channels=1, frames_per_buffer=2000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def callback(self, in_data, frame_count, time_info, status):
        """
        This will be called for each audio block
        that gets recorded.
        """
        pcm_audio_chunk = np.frombuffer(in_data, dtype=np.int16)
        self.audio_queue.put(pcm_audio_chunk)
        return (None, pyaudio.paContinue)

    def start_recording(self):
        if self.is_recording:
            return  # Already recording

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.callback,
        )
        self.stream.start_stream()
        self.is_recording = True

    def stop_recording(self):
        if self.is_recording and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.is_recording = False

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def __del__(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
