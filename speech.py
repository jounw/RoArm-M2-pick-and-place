"""Voice I/O for the pick-and-place demo.

listen()  -- record from the default mic until ~1.5s of silence, then
             transcribe Korean speech to text.
speak()   -- synthesize Korean speech and play it through the speakers.

Backends (env vars; "auto" picks what is available, no OpenAI needed):
  STT_BACKEND = auto | gemini | openai
      gemini: audio sent to Gemini Flash (~2.3 s, good Korean) -- default when a
              Gemini key exists;  openai: gpt-4o-mini-transcribe (~1 s)
  TTS_BACKEND = auto | say | openai
      say:    macOS built-in voice (TTS_VOICE, default "Yuna"), instant, offline
              -- default on macOS;  openai: tts-1 (TTS_VOICE e.g. "nova")

Keys come from the same env-var-first/api.txt-fallback scheme as vision.py.
"""

from __future__ import annotations

import base64
import io
import os
import platform
import subprocess
import tempfile
import wave

import numpy as np
import requests
import sounddevice as sd

import vision
from vision import _api_key

# gpt-4o-mini-transcribe: ~1s per utterance vs ~3s for whisper-1, same accuracy on short Korean commands
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
STT_BACKEND = os.getenv("STT_BACKEND", "auto").lower()
TTS_BACKEND = os.getenv("TTS_BACKEND", "auto").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "")  # "" = backend default (say: Yuna, openai: nova)
STT_GEMINI_MODEL = os.getenv("STT_GEMINI_MODEL", vision.GEMINI_MODEL)


def stt_backend() -> str:
    if STT_BACKEND != "auto":
        return STT_BACKEND
    return "gemini" if vision._read_key("GEMINI_API_KEY", vision.GEMINI_KEY_FILE) else "openai"


def tts_backend() -> str:
    if TTS_BACKEND != "auto":
        return TTS_BACKEND
    return "say" if platform.system() == "Darwin" else "openai"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500  # mean |amplitude| below this counts as silence; tune per room
SILENCE_DURATION = 1.5  # this much continuous silence ends the recording
MIN_SPEECH_DURATION = 0.5  # discard blips shorter than this


def _client():
    from openai import OpenAI

    return OpenAI(api_key=_api_key())


def wav_bytes(audio: np.ndarray, samplerate: int, target_rate: int = 16000) -> tuple[bytes, int]:
    """int16 mono -> wav bytes, decimated to ~16 kHz (3x smaller upload, speech-safe)."""
    step = max(1, int(round(samplerate / target_rate)))
    if step > 1:
        audio = audio[::step]
        samplerate = samplerate // step
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(np.ascontiguousarray(audio, dtype=np.int16).tobytes())
    return buf.getvalue(), samplerate


def _transcribe_gemini(audio: np.ndarray, samplerate: int, prompt: str | None) -> str:
    data, _ = wav_bytes(audio, samplerate)
    hint = f" Expected vocabulary / context: {prompt}" if prompt else ""
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(data).decode("ascii")}},
            {"text": "Transcribe this Korean speech exactly as spoken. Output only the transcript, no quotes, "
                     "no commentary. If there is no intelligible speech, output an empty string." + hint},
        ]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}, "temperature": 0.0},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{STT_GEMINI_MODEL}:generateContent"
    r = requests.post(url, json=body, headers={"x-goog-api-key": vision._gemini_key()}, timeout=30)
    r.raise_for_status()
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"')
    except (KeyError, IndexError):
        return ""


def _is_speech(chunk: np.ndarray) -> bool:
    return np.abs(chunk).mean() > SILENCE_THRESHOLD


def record_until_silence(max_seconds: float = 15.0) -> np.ndarray | None:
    """Wait for speech to start, record until it stops. None if nothing usable."""
    print("(듣는 중... 말씀하세요)")
    frames: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    silent_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, blocksize=CHUNK_SIZE) as stream:
        for _ in range(max_chunks):
            chunk, _overflow = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()
            if not speech_started:
                if _is_speech(chunk):
                    speech_started = True
                    frames.append(chunk)
            else:
                frames.append(chunk)
                if _is_speech(chunk):
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                    if silent_chunks >= silent_limit:
                        break

    if not frames:
        return None
    audio = np.concatenate(frames)
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


def transcribe(audio: np.ndarray, samplerate: int = SAMPLE_RATE, prompt: str | None = None) -> str:
    """`prompt` biases the recognizer toward the expected vocabulary (e.g. short
    Korean robot commands) -- helps a lot with 1-2 word utterances."""
    if stt_backend() == "gemini":
        return _transcribe_gemini(audio, samplerate, prompt)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        with open(path, "rb") as f:
            kwargs = {"model": STT_MODEL, "file": f, "language": "ko"}
            if prompt:
                kwargs["prompt"] = prompt
            result = _client().audio.transcriptions.create(**kwargs)
        return result.text.strip()
    finally:
        os.unlink(path)


def listen() -> str | None:
    """One utterance from the mic -> Korean text, or None if nothing was heard."""
    audio = record_until_silence()
    if audio is None:
        return None
    return transcribe(audio)


def speak(text: str) -> None:
    """Synthesize `text` and play it. Prints instead if synthesis/playback fails
    (a dead speaker should never kill the robot loop)."""
    print(f"[로봇] {text}")
    if tts_backend() == "say":
        try:
            subprocess.run(["say", "-v", TTS_VOICE or "Yuna", "-r", os.getenv("TTS_RATE", "215"), text], check=False)
        except Exception as e:
            print(f"(TTS 실패: {e})")
        return
    try:
        response = _client().audio.speech.create(
            model=TTS_MODEL, voice=TTS_VOICE or "nova", input=text, response_format="wav"
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(response.content)
            path = tmp.name
        try:
            subprocess.run(["afplay", path], check=False)
        finally:
            os.unlink(path)
    except Exception as e:
        print(f"(TTS 실패: {e})")


if __name__ == "__main__":
    # quick mic->STT->TTS loop for testing: speak, hear it echoed back
    while True:
        heard = listen()
        if heard:
            print(f"[나] {heard}")
            speak(f"{heard} 라고 하셨네요")
