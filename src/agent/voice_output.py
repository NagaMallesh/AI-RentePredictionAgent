from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI


EMAIL_DELIMITER = "--- DRAFT EMAIL ---"
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_VOICE = "nova"


def _voice_enabled() -> bool:
    value = os.getenv("RENT_AGENT_VOICE_ENABLED", "false").strip().lower()
    return value in TRUE_VALUES


def _strip_markdown(text: str) -> str:
    # Remove common markdown formatting so TTS reads clean prose.
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^(#{1,6})\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_recommendation_and_email(response: str) -> tuple[str, bool]:
    if EMAIL_DELIMITER in response:
        recommendation, _ = response.split(EMAIL_DELIMITER, maxsplit=1)
        return recommendation.strip(), True
    return response.strip(), False


def _build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for voice output")

    base_url = os.getenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def speak_text(text: str) -> None:
    cleaned_text = _strip_markdown(text)
    if not cleaned_text:
        return

    client = _build_openai_client()
    voice = os.getenv("RENT_AGENT_TTS_VOICE", DEFAULT_TTS_VOICE)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with client.audio.speech.with_streaming_response.create(
            model=DEFAULT_TTS_MODEL,
            voice=voice,
            input=cleaned_text,
        ) as response:
            response.stream_to_file(str(temp_path))

        if shutil.which("afplay") is None:
            raise RuntimeError("'afplay' command not found on this system")

        play_result = subprocess.run(["afplay", str(temp_path)], check=False, capture_output=True, text=True)
        if play_result.returncode != 0:
            stderr = (play_result.stderr or "").strip()
            raise RuntimeError(f"Audio playback failed (afplay exit code {play_result.returncode}): {stderr}")
    finally:
        temp_path.unlink(missing_ok=True)


def speak_response(full_response: str) -> None:
    if not _voice_enabled():
        return

    recommendation, has_email = split_recommendation_and_email(full_response)

    try:
        speak_text(recommendation)
        if has_email:
            speak_text("A draft email inquiry is ready to send.")
    except Exception as error:
        print(f"Voice output unavailable: {error}")
