from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from openai import OpenAI


EMAIL_DELIMITER = "--- DRAFT EMAIL ---"
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_VOICE = "nova"
DEFAULT_TTS_TIMEOUT_SECONDS = 30.0
DEFAULT_TTS_RETRIES = 2


logger = logging.getLogger(__name__)


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


def _tts_timeout_seconds() -> float:
    raw_value = os.getenv("RENT_AGENT_TTS_TIMEOUT_SECONDS", str(DEFAULT_TTS_TIMEOUT_SECONDS)).strip()
    try:
        parsed = float(raw_value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid RENT_AGENT_TTS_TIMEOUT_SECONDS=%s; using default %.1f",
            raw_value,
            DEFAULT_TTS_TIMEOUT_SECONDS,
        )
        return DEFAULT_TTS_TIMEOUT_SECONDS


def _tts_retries() -> int:
    raw_value = os.getenv("RENT_AGENT_TTS_RETRIES", str(DEFAULT_TTS_RETRIES)).strip()
    try:
        parsed = int(raw_value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning("Invalid RENT_AGENT_TTS_RETRIES=%s; using default %d", raw_value, DEFAULT_TTS_RETRIES)
        return DEFAULT_TTS_RETRIES


def _play_audio_file(audio_path: Path) -> None:
    system_name = platform.system().lower()
    candidate_commands: list[list[str]] = []

    if system_name == "darwin":
        candidate_commands = [
            ["afplay", str(audio_path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(audio_path)],
        ]
    elif system_name == "linux":
        candidate_commands = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(audio_path)],
            ["mpg123", "-q", str(audio_path)],
            ["play", "-q", str(audio_path)],
        ]
    else:
        candidate_commands = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(audio_path)],
        ]

    available_commands = [command for command in candidate_commands if shutil.which(command[0])]
    if not available_commands:
        raise RuntimeError(
            f"No supported audio player found for platform '{system_name}'. Install ffmpeg/ffplay, or use a platform-native player"
        )

    last_error = ""
    for command in available_commands:
        play_result = subprocess.run(command, check=False, capture_output=True, text=True)
        if play_result.returncode == 0:
            logger.debug("Audio playback succeeded with player '%s'", command[0])
            return

        stderr = (play_result.stderr or "").strip()
        last_error = f"{command[0]} exit code {play_result.returncode}: {stderr}"
        logger.warning("Audio playback attempt failed: %s", last_error)

    raise RuntimeError(f"Audio playback failed with all available players. Last error: {last_error}")


def speak_text(text: str) -> None:
    cleaned_text = _strip_markdown(text)
    if not cleaned_text:
        return

    client = _build_openai_client()
    voice = os.getenv("RENT_AGENT_TTS_VOICE", DEFAULT_TTS_VOICE)
    timeout_seconds = _tts_timeout_seconds()
    retries = _tts_retries()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=DEFAULT_TTS_MODEL,
                    voice=voice,
                    input=cleaned_text,
                    timeout=timeout_seconds,
                ) as response:
                    response.stream_to_file(str(temp_path))

                _play_audio_file(temp_path)
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "TTS attempt %d/%d failed: %s",
                    attempt + 1,
                    retries + 1,
                    error,
                )
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))

        if last_error is not None:
            raise RuntimeError(f"TTS failed after {retries + 1} attempt(s): {last_error}")
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
        logger.exception("Voice output unavailable")
        print(f"Voice output unavailable: {error}")
