import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from src.agent import voice_output


class TestVoiceOutput(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _mock_openai_client(self):
        client = mock.MagicMock()
        streamed_response = mock.MagicMock()
        context_manager = mock.MagicMock()
        context_manager.__enter__.return_value = streamed_response
        context_manager.__exit__.return_value = False
        client.audio.speech.with_streaming_response.create.return_value = context_manager
        return client, streamed_response

    def test_speak_response_skips_when_voice_disabled(self):
        os.environ["RENT_AGENT_VOICE_ENABLED"] = "false"
        with mock.patch("src.agent.voice_output.speak_text") as mock_speak_text:
            voice_output.speak_response("hello")
            mock_speak_text.assert_not_called()

    def test_speak_response_reads_email_notice(self):
        with mock.patch("src.agent.voice_output._voice_enabled", return_value=True), mock.patch(
            "src.agent.voice_output.speak_text"
        ) as mock_speak_text:
            voice_output.speak_response("Analysis\n--- DRAFT EMAIL ---\nEmail draft")

        self.assertEqual(mock_speak_text.call_count, 2)
        mock_speak_text.assert_any_call("Analysis")
        mock_speak_text.assert_any_call("A draft email inquiry is ready to send.")

    def test_play_audio_file_uses_available_fallback(self):
        with mock.patch("src.agent.voice_output.platform.system", return_value="Linux"), mock.patch(
            "src.agent.voice_output.shutil.which", side_effect=lambda cmd: "/usr/bin/mpg123" if cmd == "mpg123" else None
        ), mock.patch("src.agent.voice_output.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["mpg123"], returncode=0)
            voice_output._play_audio_file(Path("dummy.mp3"))

        mock_run.assert_called_once()
        command = mock_run.call_args[0][0]
        self.assertEqual(command[0], "mpg123")

    def test_speak_text_retries_and_succeeds(self):
        client, streamed_response = self._mock_openai_client()
        os.environ["RENT_AGENT_TTS_RETRIES"] = "1"
        os.environ["RENT_AGENT_TTS_TIMEOUT_SECONDS"] = "10"

        with mock.patch("src.agent.voice_output._build_openai_client", return_value=client), mock.patch(
            "src.agent.voice_output._play_audio_file", side_effect=[RuntimeError("player failed"), None]
        ) as mock_play_audio, mock.patch("src.agent.voice_output.time.sleep"):
            voice_output.speak_text("hello")

        self.assertEqual(client.audio.speech.with_streaming_response.create.call_count, 2)
        self.assertEqual(mock_play_audio.call_count, 2)
        streamed_response.stream_to_file.assert_called()

    def test_play_audio_file_raises_when_no_supported_player_exists(self):
        with mock.patch("src.agent.voice_output.platform.system", return_value="Darwin"), mock.patch(
            "src.agent.voice_output.shutil.which", return_value=None
        ):
            with self.assertRaises(RuntimeError):
                voice_output._play_audio_file(Path("dummy.mp3"))


if __name__ == "__main__":
    unittest.main()
