import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent.rent_prediction_agent import run_agent_task
from src.agent.voice_output import speak_response


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")


def _configure_logging() -> None:
    level_name = os.getenv("RENT_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="time=%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
    )


def _validate_runtime_configuration() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your shell environment or create AI-RentePredictionAgent/.env with OPENAI_API_KEY=your_key"
        )

    model_path = Path(os.getenv("RENT_AGENT_MODEL_PATH", "models/rent_model.pkl")).expanduser()
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. Export or configure RENT_AGENT_MODEL_PATH to a valid file"
        )


def _show_configuration() -> None:
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("RENT_AGENT_LLM_MODEL", "gpt-4o-mini")
    model_path = os.getenv("RENT_AGENT_MODEL_PATH", "models/rent_model.pkl")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    voice_enabled = os.getenv("RENT_AGENT_VOICE_ENABLED", "false")
    voice_name = os.getenv("RENT_AGENT_TTS_VOICE", "nova")
    request_timeout = os.getenv("RENT_AGENT_REQUEST_TIMEOUT_SECONDS", "45")
    request_retries = os.getenv("RENT_AGENT_REQUEST_RETRIES", "2")
    tts_timeout = os.getenv("RENT_AGENT_TTS_TIMEOUT_SECONDS", "30")
    tts_retries = os.getenv("RENT_AGENT_TTS_RETRIES", "2")
    hybrid_mode = os.getenv("RENT_AGENT_HYBRID_MODE", "true")
    llm_fallback = os.getenv("RENT_AGENT_LLM_FALLBACK_ENABLED", "true")

    print("\nCurrent configuration")
    print(f"- OPENAI_API_KEY set: {'Yes' if api_key_set else 'No'}")
    print(f"- RENT_AGENT_LLM_MODEL: {model_name}")
    print(f"- RENT_AGENT_MODEL_PATH: {model_path}")
    print(f"- OPENAI_BASE_URL: {openai_base_url}")
    print(f"- RENT_AGENT_VOICE_ENABLED: {voice_enabled}")
    print(f"- RENT_AGENT_TTS_VOICE: {voice_name}")
    print(f"- RENT_AGENT_REQUEST_TIMEOUT_SECONDS: {request_timeout}")
    print(f"- RENT_AGENT_REQUEST_RETRIES: {request_retries}")
    print(f"- RENT_AGENT_TTS_TIMEOUT_SECONDS: {tts_timeout}")
    print(f"- RENT_AGENT_TTS_RETRIES: {tts_retries}")
    print(f"- RENT_AGENT_HYBRID_MODE: {hybrid_mode}")
    print(f"- RENT_AGENT_LLM_FALLBACK_ENABLED: {llm_fallback}")


def _interactive_menu(model_name_override: str | None = None) -> None:
    print("=" * 64)
    print("🏠 AI Rent Prediction Agent")
    print("LangChain-powered listing analysis")
    print("=" * 64)

    while True:
        print("\nChoose an option:")
        print("1. Analyze a listing")
        print("2. Show configuration")
        print("3. Exit")

        choice = input("> ").strip()

        if choice == "1":
            user_input = input("\nEnter your question/listing details:\n> ").strip()
            if not user_input:
                print("Please enter some text.")
                continue

            try:
                output = run_agent_task(user_input, model_name=model_name_override)
                print(f"\n{output}")
                speak_response(output)
            except Exception as error:
                print(f"\nCould not run agent: {error}")
                if "incorrect_hostname" in str(error):
                    print("Tip: set OPENAI_BASE_URL=https://us.api.openai.com/v1 in .env")
                else:
                    print("Tip: ensure OPENAI_API_KEY is set and model file exists.")

        elif choice == "2":
            _show_configuration()

        elif choice == "3":
            print("\nGoodbye!")
            return

        else:
            print("Please choose 1, 2, or 3.")


def main() -> None:
    _load_environment()
    _configure_logging()

    parser = argparse.ArgumentParser(description="LangChain rent assistant")
    parser.add_argument(
        "--input",
        required=False,
        help="Natural language task, e.g. listing details and requested analysis",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional LLM model override, otherwise uses RENT_AGENT_LLM_MODEL or gpt-4o-mini",
    )
    args = parser.parse_args()

    _validate_runtime_configuration()

    if args.input:
        output = run_agent_task(args.input, model_name=args.model)
        print(output)
        speak_response(output)
        return

    _interactive_menu(model_name_override=args.model)


if __name__ == "__main__":
    main()
