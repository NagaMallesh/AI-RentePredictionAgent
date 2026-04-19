# AI Rent Prediction Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)
![Model Runtime](https://img.shields.io/badge/Model-Pickle%20Runtime-purple.svg)

An AI assistant for rental listing analysis that combines:
- A trained multi-feature rent model exported from ML-LinearRegression
- A LangChain tool-enabled agent for reasoning over listing details
- Optional voice output (OpenAI TTS + local audio playback on macOS)
- CLI workflows for interactive analysis and one-shot prompts

The agent estimates fair monthly rent and compares it against listed rent to classify deals as underpriced, fair, or overpriced.

## Key Features

- Fair-rent estimation with a reusable tool: `predict_rent_value`
- LangChain agent orchestration via `create_agent`
- Structured guidance in the system prompt for consistent outputs
- Interactive menu mode and one-shot CLI mode
- Environment-based configuration (`.env` supported)
- Runtime model validation with clear error messages
- Optional spoken responses for recommendations and email-ready alerts

## Project Architecture

```text
AI-RentePredictionAgent/
├── src/
│   └── agent/
│       ├── model_runtime.py          # Model loading, validation, prediction, feature contribution helpers
│       ├── rent_prediction_agent.py  # LangChain tool + agent creation + task execution (with retry/timeout)
│       └── voice_output.py           # TTS generation, platform-aware playback, structured logging
├── models/
│   └── rent_model.pkl                # Exported model artifact from ML-LinearRegression
├── tests/
│   ├── test_model_runtime.py         # Unit tests for loading, validation, prediction, explanations
│   ├── test_langchain_tool_agent.py  # Tool invocation test
│   └── test_voice_output.py          # Mocked tests for TTS, retries, playback fallback
├── main.py                           # CLI entrypoint (interactive + one-shot, logging config)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── README.md
```

## Prerequisites

- Python 3.11+
- OpenAI-compatible API key
- Exported pickle model (`rent_model.pkl`) produced from the ML-LinearRegression project

## Installation

From AI-RentePredictionAgent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Export Model Artifact (From ML-LinearRegression)

This project expects a 5-feature linear regression model in pickle format.

From ML-LinearRegression:

```bash
python scripts/export_model_for_agent.py \
  --output ../AI-RentePredictionAgent/models/rent_model.pkl
```

Default source CSV in that script is:
- `data/sample_data.csv`

The exported model payload must include:
- `coefficients`
- `intercept`
- `n_features` (must be 5 for this agent)

## Environment Configuration

Required:
- `OPENAI_API_KEY`

Recommended:
- `OPENAI_BASE_URL=https://us.api.openai.com/v1`

Optional overrides:
  - `RENT_AGENT_LLM_MODEL` (default: `gpt-4o-mini`) — LLM model identifier
  - `RENT_AGENT_MODEL_PATH` (default: `models/rent_model.pkl`) — Path to pickle model file
  - `RENT_AGENT_VOICE_ENABLED` (default: `false`) — Enable TTS voice output (true/false)
  - `RENT_AGENT_TTS_VOICE` (default: `nova`) — OpenAI TTS voice name
  - `RENT_AGENT_REQUEST_TIMEOUT_SECONDS` (default: `45`) — LLM request timeout in seconds
  - `RENT_AGENT_REQUEST_RETRIES` (default: `2`) — LLM request retry attempts
  - `RENT_AGENT_TTS_TIMEOUT_SECONDS` (default: `30`) — TTS request timeout in seconds
  - `RENT_AGENT_TTS_RETRIES` (default: `2`) — TTS request retry attempts
  - `RENT_AGENT_LOG_LEVEL` (default: `INFO`) — Logging level (DEBUG/INFO/WARNING/ERROR)

Create `.env` in AI-RentePredictionAgent (or copy from `.env.example`):

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://us.api.openai.com/v1
RENT_AGENT_VOICE_ENABLED=false
# Optional overrides
# RENT_AGENT_LLM_MODEL=gpt-4o-mini
# RENT_AGENT_MODEL_PATH=models/rent_model.pkl
# RENT_AGENT_TTS_VOICE=nova
# RENT_AGENT_REQUEST_TIMEOUT_SECONDS=45
# RENT_AGENT_REQUEST_RETRIES=2
# RENT_AGENT_TTS_TIMEOUT_SECONDS=30
# RENT_AGENT_TTS_RETRIES=2
# RENT_AGENT_LOG_LEVEL=INFO
```

## Production Hardening

The agent includes production-grade resilience:

### Timeouts and Retries

- **LLM Requests**: Configurable timeout (default 45s) with exponential backoff retry (default 2 retries)
- **TTS Requests**: Configurable timeout (default 30s) with exponential backoff retry (default 2 retries)
- Use `RENT_AGENT_REQUEST_TIMEOUT_SECONDS`, `RENT_AGENT_REQUEST_RETRIES`, `RENT_AGENT_TTS_TIMEOUT_SECONDS`, `RENT_AGENT_TTS_RETRIES` to tune

### Structured Logging

- All errors and warnings are logged in structured format: `time=... level=... logger=... message=...`
- Set `RENT_AGENT_LOG_LEVEL=DEBUG` to see verbose retry and fallback activity
- Errors are printed to stdout (user-facing) AND logged (operational observability)

### Cross-Platform Audio Playback

- Playback uses platform-aware fallback players:
  - macOS: `afplay` → `ffplay`
  - Linux: `ffplay` → `mpg123` → `play`
  - Other OS: `ffplay`
- If no player is available, a clear error message guides you to install one

## Voice Agent (Text-to-Speech)

When voice is enabled, the app converts the assistant response to speech and plays it locally.

- Voice runs only when `RENT_AGENT_VOICE_ENABLED=true`.
- The app uses OpenAI speech generation (TTS model: `tts-1`) and plays audio via platform-specific players.
- If the response contains a draft-email section, the app also speaks a short follow-up message: "A draft email inquiry is ready to send."
- Retry logic means transient TTS or playback failures often succeed on the second attempt.

### Enable Voice

In `AI-RentePredictionAgent/.env`:

```bash
RENT_AGENT_VOICE_ENABLED=true
# Optional
# RENT_AGENT_TTS_VOICE=nova
```

Then run from project root:

```bash
source .venv/bin/activate
python main.py
```

### Voice Quick Test

Use this to verify local audio output first:

```bash
afplay /System/Library/Sounds/Glass.aiff
```

Then run an end-to-end app voice test from project root:

```bash
source .venv/bin/activate
python main.py --input "Please do a voice test. Respond in exactly one short sentence: Audio test successful."
```

Expected result:
- You should hear a short spoken sentence.
- If something fails, the terminal will print a message starting with `Voice output unavailable:`.

## Usage

### Interactive Mode

```bash
python main.py
```

Menu options:
1. Analyze a listing
2. Show configuration
3. Exit

### One-Shot Mode

```bash
python main.py --input "The apartment at 123 Maple St is 1200 sq ft with 2 beds, listed at $2200. Is it a good deal?"
```

### LLM Model Override

```bash
python main.py --model gpt-4o-mini --input "2 bed, 1100 sqft, listed at $2100, furnished. Should I negotiate?"
```

## Expected Agent Behavior

When enough details are present, the agent calls the prediction tool with:
- `bedrooms`
- `size_sqft`
- `location_score` (default `7.5`)
- `amenities` (default `3`)
- `furnished` (default `0`)

After prediction, the response should:
1. Explain estimated fair rent
2. Compare estimate vs listed price
3. Classify the listing (underpriced/fair/overpriced)
4. Draft a short email inquiry if underpriced

## Testing

Run all tests (9 tests total):

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or run individually:

```bash
python tests/test_model_runtime.py          # Model loading, validation, prediction
python tests/test_langchain_tool_agent.py   # Tool invocation (skipped if model missing)
python tests/test_voice_output.py           # Voice TTS, retries, playback fallback (mocked)
```

Notes:
- `test_voice_output.py` uses mocks — no real OpenAI API calls or audio output during test.
- `test_langchain_tool_agent.py` is skipped automatically if model file is missing.
- All tests pass with zero external dependencies or secrets required.

## Troubleshooting

- Error: `OPENAI_API_KEY is not set`
  - Set `OPENAI_API_KEY` in your shell or `.env`.

- Error: `Model file not found`
  - Export the model from ML-LinearRegression.
  - Or set `RENT_AGENT_MODEL_PATH` to a valid pickle file.

- Error mentioning `incorrect_hostname`
  - Set `OPENAI_BASE_URL=https://us.api.openai.com/v1`.

- No voice/audio is heard
  - Ensure `RENT_AGENT_VOICE_ENABLED=true` in `.env`.
  - Run the quick test command in the Voice Quick Test section first.
  - Verify you run from project root and activate the correct venv: `source .venv/bin/activate`.
  - Confirm system audio works with: `afplay /System/Library/Sounds/Glass.aiff`.
  - If playback fails, the app now prints a detailed message starting with: `Voice output unavailable:`.

- Error: model has wrong feature count
  - This agent supports only 5-feature models.
  - Re-export using the multi-feature CSV pipeline.

## Integration Notes

- Model training and export live in the ML-LinearRegression repository.
- Inference and conversational analysis live in this repository.
- This separation lets you retrain independently while keeping agent logic stable.

## Main Components

- **CLI entrypoint and logging**: `main.py` — Interactive menu, one-shot mode, structured logging setup
- **LangChain agent + tool**: `src/agent/rent_prediction_agent.py` — Agent orchestration with timeout/retry logic
- **Model runtime and validation**: `src/agent/model_runtime.py` — Model loading, prediction, feature contribution
- **Voice output and TTS**: `src/agent/voice_output.py` — TTS generation with platform-aware playback, retries, structured logging
- **Tests**: `tests/test_*.py` — Unit tests with mocks for reproducibility
