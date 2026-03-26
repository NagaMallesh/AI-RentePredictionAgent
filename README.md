# AI Rent Prediction Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)
![Model Runtime](https://img.shields.io/badge/Model-Pickle%20Runtime-purple.svg)

An AI assistant for rental listing analysis that combines:
- A trained multi-feature rent model exported from ML-LinearRegression
- A LangChain tool-enabled agent for reasoning over listing details
- CLI workflows for interactive analysis and one-shot prompts

The agent estimates fair monthly rent and compares it against listed rent to classify deals as underpriced, fair, or overpriced.

## Key Features

- Fair-rent estimation with a reusable tool: `predict_rent_value`
- LangChain agent orchestration via `create_agent`
- Structured guidance in the system prompt for consistent outputs
- Interactive menu mode and one-shot CLI mode
- Environment-based configuration (`.env` supported)
- Runtime model validation with clear error messages

## Project Architecture

```text
AI-RentePredictionAgent/
├── src/
│   └── agent/
│       ├── model_runtime.py          # Model loading, validation, prediction, feature contribution helpers
│       └── rent_prediction_agent.py  # LangChain tool + agent creation + task execution
├── models/
│   └── rent_model.pkl                # Exported model artifact from ML-LinearRegression
├── tests/
│   ├── test_model_runtime.py         # Unit tests for loading, validation, prediction, explanations
│   └── test_langchain_tool_agent.py  # Tool invocation test
├── main.py                           # CLI entrypoint (interactive + one-shot)
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
- `RENT_AGENT_LLM_MODEL` (default: `gpt-4o-mini`)
- `RENT_AGENT_MODEL_PATH` (default: `models/rent_model.pkl`)

Create `.env` in AI-RentePredictionAgent (or copy from `.env.example`):

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://us.api.openai.com/v1
# Optional overrides
# RENT_AGENT_LLM_MODEL=gpt-4o-mini
# RENT_AGENT_MODEL_PATH=models/rent_model.pkl
```

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

Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or run individually:

```bash
python tests/test_model_runtime.py
python tests/test_langchain_tool_agent.py
```

Note:
- `test_langchain_tool_agent.py` skips automatically if model file is missing.

## Troubleshooting

- Error: `OPENAI_API_KEY is not set`
  - Set `OPENAI_API_KEY` in your shell or `.env`.

- Error: `Model file not found`
  - Export the model from ML-LinearRegression.
  - Or set `RENT_AGENT_MODEL_PATH` to a valid pickle file.

- Error mentioning `incorrect_hostname`
  - Set `OPENAI_BASE_URL=https://us.api.openai.com/v1`.

- Error: model has wrong feature count
  - This agent supports only 5-feature models.
  - Re-export using the multi-feature CSV pipeline.

## Integration Notes

- Model training and export live in the ML-LinearRegression repository.
- Inference and conversational analysis live in this repository.
- This separation lets you retrain independently while keeping agent logic stable.

## Main Components

- Agent entrypoint and CLI: `main.py`
- LangChain agent + tool: `src/agent/rent_prediction_agent.py`
- Model runtime and validation: `src/agent/model_runtime.py`
