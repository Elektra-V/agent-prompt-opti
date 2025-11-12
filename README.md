# Room Selector Agent - Agent-lightning APO Demo

A minimal demo project demonstrating **Automatic Prompt Optimization (APO)** using the Agent-lightning framework. This project shows how to train an agent to automatically improve its prompt through reinforcement learning.

## Overview

The Room Selector Agent is an AI agent that selects meeting rooms based on requirements like capacity, whiteboard availability, and projector needs. The agent uses OpenAI's function calling capability to query room availability and make decisions.

**Key Features:**
- ✅ Agent with `@rollout` decorator
- ✅ OpenAI function calling/tool use
- ✅ APO (Automatic Prompt Optimization) training
- ✅ Automatic prompt improvement through training
- ✅ Uses UV package manager (as shown in official tutorial)

## Quick Start

### Prerequisites

- Python 3.10+ (Linux required - Agent-lightning doesn't support macOS natively)
- [UV](https://github.com/astral-sh/uv) package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY=your_openai_key_here
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ```

### Running the Demo

Simply run:
```bash
uv run python run_apo.py
```

That's it! The training will start automatically, just like in the official tutorial.

**Note:** AgentOps is disabled by default to avoid permission issues. The APO training works perfectly without it.

## Project Structure

```
multiagent/
├── room_selector.py      # Agent implementation with @rollout decorator
├── run_apo.py            # Training script with APO algorithm
├── config.py             # LLM provider configuration
├── pyproject.toml         # UV project dependencies
├── env.example           # Environment variables template
└── README.md             # This file
```

## How It Works

### The Agent (`room_selector.py`)

The `room_selector` function is decorated with `@agl.rollout` and:
- Receives a task with room requirements
- Uses a prompt template (which gets optimized by APO)
- Calls OpenAI's API with function calling enabled
- Uses the `get_rooms_and_availability` tool to query available rooms
- Makes a final room selection
- Returns a reward score (0.0 to 1.0) based on correctness

### The Training Script (`run_apo.py`)

The training script:
- Creates an APO algorithm instance
- Configures a Trainer with parallel runners
- Provides an initial baseline prompt template
- Loads training and validation datasets
- Runs `trainer.fit()` to optimize the prompt automatically

### APO Algorithm

The APO (Automatic Prompt Optimization) algorithm:
1. **Evaluate**: Runs rollouts with the current prompt to measure performance
2. **Critique**: Analyzes spans and generates a textual critique of the prompt using GPT-5-mini
3. **Rewrite**: Applies the critique to generate an improved prompt template

This cycle repeats, improving the prompt with each iteration.

## Key Concepts

- **Task**: A specific input/problem (e.g., "Find a room for 4 people with a whiteboard")
- **Rollout**: A complete execution of the agent solving a task
- **Span**: A single unit of work within a rollout (e.g., an LLM call, tool execution)
- **Prompt Template**: The reusable instruction that gets optimized by APO
- **Reward**: A score (0.0-1.0) indicating how well the agent performed

## Expected Results

With the provided hyperparameters and datasets, you should see:
- Initial baseline performance (validation accuracy)
- Gradual improvement as APO optimizes the prompt
- Final improved performance after training rounds

The training typically takes a few minutes depending on your API rate limits and the number of parallel runners.

## Package Management

This project uses [UV](https://github.com/astral-sh/uv) as the package manager, matching the official Agent-lightning tutorial.

### Adding Dependencies

To add a new dependency:
```bash
uv add package-name
```

### Updating Dependencies

To update all dependencies:
```bash
uv sync --upgrade
```

## Docker (Optional - for macOS users only)

If you're on macOS and need Linux compatibility, Docker is available but not required:

```bash
docker compose build
docker compose run --rm app
```

**Note:** The official tutorial uses `uv run` directly on Linux, which is simpler and faster.

## Learn More

- Agent-lightning: [Official documentation](https://microsoft.github.io/agent-lightning/)
- UV: [UV documentation](https://github.com/astral-sh/uv)
- OpenAI API: [OpenAI documentation](https://platform.openai.com/docs)
