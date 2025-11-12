# Room Selector Agent - Agent-lightning APO Demo

A minimal demo project demonstrating **Automatic Prompt Optimization (APO)** using the Agent-lightning framework. This project shows how to train an agent to automatically improve its prompt through reinforcement learning.

## Overview

The Room Selector Agent is an AI agent that selects meeting rooms based on requirements like capacity, whiteboard availability, and projector needs. The agent uses OpenAI's function calling capability to query room availability and make decisions.

**Key Features:**
- ✅ Agent with `@rollout` decorator
- ✅ OpenAI function calling/tool use
- ✅ APO (Automatic Prompt Optimization) training
- ✅ Automatic prompt improvement through training
- ✅ Containerized with Docker + UV package manager
- ✅ Linux-compatible (runs on macOS via Docker)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Running the Demo

1. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY=your_openai_key_here
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ```

2. **Build and run with Docker**:
   ```bash
   docker compose build
   docker compose run --rm app
   ```

That's it! The training will start automatically.

**Note for Cluster Environments:** This setup works without root privileges. If you encounter permission issues with AgentOps logging, you can disable it by setting `AGENTOPS_DISABLED=true` in your `.env` file or environment.

## Project Structure

```
multiagent/
├── room_selector.py      # Agent implementation with @rollout decorator
├── run_apo.py            # Training script with APO algorithm
├── config.py             # LLM provider configuration
├── Dockerfile            # Container configuration
├── docker-compose.yaml   # Docker Compose setup
├── pyproject.toml        # UV project dependencies
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

## Requirements

- **OpenAI API Key**: Required for both the agent and APO algorithm
- **Docker**: For running on macOS (Agent-lightning requires Linux)
- **UV**: Package manager (installed automatically in Docker)

## Package Management

This project uses [UV](https://github.com/astral-sh/uv) as the package manager for fast and reliable dependency management. Dependencies are defined in `pyproject.toml`.

### Adding Dependencies

To add a new dependency:
```bash
docker compose run --rm app uv add package-name
```

## Docker Usage

### Building the Image

```bash
docker compose build
```

### Running Commands

Run any Python command in the container:
```bash
docker compose run --rm app python your_script.py
```

### Interactive Shell

Get an interactive shell in the container:
```bash
docker compose run --rm app /bin/bash
```

## Learn More

- Agent-lightning: [Official documentation](https://microsoft.github.io/agent-lightning/)
- UV: [UV documentation](https://github.com/astral-sh/uv)
- OpenAI API: [OpenAI documentation](https://platform.openai.com/docs)
