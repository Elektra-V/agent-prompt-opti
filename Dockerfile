# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV using pip (simplest method for Docker)
RUN pip install --no-cache-dir uv && \
    uv --version

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY config.py ./
COPY room_selector.py ./
COPY run_apo.py ./

# Install dependencies using UV (system-wide installation)
# Install agentlightning and its dependencies, including poml
RUN uv pip install --system agentlightning "openai>=1.100.0" poml

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the default command (using entrypoint script)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["run_apo.py"]

