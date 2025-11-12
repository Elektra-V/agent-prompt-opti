#!/bin/bash
# Entrypoint script to ensure log directory exists with proper permissions
# Works without root privileges by using /tmp as fallback

# Try to create log directory in /app (if mounted volume allows)
if mkdir -p /app/.agentops 2>/dev/null && [ -w /app/.agentops ] 2>/dev/null; then
    export AGENTOPS_LOG_DIR=/app/.agentops
else
    # Fallback to /tmp if /app is not writable (no root needed)
    export AGENTOPS_LOG_DIR=/tmp/agentops-$$  # Use process ID to avoid conflicts
    mkdir -p "$AGENTOPS_LOG_DIR" 2>/dev/null || true
fi

# Run the Python script
exec python "$@"

