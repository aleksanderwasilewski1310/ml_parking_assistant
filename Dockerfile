# syntax=docker/dockerfile:1

# --- Stage 1: Extract the official uv binary ---
FROM python:3.12-slim

# Optimize Python execution behavior inside the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl to fetch the uv installer, then install uv,
# and clean up apt lists immediately to keep the layer minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Add the installed uv binary to the system PATH so it can be called directly
ENV PATH="/root/.local/bin/:$PATH"

# Leverage Docker cache layers effectively:
# Copy only the compiled requirements file first.
COPY requirements.txt ./

# Install project dependencies into the global system environment using uv pip
# --system: Installs packages directly into the global Python environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# FastAPI application defaults to port 5000
EXPOSE 5000

# CRITICAL FIX: Create a system group and a non-root user WITH a home directory (-m)
# The home directory is essential for Kaggle library local configuration/cache storage.
RUN groupadd -r appgroup && useradd -m -r -g appgroup -s /bin/bash appuser

# CRITICAL FIX: Grant appuser ownership over BOTH the /app directory and its own home folder
RUN chown -R appuser:appgroup /app /home/appuser

# Switch the execution context to the secure non-root user
USER appuser

# Command to launch the Uvicorn ASGI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
