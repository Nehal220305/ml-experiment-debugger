FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files
COPY pyproject.toml .
COPY server/requirements.txt .

# Install dependencies using uv
RUN uv pip install --system --no-cache \
    openenv-core \
    fastapi \
    uvicorn \
    scikit-learn \
    numpy \
    pydantic \
    openai \
    groq \
    requests

# Install PyTorch CPU
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]