FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv sync --no-dev
COPY . .
ENV PROJECT_ROOT=/app/data
RUN mkdir -p /app/data/workspace/generated /app/data/workspace/intermediate /app/data/workspace/artifacts
ENTRYPOINT ["uv", "run", "python", "inference.py"]
