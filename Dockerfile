FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app" 
    
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md* /app/
RUN poetry install --no-root -v

COPY src /app/src
COPY configs /app/configs

RUN mkdir -p /app/data /app/Experiments /app/mlruns

CMD ["python", "src/train/run.py"]
