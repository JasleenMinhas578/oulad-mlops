FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
COPY pipelines/ pipelines/
RUN useradd --create-home appuser
USER appuser
CMD ["python", "-m", "pipelines.flow"]
