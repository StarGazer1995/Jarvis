FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --group dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "chainlit", "run", "src/web/app.py", "-h", "0.0.0.0", "-p", "8000"]
