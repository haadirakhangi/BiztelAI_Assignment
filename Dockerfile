FROM python:3.11-slim-bullseye AS base
ENV PYTHONDONTWRITEBYTECODE 1  
ENV PYTHONUNBUFFERED 1 

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]