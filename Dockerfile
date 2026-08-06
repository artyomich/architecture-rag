FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY scripts/ ./scripts/
COPY knowledge_base/ ./knowledge_base/

# Рабочая директория
WORKDIR /app

# Команда по умолчанию
CMD ["python", "scripts/rag_bot.py"]