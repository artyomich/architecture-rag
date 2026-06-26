# Тесты проекта Architecture RAG

## Структура

```
tests/
├── README.md              # Этот файл
├── conftest.py            # Глобальные fixtures pytest
├── __init__.py
├── malicious_doc.txt      # Тестовый злонамеренный документ
├── unit/                  # Юнит-тесты
│   ├── __init__.py
│   ├── test_rag_bot.py    # Тесты rag_bot.py
│   ├── test_build_index.py # Тесты build_index.py
│   └── test_articles_list.py # Тесты articles_list.py
└── integration/           # Интеграционные тесты (в разработке)
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск тестов

### Все юнит-тесты:
```bash
pytest tests/unit/ -v
```

### С покрытием кода:
```bash
pytest tests/unit/ -v --cov=scripts --cov-report=html
```

### Только конкретный файл:
```bash
pytest tests/unit/test_rag_bot.py -v
```

### Только конкретный класс:
```bash
pytest tests/unit/test_rag_bot.py::TestSanitizeChunk -v
```

### С выводом локальных переменных при ошибке:
```bash
pytest tests/unit/ -v --tb=long
```

## Описание тестов

### test_rag_bot.py
- **TestSanitizeChunk** — тесты функции очистки контента от prompt-инъекций
  - Проверка удаления "ignore all instructions"
  - Проверка удаления "ignore previous"
  - Проверка удаления "disregard all"
  - Проверка маскирования паролей, ключей, секретов
  - Проверка обработки пустого ввода
- **TestIsResponseSafe** — тесты проверки безопасности ответа
  - Проверка безопасных ответов
  - Проверка обнаружения опасных паттернов (swordfish, password, root, superpassword, secret_key)
- **TestFormatDocs** — тесты форматирования документов
  - Проверка форматирования одиночного/множественных документов
  - Проверка обработки пустых списков
  - Проверка использования basename для источников
- **TestConfiguration** — тесты конфигурации
  - Проверка FEW_SHOT_EXAMPLES
  - Проверка SYSTEM_PROMPT
- **TestPatterns** — тесты регулярных выражений
  - Проверка валидности regex-паттернов

### test_build_index.py
- **TestBuildIndexPaths** — тесты путей
- **TestKnowledgeBaseValidation** — тесты проверки knowledge base
- **TestDocumentLoading** — тесты загрузки документов
- **TestChunkConfiguration** — тесты конфигурации чанков
- **TestEmbeddingConfiguration** — тесты конфигурации эмбеддингов
- **TestVectorstorePersistence** — тесты персистентности векторного хранилища
- **TestBuildIndexIntegration** — интеграционные тесты

### test_articles_list.py
- **TestArticlesList** — тесты списка статей
- **TestArticlesNamingConvention** — тесты именовании
- **TestArticlesCoverage** — тесты покрытия вселенной Матрицы
- **TestArticlesKnowledgeBaseAlignment** — тесты соответствия с knowledge base

## Запуск тестов в Docker

```bash
docker-compose run --rm app pytest tests/unit/ -v
```

## CI/CD

Для интеграции с CI/CD добавьте в pipeline:

```yaml
test:
  script:
    - pytest tests/unit/ -v --cov=scripts --cov-report=xml