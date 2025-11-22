# Быстрый старт после обновлений

## Что нового

Проект был значительно улучшен! Добавлены:

- **Тестовая инфраструктура** - 30+ тестов с pytest
- **CI/CD Pipeline** - автоматическое тестирование и проверка качества
- **Docker** - полная контейнеризация с Prometheus, Grafana, MLflow
- **Линтеры** - black, isort, flake8, mypy, ruff
- **Makefile** - удобные команды для разработки

## Быстрый старт

### 1. Установка зависимостей

```bash
# Установить все зависимости для разработки
make install-dev

# Или вручную:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### 2. Запуск тестов

```bash
# Запустить все тесты
make test

# Только unit тесты
make test-unit

# Только integration тесты
make test-integration
```

### 3. Проверка качества кода

```bash
# Проверить код линтерами
make lint

# Автоматически отформатировать код
make format

# Полная проверка (lint + test + security)
make check
```

### 4. Запуск с Docker

```bash
# Собрать образы
make docker-build

# Запустить все сервисы
make docker-up

# Просмотр логов
make docker-logs

# Остановить
make docker-down
```

После запуска доступны:
- **API**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **MLflow**: http://localhost:5001

## Документация

- [README.md](README.md) - основная документация
- [DOCKER.md](DOCKER.md) - руководство по Docker
- [project_analysis.md](.gemini/antigravity/brain/*/project_analysis.md) - анализ проекта
- [critical_fixes_summary.md](.gemini/antigravity/brain/*/critical_fixes_summary.md) - что было исправлено

## Полезные команды

```bash
# Разработка
make install-dev          # Установка dev зависимостей
make test                 # Запуск тестов
make lint                 # Проверка кода
make format               # Форматирование кода
make clean                # Очистка временных файлов

# Docker
make docker-build         # Сборка образа
make docker-up            # Запуск контейнеров
make docker-down          # Остановка
make docker-logs          # Просмотр логов
make docker-shell         # Подключение к контейнеру

# Утилиты
make security             # Проверка безопасности
make pre-commit           # Запуск pre-commit
make check                # Полная проверка
make help                 # Показать все команды
```

## Структура тестов

```
tests/
├── unit/                          # Unit тесты
│   ├── test_latency_predictor.py  # 25+ тестов для LatencyPredictor
│   ├── test_jitter_predictor.py   # Тесты для JitterPredictor
│   └── test_model_registry.py     # Тесты для ModelRegistry
├── integration/                   # Integration тесты
│   └── test_quic_test_integration.py
├── e2e/                          # End-to-end тесты
└── conftest.py                   # Общие fixtures
```

## Конфигурация

Основные конфигурационные файлы:

- `pytest.ini` - настройки pytest
- `pyproject.toml` - настройки линтеров и форматтеров
- `.pre-commit-config.yaml` - pre-commit hooks
- `docker-compose.yml` - Docker сервисы
- `Makefile` - команды автоматизации

## Следующие шаги

1. **Запустить тесты**: `make test`
2. **Проверить Docker**: `make docker-up`
3. **Начать разработку** с уверенностью в качестве кода!

## Советы

- Используйте `make help` для просмотра всех доступных команд
- Перед коммитом запускайте `make check`
- Pre-commit hooks автоматически проверят код перед коммитом
- CI/CD автоматически запустится при push в GitHub

## Помощь

Если что-то не работает:

1. Проверьте, что установлены все зависимости: `make install-dev`
2. Очистите временные файлы: `make clean`
3. Посмотрите логи Docker: `make docker-logs`
4. Обратитесь к документации в `DOCKER.md`

---

**Готово к работе!**
