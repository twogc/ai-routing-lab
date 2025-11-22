# Docker Deployment Guide

## Быстрый старт

### 1. Сборка и запуск всех сервисов

```bash
# Сборка образов
docker-compose build

# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f ai-routing-lab
```

### 2. Проверка работоспособности

```bash
# Проверка статуса контейнеров
docker-compose ps

# Проверка health check
docker-compose exec ai-routing-lab curl http://localhost:5000/health

# Просмотр метрик Prometheus
open http://localhost:9090

# Открыть Grafana
open http://localhost:3000
# Логин: admin, Пароль: admin

# Открыть MLflow
open http://localhost:5001
```

## Доступные сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| AI Routing Lab API | 5000 | REST API для предсказаний |
| Prometheus | 9090 | Сбор метрик |
| Grafana | 3000 | Визуализация метрик |
| MLflow | 5001 | Tracking экспериментов |

## Управление сервисами

### Остановка сервисов

```bash
# Остановить все сервисы
docker-compose stop

# Остановить конкретный сервис
docker-compose stop ai-routing-lab
```

### Перезапуск сервисов

```bash
# Перезапустить все сервисы
docker-compose restart

# Перезапустить конкретный сервис
docker-compose restart ai-routing-lab
```

### Удаление контейнеров и volumes

```bash
# Остановить и удалить контейнеры
docker-compose down

# Удалить контейнеры и volumes
docker-compose down -v
```

## Разработка

### Запуск только AI Routing Lab

```bash
docker-compose up ai-routing-lab
```

### Пересборка после изменений кода

```bash
docker-compose build ai-routing-lab
docker-compose up -d ai-routing-lab
```

### Подключение к контейнеру

```bash
# Bash shell
docker-compose exec ai-routing-lab bash

# Python REPL
docker-compose exec ai-routing-lab python
```

## Volumes

Данные сохраняются в следующих volumes:

- `./models` - обученные модели
- `./data` - данные для обучения
- `./logs` - логи приложения
- `prometheus-data` - данные Prometheus
- `grafana-data` - конфигурация Grafana
- `mlflow-data` - эксперименты MLflow

## Переменные окружения

Создайте файл `.env` для настройки:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=5000

# Prometheus
PROMETHEUS_URL=http://prometheus:9090

# Models
MODELS_DIR=/app/models

# Logging
LOG_LEVEL=INFO
```

## Production Deployment

Для production используйте:

```bash
# Используйте production docker-compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Контейнер не запускается

```bash
# Проверить логи
docker-compose logs ai-routing-lab

# Проверить health check
docker inspect ai-routing-lab | grep Health
```

### Проблемы с зависимостями

```bash
# Пересобрать без кэша
docker-compose build --no-cache ai-routing-lab
```

### Очистка Docker

```bash
# Удалить неиспользуемые образы
docker image prune -a

# Удалить неиспользуемые volumes
docker volume prune
```
