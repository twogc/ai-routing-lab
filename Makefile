.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down

# Цвета для вывода
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Показать эту справку
	@echo "$(BLUE)AI Routing Lab - Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "$(BLUE)Установка зависимостей...$(NC)"
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Установить зависимости для разработки
	@echo "$(BLUE)Установка dev зависимостей...$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test: ## Запустить тесты
	@echo "$(BLUE)Запуск тестов...$(NC)"
	pytest --cov=. --cov-report=term-missing --cov-report=html -v

test-unit: ## Запустить только unit тесты
	@echo "$(BLUE)Запуск unit тестов...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Запустить только integration тесты
	@echo "$(BLUE)Запуск integration тестов...$(NC)"
	pytest tests/integration/ -v

lint: ## Проверить качество кода
	@echo "$(BLUE)Проверка качества кода...$(NC)"
	black --check . --exclude venv
	isort --check-only . --skip venv
	flake8 . --exclude=venv,htmlcov,.git,__pycache__ || true
	mypy . --ignore-missing-imports || true
	ruff check . --exclude venv

format: ## Форматировать код
	@echo "$(BLUE)Форматирование кода...$(NC)"
	black .
	isort .
	ruff check --fix .

clean: ## Очистить временные файлы
	@echo "$(BLUE)Очистка временных файлов...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

docker-build: ## Собрать Docker образ
	@echo "$(BLUE)Сборка Docker образа...$(NC)"
	docker-compose build

docker-up: ## Запустить Docker контейнеры
	@echo "$(BLUE)Запуск Docker контейнеров...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Сервисы запущены:$(NC)"
	@echo "  - API: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - MLflow: http://localhost:5001"

docker-down: ## Остановить Docker контейнеры
	@echo "$(BLUE)Остановка Docker контейнеров...$(NC)"
	docker-compose down

docker-logs: ## Показать логи Docker контейнеров
	docker-compose logs -f ai-routing-lab

docker-shell: ## Подключиться к контейнеру
	docker-compose exec ai-routing-lab bash

# Сбор данных
collect-data: ## Собрать данные из quic-test
	@echo "$(BLUE)Сбор данных из quic-test...$(NC)"
	python -m data.collectors.quic_test_collector --prometheus-url http://localhost:9090

# Обучение моделей
train-latency: ## Обучить модель предсказания задержки
	@echo "$(BLUE)Обучение модели latency...$(NC)"
	python -m training.train_latency_model

train-jitter: ## Обучить модель предсказания джиттера
	@echo "$(BLUE)Обучение модели jitter...$(NC)"
	python -m training.train_jitter_model

# Запуск экспериментов
experiment: ## Запустить полный эксперимент
	@echo "$(BLUE)Запуск эксперимента...$(NC)"
	python experiments/latency_jitter_experiment.py

# Документация
docs: ## Сгенерировать документацию
	@echo "$(BLUE)Генерация документации...$(NC)"
	cd docs && make html

docs-serve: ## Запустить сервер документации
	@echo "$(BLUE)Запуск сервера документации...$(NC)"
	cd docs/_build/html && python -m http.server 8000

# Pre-commit
pre-commit: ## Запустить pre-commit проверки
	@echo "$(BLUE)Запуск pre-commit...$(NC)"
	pre-commit run --all-files

# Безопасность
security: ## Проверить безопасность
	@echo "$(BLUE)Проверка безопасности...$(NC)"
	bandit -r . || true
	safety check --file requirements.txt || true

# Полная проверка перед коммитом
check: lint test security ## Полная проверка (lint + test + security)
	@echo "$(GREEN)✓ Все проверки пройдены!$(NC)"
