# AI Routing Lab

**Прогнозирование оптимальных маршрутов с использованием машинного обучения для оптимизации задержки и джиттера**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-green)]()

**Доступные языки:**
- **English:** [README_en.md](README_en.md)
- **Русский:** Этот документ (README.md)

**Язык:** Русский
**Уровень:** Магистратура, PhD исследования
**Домен:** Машинное обучение, Оптимизация маршрутизации, Сетевые технологии
**Организация:** АНО "Центр исследований и разработок сетевых технологий"

---

## Обзор проекта

AI Routing Lab — это исследовательский проект, сосредоточенный на разработке моделей машинного обучения для прогнозирования оптимальных маршрутов в сетевой инфраструктуре CloudBridge. Проект направлен на достижение **точности >92%** при прогнозировании задержки и джиттера для оптимального выбора маршрутов.

**Ключевые задачи:**
- Прогнозирование оптимальных маршрутов на основе прогнозирования задержки/джиттера
- Интеграция с [quic-test](https://github.com/twogc/quic-test) для валидации моделей на реальном QUIC трафике
- Интеграция в производство с CloudBridge Relay для оптимизации маршрутов в реальном времени

---

## Об организации

### АНО "Центр исследований и разработок сетевых технологий"

Автономная некоммерческая организация создана в целях:
- Проведения фундаментальных и прикладных исследований в области сетевых протоколов (QUIC, MASQUE, BGP и других)
- Разработки и распространения свободного программного обеспечения (Open Source)
- Образования и повышения квалификации специалистов в области сетевых технологий
- Сотрудничества с ведущими российскими и международными вузами
- Подготовки высококвалифицированных кадров для индустрии

**Больше информации:**
- Официальный веб-сайт: https://cloudbridge-research.ru/
- Email: info@cloudbridge-research.ru

---

## Цели исследования

### Основная цель

Разработать модели машинного обучения, которые могут предсказывать задержку и джиттер маршрутов с **точностью >92%** для обеспечения проактивного выбора маршрутов в сети CloudBridge.

### Области исследования

1. **Прогнозирование задержки (Latency Prediction)**
   - Прогнозирование временных рядов задержки маршрутов
   - Сравнение задержек на множественных путях
   - Анализ исторических паттернов

2. **Прогнозирование джиттера (Jitter Prediction)**
   - Моделирование изменчивости джиттера
   - Анализ влияния условий сети
   - Оценка стабильности маршрутов

3. **Оптимизация выбора маршрутов (Route Selection Optimization)**
   - Ансамблевые модели для ранжирования маршрутов
   - Инференция прогнозов в реальном времени
   - Интеграция с CloudBridge Relay

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│           AI Routing Lab (Python)                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Сбор данных (Data Collection)                   │   │
│  │  • Метрики Prometheus из quic-test               │   │
│  │  • JSON экспорт из quic-test                     │   │
│  │  • Хранение исторических данных                  │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ML Pipeline (Конвейер обучения)                 │   │
│  │  • LatencyPredictor (Random Forest)              │   │
│  │  • JitterPredictor (Random Forest)               │   │
│  │  • RoutePredictionEnsemble                       │   │
│  │  • Feature Engineering (Инженерия признаков)     │   │
│  │  • Model Evaluation (Оценка моделей)             │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Inference Engine (Движок инференции)            │   │
│  │  • Real-time Predictions (Прогнозы в реальном)   │   │
│  │  • Route Optimization (Оптимизация маршрутов)    │   │
│  │  • API для CloudBridge Relay                     │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          │ Валидация (Validation)       │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  quic-test (Go)                                  │   │
│  │  • Генерация реального QUIC трафика              │   │
│  │  • Сбор метрик                                   │   │
│  │  • Валидация ML прогнозов                        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Структура проекта

```
ai-routing-lab/
├── README.md                    # Основная документация (русский)
├── README_en.md                 # Документация (английский)
├── QUICKSTART.md                # Быстрый старт
├── DOCKER.md                    # Руководство по Docker
├── LICENSE                      # Лицензия MIT
├── requirements.txt             # Зависимости Python
├── requirements-dev.txt         # Dev зависимости
├── pyproject.toml               # Конфигурация проекта
├── pytest.ini                   # Настройки pytest
├── Makefile                     # Команды автоматизации
├── Dockerfile                   # Docker образ
├── docker-compose.yml           # Docker Compose конфигурация
├── setup.py                     # Конфигурация пакета
│
├── data/                        # Сбор и обработка данных
│   ├── collectors/
│   │   └── quic_test_collector.py    # Интеграция с quic-test
│   └── pipelines/
│       └── data_pipeline.py     # Pipeline обработки данных
│
├── models/                      # Определения ML моделей
│   ├── core/                    # Основная ML инфраструктура
│   │   ├── model_registry.py    # Версионирование моделей
│   │   ├── data_preprocessor.py  # Предобработка данных
│   │   ├── feature_extractor.py # Инженерия признаков
│   │   └── model_validator.py   # Валидация моделей
│   ├── prediction/              # Модели прогнозирования
│   │   ├── latency_predictor.py # Прогнозирование задержки
│   │   ├── jitter_predictor.py  # Прогнозирование джиттера
│   │   └── route_prediction_ensemble.py # Выбор маршрутов
│   ├── routing/                 # Модели оптимизации маршрутов
│   ├── anomaly/                 # Обнаружение аномалий
│   └── monitoring/              # Мониторинг моделей
│
├── training/                    # Скрипты обучения
│   ├── train_latency_model.py   # Обучение latency модели
│   └── train_jitter_model.py    # Обучение jitter модели
│
├── inference/                   # Движок инференции
│   └── predictor_service.py     # FastAPI сервис
│
├── evaluation/                  # Оценка моделей
│   └── model_evaluator.py       # Оценка и валидация
│
├── experiments/                 # Лабораторные эксперименты
│   ├── lab_experiment.py       # Фреймворк экспериментов
│   ├── example_experiment.py   # Пример эксперимента
│   └── latency_jitter_experiment.py # Полный workflow
│
├── tests/                       # Тесты
│   ├── unit/                    # Unit тесты
│   ├── integration/             # Integration тесты
│   └── e2e/                     # End-to-end тесты
│
├── monitoring/                  # Мониторинг
│   └── prometheus.yml           # Конфигурация Prometheus
│
└── docs/                        # Документация
    ├── architecture/            # Архитектура
    ├── development/             # Разработка
    └── guides/                  # Руководства
```

---

## Быстрый старт

### Предварительные требования

- Python 3.11+
- [quic-test](https://github.com/twogc/quic-test) запущен и экспортирует метрики
- Prometheus (опционально, для сбора метрик)

### Установка

#### Вариант 1: С использованием Makefile (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Установка всех зависимостей для разработки
make install-dev
```

#### Вариант 2: Ручная установка

```bash
# Клонирование репозитория
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Установка пакета в режиме разработки
pip install -e .

# Установка pre-commit hooks
pre-commit install
```

### Базовое использование

#### Обучение моделей

```bash
# Обучение latency модели
python training/train_latency_model.py \
  --data-path data/training_data.json \
  --model-output models/ \
  --n-estimators 100

# Обучение jitter модели
python training/train_jitter_model.py \
  --data-path data/training_data.json \
  --model-output models/
```

#### Запуск inference сервиса

```bash
# Запуск FastAPI сервиса
python -m uvicorn inference.predictor_service:app --host 0.0.0.0 --port 5000

# Или через Docker
make docker-up
```

#### Запуск экспериментов

```bash
# Запуск примера эксперимента прогнозирования задержки
python experiments/example_experiment.py

# Запуск полного эксперимента задержка/джиттер
python experiments/latency_jitter_experiment.py
```

#### Сбор данных из quic-test

```bash
# Сбор метрик из Prometheus
python -m data.collectors.quic_test_collector --prometheus-url http://localhost:9090
```

### Тестирование

```bash
# Запуск всех тестов
make test

# Только unit тесты
make test-unit

# С coverage отчетом
pytest --cov=. --cov-report=html
```

### Docker

```bash
# Сборка образа
make docker-build

# Запуск всех сервисов (API, Prometheus, Grafana, MLflow)
make docker-up

# Просмотр логов
make docker-logs

# Остановка
make docker-down
```

Подробнее см. [DOCKER.md](DOCKER.md) и [QUICKSTART.md](QUICKSTART.md).

---

## Модели

### LatencyPredictor

Модель Random Forest для прогнозирования задержки маршрутов.

**Особенности:**
- Исторические паттерны задержки
- Характеристики маршрутов (расположение PoP, BGP пути)
- Условия сети (перегруженность, потеря пакетов)
- Временные признаки (временные характеристики)

**Цель:** Точность >92% (R² метрика)

### JitterPredictor

Модель Random Forest для прогнозирования изменчивости джиттера маршрутов.

**Особенности:**
- Исторические паттерны джиттера
- Метрики стабильности маршрутов
- Индикаторы изменчивости сети

**Цель:** Точность >92% (R² метрика)

### RoutePredictionEnsemble

Комбинирует прогнозы задержки и джиттера для оптимального выбора маршрутов.

**Скоринг:**
- Вес задержки: 70%
- Вес джиттера: 30%
- Выбирает маршрут с наилучшей объединенной оценкой

**Цель:** >95% оптимального выбора маршрутов

---

## Интеграция с quic-test

AI Routing Lab интегрируется с [quic-test](https://github.com/twogc/quic-test) для обеспечения полного цикла разработки ML моделей. **Полная документация по интеграции доступна в [docs/QUIC_TEST_INTEGRATION_RU.md](docs/QUIC_TEST_INTEGRATION_RU.md).**

Ключевые возможности интеграции:

1. **Сбор данных (Data Collection):**
   - Экспорт метрик Prometheus из quic-test
   - JSON экспорт для исторических данных
   - Потоковая передача метрик в реальном времени

2. **Валидация моделей (Model Validation):**
   - Валидация ML прогнозов на реальном QUIC трафике
   - Сравнение прогнозируемых vs фактических значений задержки/джиттера
   - Расчет метрик точности прогнозирования

3. **Тестирование в производстве (Production Testing):**
   - Тестирование выбора маршрутов в контролируемой среде
   - Фреймворк A/B тестирования
   - Бенчмарки производительности

### Настройка интеграции

1. **Запуск quic-test с экспортом Prometheus:**
   ```bash
   cd cloudbridge/quic-test
   ./bin/quic-server --prometheus-port 9090
   ```

2. **Сбор метрик:**
   ```python
   from data.collectors.quic_test_collector import PrometheusCollector

   collector = PrometheusCollector(prometheus_url="http://localhost:9090")
   metrics = collector.collect_all_metrics()
   ```

---

## Лабораторные эксперименты

Проект включает комплексный фреймворк лабораторных экспериментов:

```python
from experiments.lab_experiment import create_latency_prediction_experiment
from models.prediction import LatencyPredictor

# Создание эксперимента
lab = create_latency_prediction_experiment()

# Подготовка данных
X_train_proc, y_train_proc, _ = lab.prepare_data(X_train, y_train)

# Обучение модели
model = LatencyPredictor(n_estimators=100, max_depth=15)
model.fit(X_train_proc, y_train_proc)

# Оценка
metrics = model.evaluate(X_test_proc, y_test_proc)
print(f"Точность (R²): {metrics['r2_score']:.4f}")
```

Более подробную документацию см. в `experiments/README.md`.

---

## Команды Makefile

Проект включает Makefile с удобными командами для разработки:

```bash
# Показать все доступные команды
make help

# Установка
make install          # Установить зависимости
make install-dev      # Установить dev зависимости + pre-commit

# Тестирование
make test             # Запустить все тесты с coverage
make test-unit        # Только unit тесты
make test-integration # Только integration тесты

# Качество кода
make lint             # Проверить код линтерами
make format           # Автоматически отформатировать код
make check            # Полная проверка (lint + test + security)

# Docker
make docker-build     # Собрать Docker образ
make docker-up        # Запустить все сервисы
make docker-down      # Остановить контейнеры
make docker-logs      # Просмотр логов
make docker-shell     # Подключиться к контейнеру

# Обучение моделей
make train-latency    # Обучить latency модель
make train-jitter     # Обучить jitter модель

# Утилиты
make clean            # Очистить временные файлы
make security         # Проверка безопасности
```

Подробнее см. [QUICKSTART.md](QUICKSTART.md).

---

## Документация

### Быстрый старт
- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт после обновлений
- [DOCKER.md](DOCKER.md) - Руководство по Docker deployment

### Архитектура
- [Документация архитектуры](docs/architecture/ARCHITECTURE.md)
- [Руководство интеграции](docs/architecture/INTEGRATION_GUIDE.md)

### Руководства
- [Примеры использования](docs/guides/USAGE_EXAMPLES.md)
- [Руководство по вкладу](docs/guides/CONTRIBUTING.md)

### Разработка
- [Статус адаптации](docs/development/ADAPTATION_STATUS.md)
- [Заметки адаптации](docs/development/ADAPTATION_NOTES.md)
- [Резюме обновления](docs/development/UPGRADE_SUMMARY.md)

### Эксперименты
- [Лабораторные эксперименты](experiments/README.md)

### Отчеты
- [Лабораторные отчеты](reports/README.md) - Отчеты об испытаниях, организованные по датам и версиям

---

## Стек технологий

- **Язык:** Python 3.11+
- **ML фреймворк:** scikit-learn (Random Forest), TensorFlow/PyTorch (опционально)
- **Отслеживание экспериментов:** MLflow
- **Обработка данных:** pandas, numpy
- **Сбор метрик:** prometheus-client
- **API:** FastAPI / gRPC

---

## Интеграция с CloudBridge

### CloudBridge Relay

AI Routing Lab интегрируется с [CloudBridge Relay](https://github.com/twogc/cloudbridge-scalable-relay) для:

- **Использование прогнозов в реальном времени:** Встроенные предсказания используются для выбора оптимального маршрута при маршрутизации пакетов
- **Оптимизация сетевых путей:** Динамическая адаптация маршрутов на основе предсказанной задержки и стабильности
- **Мониторинг и обратная связь:** Сравнение прогнозов с реальными результатами для постоянного улучшения моделей

### CloudBridge Research

Этот проект является частью инициативы CloudBridge Research Center:

- **Научные исследования:** Разработка новых подходов к оптимизации маршрутизации
- **Образовательные программы:** Использование в курсах машинного обучения и сетевых технологий
- **Open Source публикации:** Результаты исследований доступны как открытый исходный код

---

## Рабочий процесс интеграции

### Фаза 1: Сбор данных

```bash
# Запуск quic-test для сбора метрик
./bin/quic-test --mode=test --network-profile=mobile --duration=600 --report=metrics.json

# Сбор метрик из различных сетевых условий
for profile in excellent good poor mobile satellite adversarial; do
  ./bin/quic-test --mode=test --network-profile=$profile --duration=600 --report=metrics_${profile}.json
done
```

### Фаза 2: Подготовка данных

```python
from models.core.data_preprocessor import DataPreprocessor
from data.pipelines.data_pipeline import DataPipeline
import numpy as np
import json

# Загрузка данных
with open('metrics.json', 'r') as f:
    data = json.load(f)

# Извлечение features
X = np.array([item['features'] for item in data])

# Создание pipeline для предобработки
pipeline = DataPipeline()
pipeline.add_preprocessing_stage(strategy='mean', normalization='standard')
pipeline.add_feature_extraction_stage()

# Обработка данных
result = pipeline.process(X)
X_processed = result['data']
```

### Фаза 3: Обучение моделей

```bash
# Обучение LatencyPredictor
python training/train_latency_model.py \
  --data-path data/preprocessed_metrics.json \
  --model-output models/ \
  --n-estimators 100

# Обучение JitterPredictor
python training/train_jitter_model.py \
  --data-path data/preprocessed_metrics.json \
  --model-output models/

# Модели автоматически сохраняются в ModelRegistry
```

### Фаза 4: Валидация моделей

```python
from evaluation.model_evaluator import ModelEvaluator
from models.core.model_registry import ModelRegistry
import numpy as np

# Загрузка модели из registry
registry = ModelRegistry(models_dir="models/")
model, metadata = registry.get_model("latency_predictor")

# Оценка модели
evaluator = ModelEvaluator()
result = evaluator.evaluate(model, X_test, y_test, use_cross_validation=True)

# Генерация отчета
report = evaluator.generate_report(result, model_name="LatencyPredictor")
print(report)
```

### Фаза 5: Развертывание в производство

```bash
# Запуск inference сервиса (FastAPI)
python -m uvicorn inference.predictor_service:app --host 0.0.0.0 --port 5000

# Или через Docker
make docker-up

# API будет доступен на http://localhost:5000
# Endpoints:
#   GET  /health - проверка здоровья
#   GET  /models - список моделей
#   GET  /metrics - Prometheus метрики
#   POST /predict - предсказание для одного маршрута
#   POST /predict/routes - сравнение нескольких маршрутов
```

---

## Примеры использования

### Прогнозирование задержки для одного маршрута

```python
from models.prediction import LatencyPredictor
from models.core.model_registry import ModelRegistry
import numpy as np

# Создание и обучение модели
model = LatencyPredictor(n_estimators=100, max_depth=15)

# Подготовка данных
X_train = np.array([
    [25.5, 2.3, 0.95, 1.0],
    [30.1, 3.1, 0.85, 1.2],
    # ... больше данных
])
y_train = np.array([26.0, 31.0, ...])  # Фактические значения задержки

# Обучение
model.fit(X_train, y_train)

# Подготовка признаков для предсказания
features = np.array([[25.5, 2.3, 0.95, 1.0]])

# Прогнозирование
prediction = model.predict(features)
print(f"Прогнозируемая задержка: {prediction.predicted_latency_ms:.2f} мс")
print(f"Доверительный интервал: {prediction.confidence_interval}")
print(f"Уверенность: {prediction.confidence_score:.2%}")

# Сохранение модели через ModelRegistry
registry = ModelRegistry(models_dir="models/")
registry.register_model(
    model_id="latency_predictor",
    model=model,
    model_type="prediction",
    accuracy=0.95,
    framework="scikit-learn"
)
```

### Выбор оптимального маршрута из множества

```python
from models.prediction import LatencyPredictor, JitterPredictor, RoutePredictionEnsemble
import numpy as np

# Создание и обучение моделей
latency_model = LatencyPredictor()
jitter_model = JitterPredictor()

# Обучение на данных
X_train = np.random.randn(100, 4)
y_latency = np.random.randn(100) * 10 + 25
y_jitter = np.random.randn(100) * 2 + 2

latency_model.fit(X_train, y_latency)
jitter_model.fit(X_train, y_jitter)

# Создание ансамбля
ensemble = RoutePredictionEnsemble(
    latency_model=latency_model,
    jitter_model=jitter_model
)

# Данные о нескольких маршрутах (объединенные features)
routes_features = {
    'route_0': np.array([[25.5, 2.3, 0.95, 1.0]]),
    'route_1': np.array([[30.1, 3.1, 0.85, 1.2]]),
    'route_2': np.array([[20.3, 1.8, 0.98, 0.8]]),
}

# Прогнозирование и выбор оптимального маршрута
best_route, predictions = ensemble.select_best_route(routes_features)
print(f"Оптимальный маршрут: {best_route}")
```

### Использование через REST API

```python
import requests

# Предсказание для одного маршрута
response = requests.post('http://localhost:5000/predict', json={
    'features': [25.5, 2.3, 0.95, 1.0],
    'route_id': 'route_0'
})
print(response.json())

# Сравнение нескольких маршрутов
response = requests.post('http://localhost:5000/predict/routes', json={
    'routes': {
        'route_0': [25.5, 2.3, 0.95, 1.0],
        'route_1': [30.1, 3.1, 0.85, 1.2],
        'route_2': [20.3, 1.8, 0.98, 0.8]
    }
})
result = response.json()
print(f"Лучший маршрут: {result['best_route']}")
print(f"Ранжирование: {result['ranking']}")
```

---

## Результаты исследований

### Текущая точность моделей

Точность моделей зависит от качества и количества обучающих данных. При обучении на реальных данных из quic-test ожидаются следующие результаты:

| Модель | Целевой R² Score | Ожидаемый MAE (мс) | Ожидаемый RMSE (мс) | Статус |
|--------|------------------|-------------------|---------------------|--------|
| LatencyPredictor | >0.92 | <3.0 | <4.0 | В разработке |
| JitterPredictor | >0.92 | <1.5 | <2.0 | В разработке |
| RoutePredictionEnsemble | >0.95 | - | - | В разработке |

### Улучшение производительности сети

Ожидаемые улучшения при использовании ML выбора маршрутов:
- **Уменьшение средней задержки:** На 15-20%
- **Уменьшение пиковых задержек:** На 25-30% в условиях перегруженной сети
- **Повышение стабильности:** На 40% (снижение джиттера)

---

## Лицензия

Этот проект лицензирован под лицензией MIT - см. файл [LICENSE](LICENSE) для деталей.

---

## Связанные проекты

- [quic-test](https://github.com/twogc/quic-test) - Инструмент тестирования QUIC протокола

---

## Контакты

- **GitHub:** [@twogc](https://github.com/twogc)
- **Email:** info@cloudbridge-research.ru
- **Веб-сайт:** [cloudbridge-research.ru](https://cloudbridge-research.ru/)
- **Slack:** #ai-routing-support (для членов CloudBridge Research)

---

## Благодарности

Модели и инфраструктура адаптированы из экосистемы 2GC CloudBridge Global Network.

---

## Инфраструктура разработки

### Тестирование

Проект включает комплексную тестовую инфраструктуру:
- **62 unit тестов** для основных компонентов
- **Integration тесты** для интеграции с quic-test
- **E2E тесты** для полного workflow
- **Coverage:** 22.73% (цель: 70%+)

### CI/CD

Автоматизированный pipeline включает:
- Автоматический запуск тестов на Python 3.11 и 3.12
- Проверка качества кода (black, isort, flake8, mypy, ruff)
- Security scanning (bandit, safety)
- Coverage reporting

### Инструменты разработки

- **Makefile** - удобные команды для всех операций
- **Pre-commit hooks** - автоматическая проверка перед коммитом
- **Docker** - полная контейнеризация с Prometheus, Grafana, MLflow
- **Линтеры и форматтеры** - поддержание качества кода

Подробнее см. [QUICKSTART.md](QUICKSTART.md) и [Makefile](Makefile).

---

**Статус:** В активной разработке
**Последнее обновление:** Ноябрь 2025
**Версия:** 0.2.1
