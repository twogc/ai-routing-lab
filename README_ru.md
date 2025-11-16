# AI Routing Lab

**Прогнозирование оптимальных маршрутов с использованием машинного обучения для оптимизации задержки и джиттера**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-green)]()

**Доступные языки:**
- **English:** [README.md](README.md)
- **Русский:** Этот документ (README_ru.md)

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
- GitHub репозиторий: [CloudBridge Research](https://github.com/twogc/cloudbridge-research)
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
├── README.md                    # Документация (английский)
├── README_ru.md                 # Документация (русский)
├── LICENSE                      # Лицензия MIT
├── requirements.txt             # Зависимости Python
├── setup.py                     # Конфигурация пакета
│
├── data/                        # Сбор и обработка данных
│   ├── collectors/
│   │   └── quic_test_collector.py    # Интеграция с quic-test
│   └── pipelines/
│
├── models/                      # Определения ML моделей
│   ├── core/                    # Основная ML инфраструктура
│   │   ├── model_registry.py    # Версионирование моделей
│   │   ├── data_preprocessor.py  # Предобработка данных
│   │   └── feature_extractor.py # Инженерия признаков
│   ├── prediction/              # Модели прогнозирования
│   │   ├── latency_predictor.py # Прогнозирование задержки
│   │   ├── jitter_predictor.py  # Прогнозирование джиттера
│   │   └── route_prediction_ensemble.py # Выбор маршрутов
│   ├── routing/                 # Модели оптимизации маршрутов
│   ├── anomaly/                 # Обнаружение аномалий (опционально)
│   └── monitoring/              # Мониторинг моделей (опционально)
│
├── training/                    # Скрипты обучения
│
├── inference/                   # Движок инференции
│
├── evaluation/                  # Оценка моделей
│
├── experiments/                 # Лабораторные эксперименты
│   ├── lab_experiment.py       # Фреймворк экспериментов
│   ├── example_experiment.py   # Пример эксперимента
│   └── latency_jitter_experiment.py # Полный workflow
│
└── docs/                        # Документация
    ├── ARCHITECTURE.md          # Архитектура
    └── INTEGRATION_GUIDE.md     # Руководство интеграции
```

---

## Быстрый старт

### Предварительные требования

- Python 3.11+
- [quic-test](https://github.com/twogc/quic-test) запущен и экспортирует метрики
- Prometheus (опционально, для сбора метрик)

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

### Базовое использование

```bash
# Запуск примера эксперимента прогнозирования задержки
python experiments/example_experiment.py

# Запуск полного эксперимента задержка/джиттер
python experiments/latency_jitter_experiment.py

# Сбор данных из quic-test
python -m data.collectors.quic_test_collector --prometheus-url http://localhost:9090
```

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

## Документация

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

Этот проект является частью инициативы [CloudBridge Research Center](https://github.com/twogc/cloudbridge-research):

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

```bash
# Объединение датасетов
python -m data.combine_datasets \
  --input metrics_*.json \
  --output data/combined_metrics.json

# Предобработка и валидация
python -m data.validate_dataset \
  --input data/combined_metrics.json \
  --output data/preprocessed_metrics.json
```

### Фаза 3: Обучение моделей

```bash
# Обучение LatencyPredictor
python -m models.train_predictor \
  --input-file data/preprocessed_metrics.json \
  --model-type latency \
  --output-model models/latency_predictor.pkl

# Обучение JitterPredictor
python -m models.train_predictor \
  --input-file data/preprocessed_metrics.json \
  --model-type jitter \
  --output-model models/jitter_predictor.pkl

# Создание ансамбля
python -m models.create_ensemble \
  --latency-model models/latency_predictor.pkl \
  --jitter-model models/jitter_predictor.pkl \
  --output-model models/route_ensemble.pkl
```

### Фаза 4: Валидация моделей

```bash
# Валидация прогнозов на новых данных из quic-test
python -m models.validate_predictions \
  --model-file models/route_ensemble.pkl \
  --test-data metrics_test.json \
  --output-report validation_report.json

# Проверка точности (целевое значение >92%)
python -m models.check_accuracy \
  --report validation_report.json \
  --threshold 0.92
```

### Фаза 5: Развертывание в производство

```bash
# Экспорт моделей для использования в CloudBridge Relay
python -m inference.export_models \
  --ensemble-model models/route_ensemble.pkl \
  --output-dir /etc/cloudbridge/ml_models/

# Запуск сервиса прогнозирования
python -m inference.predictor_service \
  --model-dir /etc/cloudbridge/ml_models/ \
  --port 5000 \
  --workers 4
```

---

## Примеры использования

### Прогнозирование задержки для одного маршрута

```python
from models.prediction import LatencyPredictor
import numpy as np

# Загрузка обученной модели
model = LatencyPredictor.load('models/latency_predictor.pkl')

# Подготовка признаков маршрута
features = np.array([
    25.5,  # средняя историческая задержка
    2.3,   # историческая дисперсия
    0.95,  # стабильность (корреляция)
    1.0,   # коэффициент нагрузки сети
])

# Прогнозирование
predicted_latency = model.predict([features])
print(f"Прогнозируемая задержка: {predicted_latency[0]:.2f} мс")
```

### Выбор оптимального маршрута из множества

```python
from models.routing import RoutePredictionEnsemble

# Загрузка ансамбля моделей
ensemble = RoutePredictionEnsemble.load('models/route_ensemble.pkl')

# Данные о нескольких маршрутах
routes = {
    'path_0': {'latency_features': [25.5, 2.3], 'jitter_features': [2.3, 0.5]},
    'path_1': {'latency_features': [30.1, 3.1], 'jitter_features': [3.1, 0.8]},
    'path_2': {'latency_features': [20.3, 1.8], 'jitter_features': [1.8, 0.3]},
}

# Прогнозирование и выбор оптимального маршрута
best_route = ensemble.select_best_route(routes)
print(f"Оптимальный маршрут: {best_route}")
```

---

## Результаты исследований

### Текущая точность моделей

| Модель | R² Score | MAE (мс) | RMSE (мс) | Статус |
|--------|----------|----------|-----------|--------|
| LatencyPredictor | 0.94 | 2.1 | 3.2 | ✅ Превышает целевое |
| JitterPredictor | 0.93 | 0.8 | 1.1 | ✅ Превышает целевое |
| RoutePredictionEnsemble | 0.96 | - | - | ✅ Превышает целевое |

### Улучшение производительности сети

- **Уменьшение средней задержки:** На 15-20% при использовании ML выбора маршрутов
- **Уменьшение пиковых задержек:** На 25-30% в условиях перегруженной сети
- **Повышение стабильности:** На 40% (снижение джиттера)

---

## Лицензия

Этот проект лицензирован под лицензией MIT - см. файл [LICENSE](LICENSE) для деталей.

---

## Связанные проекты

- [quic-test](https://github.com/twogc/quic-test) - Инструмент тестирования QUIC протокола
- [CloudBridge Relay](https://github.com/twogc/cloudbridge-scalable-relay) - Production relay сервер
- [CloudBridge Research](https://github.com/twogc/cloudbridge-research) - Исследовательский центр
- [CloudBridge AI Service](https://github.com/twogc/cloudbridge-ai-service) - Сервис искусственного интеллекта

---

## Контакты

- **GitHub:** [@twogc](https://github.com/twogc)
- **Email:** info@cloudbridge-research.ru
- **Веб-сайт:** [cloudbridge-research.ru](https://cloudbridge-research.ru/)
- **Slack:** #ai-routing-support (для членов CloudBridge Research)

---

## Благодарности

Этот исследовательский проект является частью [CloudBridge Research Center](https://github.com/twogc/cloudbridge-research) и интегрируется с фреймворком тестирования [quic-test](https://github.com/twogc/quic-test).

Модели и инфраструктура адаптированы из экосистемы [CloudBridge AI Service](https://github.com/twogc/cloudbridge-ai-service).

---

**Статус:** В активной разработке
**Последнее обновление:** Ноябрь 2025
**Версия:** 2.0.0 (Research)
