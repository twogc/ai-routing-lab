# Интеграция AI Routing Lab с quic-test

Полное руководство по интеграции AI Routing Lab с инструментом тестирования QUIC протокола quic-test для сбора данных и валидации моделей машинного обучения.

## Обзор

Интеграция между AI Routing Lab и quic-test обеспечивает полный цикл разработки ML моделей:

```
quic-test (Go)
    ↓
Генерация метрик QUIC
    ↓
Prometheus Metrics Export
    ↓
AI Routing Lab Collector
    ↓
Обработка данных и Feature Engineering
    ↓
Обучение ML моделей
    ↓
Валидация на реальных данных quic-test
    ↓
Экспорт моделей для CloudBridge Relay
```

## Архитектура интеграции

### Компоненты

| Компонент | Язык | Функция |
|-----------|------|---------|
| quic-test | Go | Генерация QUIC трафика, экспорт метрик |
| Prometheus | Go | Хранение метрик в реальном времени |
| AI Routing Lab Collector | Python | Сбор метрик из Prometheus |
| ML Pipeline | Python | Обучение и валидация моделей |
| CloudBridge Relay | Go | Использование моделей для маршрутизации |

### Типы метрик

AI Routing Lab использует следующие метрики из quic-test:

#### Основные метрики производительности
- `quic_latency_ms` - Одностороняя задержка
- `quic_rtt_ms` - Время в пути туда и обратно
- `quic_jitter_ms` - Изменчивость RTT
- `quic_throughput_mbps` - Пропускная способность
- `quic_packet_loss_rate` - Коэффициент потери пакетов

#### Метрики надежности
- `quic_retransmits` - Количество переданных пакетов
- `quic_lost_packets` - Потерянные пакеты
- `quic_duplicate_packets` - Дублированные пакеты
- `quic_reordered_packets` - Переупорядоченные пакеты

#### Метрики управления перегруженностью
- `quic_congestion_window` - Размер окна перегруженности
- `quic_bytes_in_flight` - Непотвержденные данные в сети
- `quic_pacing_rate_mbps` - Скорость отправки пакетов

## Быстрая установка

### Шаг 1: Запуск quic-test с Prometheus

```bash
# Клонирование и сборка quic-test
cd quic-test
./bin/quic-test --mode=server --addr=:9000 --prometheus-port 9090
```

**Проверка Prometheus endpoint:**
```bash
curl http://localhost:9090/metrics
```

### Шаг 2: Установка AI Routing Lab

```bash
# Клонирование и установка
git clone https://github.com/twogc/ai-routing-lab.git
cd ai-routing-lab

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Шаг 3: Настройка сборщика данных

```yaml
# config/collectors/quic_test.yaml
collector:
  type: prometheus
  name: quic_test
  prometheus_url: http://localhost:9090
  scrape_interval: 1  # секунды

  # Метрики для сбора
  metrics:
    - quic_latency_ms
    - quic_rtt_ms
    - quic_jitter_ms
    - quic_throughput_mbps
    - quic_packet_loss_rate
    - quic_retransmits
    - quic_congestion_window
    - quic_bytes_in_flight
    - quic_pacing_rate_mbps
```

### Шаг 4: Сбор данных

```bash
# Сбор метрик в течение 10 минут
python -m data.collectors.quic_test_collector \
  --config config/collectors/quic_test.yaml \
  --output-file data/raw/quic_metrics.json \
  --duration 600
```

## Сбор данных для различных сетевых условий

Для обучения эффективных моделей необходимо собрать данные в разнообразных условиях:

### Сценарий 1: Отличные условия сети

```bash
# Отличная сеть (низкая задержка, без потерь)
./bin/quic-test --mode=test \
  --network-profile=excellent \
  --duration=300 \
  --report=data/raw/metrics_excellent.json
```

**Ожидаемые характеристики:**
- Latency: 5-10 мс
- Jitter: <1 мс
- Packet Loss: <0.1%

### Сценарий 2: Мобильная сеть

```bash
# Мобильная сеть (высокая задержка, потери)
./bin/quic-test --mode=test \
  --network-profile=mobile \
  --duration=300 \
  --report=data/raw/metrics_mobile.json
```

**Ожидаемые характеристики:**
- Latency: 150-250 мс
- Jitter: 10-30 мс
- Packet Loss: 5-15%

### Сценарий 3: Спутниковый канал

```bash
# Спутниковая сеть (очень высокая задержка)
./bin/quic-test --mode=test \
  --network-profile=satellite \
  --duration=300 \
  --report=data/raw/metrics_satellite.json
```

**Ожидаемые характеристики:**
- Latency: 500-1000 мс
- Jitter: 50-150 мс
- Packet Loss: 1-5%

### Сценарий 4: Враждебные условия

```bash
# Враждебные условия (высокие потери и задержка)
./bin/quic-test --mode=test \
  --network-profile=adversarial \
  --duration=300 \
  --report=data/raw/metrics_adversarial.json
```

**Ожидаемые характеристики:**
- Latency: 800-1500 мс
- Jitter: 100-300 мс
- Packet Loss: 15-30%

### Объединение всех датасетов

```bash
# Сбор данных из всех профилей
for profile in excellent good poor mobile satellite adversarial; do
  echo "Собираем данные для профиля: $profile"
  ./bin/quic-test --mode=test \
    --network-profile=$profile \
    --duration=600 \
    --report=data/raw/metrics_${profile}.json
done

# Объединение датасетов
python -m data.combine_datasets \
  --input data/raw/metrics_*.json \
  --output data/combined/all_metrics.json
```

## Обработка и подготовка данных

### Валидация данных

```bash
# Проверка качества собранных данных
python -m data.validate_dataset \
  --input data/combined/all_metrics.json \
  --output data/analysis/validation_report.json \
  --verbose
```

**Проверяет:**
- Наличие обязательных полей
- Диапазон значений метрик
- Отсутствие выбросов
- Временное покрытие данных

### Предобработка данных

```bash
# Нормализация и подготовка данных для ML
python -m data.preprocess \
  --input data/combined/all_metrics.json \
  --output data/processed/preprocessed_metrics.json \
  --normalize \
  --handle-missing-values \
  --remove-outliers
```

### Анализ признаков

```bash
# Анализ важности признаков и корреляций
python -m data.analyze_features \
  --input data/processed/preprocessed_metrics.json \
  --output data/analysis/feature_analysis.html
```

## Обучение моделей

### Обучение модели прогнозирования задержки

```bash
# Обучение LatencyPredictor
python -m models.train_predictor \
  --input-file data/processed/preprocessed_metrics.json \
  --model-type latency \
  --test-split 0.2 \
  --output-model models/latency_predictor.pkl \
  --hyperparams n_estimators=100,max_depth=15
```

**Выходная информация:**
```
Training Latency Predictor...
- Features: 42
- Training samples: 8000
- Test samples: 2000
- Model type: Random Forest

Results:
- R² Score: 0.942
- MAE: 2.15 мс
- RMSE: 3.21 мс
- Training time: 45 seconds

Model saved: models/latency_predictor.pkl
```

### Обучение модели прогнозирования джиттера

```bash
# Обучение JitterPredictor
python -m models.train_predictor \
  --input-file data/processed/preprocessed_metrics.json \
  --model-type jitter \
  --test-split 0.2 \
  --output-model models/jitter_predictor.pkl \
  --hyperparams n_estimators=100,max_depth=12
```

**Выходная информация:**
```
Training Jitter Predictor...
- Features: 35
- Training samples: 8000
- Test samples: 2000
- Model type: Random Forest

Results:
- R² Score: 0.931
- MAE: 0.82 мс
- RMSE: 1.14 мс
- Training time: 42 seconds

Model saved: models/jitter_predictor.pkl
```

### Создание ансамбля моделей

```bash
# Создание ансамбля для выбора маршрутов
python -m models.create_ensemble \
  --latency-model models/latency_predictor.pkl \
  --jitter-model models/jitter_predictor.pkl \
  --output-model models/route_ensemble.pkl \
  --latency-weight 0.7 \
  --jitter-weight 0.3
```

## Валидация моделей на реальных данных quic-test

### Генерация тестовых данных

```bash
# Генерация новых тестовых данных (не использованных при обучении)
./bin/quic-test --mode=test \
  --network-profile=mobile \
  --duration=600 \
  --connections=10 \
  --streams=20 \
  --report=data/test/validation_metrics.json
```

### Валидация прогнозов

```bash
# Валидация моделей на новых данных
python -m models.validate_predictions \
  --model-file models/route_ensemble.pkl \
  --test-file data/test/validation_metrics.json \
  --output-report data/reports/validation_report.json \
  --detailed
```

**Пример выходного отчета:**
```json
{
  "model": "route_ensemble",
  "validation_samples": 2000,
  "metrics": {
    "latency_predictor": {
      "r2_score": 0.938,
      "mae": 2.34,
      "rmse": 3.45,
      "mape": 8.2
    },
    "jitter_predictor": {
      "r2_score": 0.925,
      "mae": 0.91,
      "rmse": 1.28,
      "mape": 12.1
    },
    "route_selection": {
      "optimal_selection_rate": 0.963
    }
  },
  "status": "PASSED"
}
```

### Проверка соответствия целевым показателям

```bash
# Проверка, превышает ли точность целевое значение
python -m models.check_accuracy \
  --report data/reports/validation_report.json \
  --latency-threshold 0.92 \
  --jitter-threshold 0.92 \
  --route-threshold 0.95
```

## Постоянная валидация в производстве

### Мониторинг производительности моделей

```bash
# Запуск мониторинга производительности моделей
python -m inference.model_monitor \
  --model-dir /etc/cloudbridge/ml_models/ \
  --prometheus-port 9091 \
  --check-interval 300
```

**Мониторит:**
- Точность прогнозов на реальных данных
- Дрейф моделей (model drift)
- Время инференции
- Качество выбора маршрутов

### Переобучение моделей

```bash
# Периодическое переобучение моделей на новых данных
python -m training.continuous_retraining \
  --model-dir /etc/cloudbridge/ml_models/ \
  --quic-test-url http://localhost:9090 \
  --retrain-interval 86400 \
  --min-samples 5000
```

## Интеграция с CloudBridge Relay

### Экспорт моделей для Relay

```bash
# Экспорт обученных моделей в формат для Relay
python -m inference.export_models \
  --ensemble-model models/route_ensemble.pkl \
  --export-format onnx \
  --output-dir /etc/cloudbridge/ml_models/
```

### Запуск сервиса предсказания

```bash
# Запуск FastAPI сервиса для предсказания маршрутов
python -m inference.predictor_service \
  --model-dir /etc/cloudbridge/ml_models/ \
  --port 5000 \
  --workers 4
```

**Примеры API запросов:**

```bash
# Прогнозирование задержки
curl -X POST http://localhost:5000/predict/latency \
  -H "Content-Type: application/json" \
  -d '{
    "features": [25.5, 2.3, 0.95, 1.0, 150.2, 0.001]
  }'

# Выбор оптимального маршрута
curl -X POST http://localhost:5000/select/best-route \
  -H "Content-Type: application/json" \
  -d '{
    "routes": {
      "path_0": {"features": [25.5, 2.3]},
      "path_1": {"features": [30.1, 3.1]},
      "path_2": {"features": [20.3, 1.8]}
    }
  }'
```

## Устранение неполадок

### Проблема: Collector не подключается к Prometheus

**Решение:**
```bash
# Проверка доступности Prometheus
curl http://localhost:9090/metrics

# Проверка логов quic-test
./bin/quic-test --mode=server --debug --prometheus-port 9090

# Проверка сетевого соединения
netstat -tlnp | grep 9090
```

### Проблема: Низкая точность моделей

**Решение 1:** Увеличить размер датасета
```bash
# Собрать данные на более продолжительный период
./bin/quic-test --mode=test --duration=1800 --report=extended_metrics.json
```

**Решение 2:** Улучшить feature engineering
```bash
# Анализ признаков
python -m data.analyze_features --input data/processed/preprocessed_metrics.json

# Создание новых признаков
python -m data.engineer_features --input data/processed/preprocessed_metrics.json
```

**Решение 3:** Попробовать другой тип модели
```bash
# Обучить XGBoost вместо Random Forest
python -m models.train_predictor \
  --input-file data/processed/preprocessed_metrics.json \
  --model-type xgboost \
  --output-model models/xgboost_predictor.pkl
```

### Проблема: Model Drift (дрейф моделей)

**Решение:** Настроить автоматическое переобучение
```bash
# Настройка порога дрейфа и переобучения
python -m training.setup_retraining \
  --model-dir /etc/cloudbridge/ml_models/ \
  --drift-threshold 0.05 \
  --retrain-frequency daily \
  --min-accuracy-drop 0.02
```

## Рекомендуемый рабочий процесс

### Еженедельный цикл разработки

```bash
# Понедельник: Сбор новых данных
for profile in excellent good poor mobile satellite adversarial; do
  ./bin/quic-test --mode=test --network-profile=$profile --duration=600 \
    --report=data/weekly/metrics_${profile}_$(date +%Y%m%d).json
done

# Вторник: Объединение и предобработка
python -m data.combine_datasets \
  --input data/weekly/metrics_*.json \
  --output data/weekly/combined_metrics.json

# Среда: Переобучение моделей
python -m models.train_predictor --input-file data/weekly/combined_metrics.json

# Четверг: Валидация
python -m models.validate_predictions --test-file data/test/validation_metrics.json

# Пятница: Развертывание в production
python -m inference.export_models --output-dir /etc/cloudbridge/ml_models/
```

## Примеры интеграции кода

### Интеграция в Python приложение

```python
from models.routing import RoutePredictionEnsemble
import numpy as np

# Загрузка обученного ансамбля
ensemble = RoutePredictionEnsemble.load('models/route_ensemble.pkl')

# Получение метрик из quic-test Prometheus
from data.collectors.quic_test_collector import PrometheusCollector

collector = PrometheusCollector(prometheus_url="http://localhost:9090")
metrics = collector.get_latest_metrics()

# Подготовка данных для прогнозирования
features = ensemble.prepare_features(metrics)

# Выбор оптимального маршрута
best_route = ensemble.predict_best_route(features)
print(f"Оптимальный маршрут: {best_route}")
```

### Интеграция в Go приложение (CloudBridge Relay)

```go
import "github.com/twogc/ai-routing-lab/inference"

// Инициализация ML модели
model, err := inference.LoadEnsemble("/etc/cloudbridge/ml_models/route_ensemble.onnx")
if err != nil {
    log.Fatal(err)
}

// Получение метрик маршрутов
metrics := getRouteMetrics()

// Прогнозирование оптимального маршрута
bestRoute := model.PredictBestRoute(metrics)
fmt.Printf("Выбран маршрут: %s\n", bestRoute)
```

## Статистика интеграции

### Требования к производительности

| Компонент | Требование | Статус |
|-----------|-----------|--------|
| Latency Prediction R² | > 0.92 | ✅ Достигнуто (0.94) |
| Jitter Prediction R² | > 0.92 | ✅ Достигнуто (0.93) |
| Route Selection Accuracy | > 0.95 | ✅ Достигнуто (0.96) |
| Inference Latency | < 10 мс | ✅ Достигнуто (5-8 мс) |
| Model Update Frequency | Daily/Weekly | ✅ Возможно |

### Требуемые объемы данных

- **Минимум:** 1000 образцов (15-20 минут тестирования)
- **Оптимально:** 10000+ образцов (2-3 часа разнообразного тестирования)
- **Рекомендуемо:** 50000+ образцов (для production моделей)

---

**Последнее обновление:** Ноябрь 2025
**Версия:** 2.0
**Статус:** Production Ready
