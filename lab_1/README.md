# Лабораторная работа 1 — Компьютерное зрение: обнаружение объектов с YOLOv11

**Курс:** Киберфизические системы  
**Студент:** Мудров Павел Федорович  
**Группа:** M80-403Б-22  
**Вариант:** Задание на тройку — обнаружение и распознавание объектов  
**Фреймворк:** [ultralytics](https://github.com/ultralytics/ultralytics) — семейство моделей YOLOv11

---

## 1. Выбор датасета

**Датасет:** [VisDrone2019-DET](https://aiskyeye.com/submit-2023/object-detection-2/)

**10 классов:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

**Размер:** train: 6471; val: 548 изображений

**Обоснование выбора:**  
Автоматический мониторинг дорожной обстановки, пешеходных потоков и городских объектов с дронов для задач безопасности и транспортной аналитики. Датасет уже размечен в формате object detection и подходит для воспроизводимого сравнения компактной и более емкой модели YOLO.

---

## 2. Выбор метрик

| Метрика | Обоснование |
|---------|-------------|
| **mAP@0.5** | Основная метрика object detection: усредняет качество по всем классам при IoU=0.5. |
| **mAP@0.5:0.95** | Более строгая метрика, усредняющая mAP по нескольким порогам IoU и штрафующая за неточную локализацию. |
| **Precision** | Показывает долю корректных детекций среди всех найденных объектов и отражает уровень ложных срабатываний. |
| **Recall** | Показывает долю найденных объектов среди всех реальных и отражает количество пропусков. |

---

## 3. Структура проекта

```text
lab_1/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset/
│   │   └── README.md
│   └── experiments/
│       ├── baseline.yaml
│       └── improved.yaml
├── data/
│   ├── README.md
│   └── raw/
├── reports/
│   └── final_report.md
└── src/
    ├── download_visdrone.py
    ├── prepare_visdrone.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    ├── generate_report.py
    └── utils.py
```

---

## 4. Установка и запуск

### 4.1 Установка зависимостей

```bash
cd /Users/stiks/Desktop/ai/lab_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Подготовка датасета

```bash
.venv/bin/python src/download_visdrone.py
```

Скрипт скачивает архивы VisDrone train/val, распаковывает их в `data/raw/`, конвертирует аннотации в формат YOLO и создает локальный `data/visdrone_yolo/dataset.yaml`.

### 4.3 Обучение и сравнение

```bash
.venv/bin/python src/train.py --config configs/experiments/baseline.yaml --device mps
.venv/bin/python src/evaluate.py --config configs/experiments/baseline.yaml --device mps

.venv/bin/python src/train.py --config configs/experiments/improved.yaml --device mps
.venv/bin/python src/evaluate.py --config configs/experiments/improved.yaml --device mps

.venv/bin/python src/generate_report.py \
  --baseline-config configs/experiments/baseline.yaml \
  --improved-config configs/experiments/improved.yaml \
  --output reports/final_report.md
```

При необходимости этот же генератор можно направить и в `README.md`, чтобы обновить таблицы после запуска экспериментов.

---

## 5. Дизайн экспериментов

### Пункт 2 — Baseline

| Параметр | Значение |
|----------|----------|
| Модель | yolo11n.pt |
| Веса | Pretrained |
| Эпохи | 3 |
| Размер изображения | 640 |
| Оптимизатор | auto |
| Аугментации | mosaic=1.0, mixup=0.0, fliplr=0.5, translate=0.1, scale=0.5 |

### Пункт 3 — Улучшенный baseline

Гипотезы, проверяемые относительно baseline:

| # | Гипотеза | Изменение |
|---|---------|-----------|
| H1 | Более ёмкая модель улучшит качество детекции | `yolo11n.pt` -> `yolo11s.pt` |
| H2 | Увеличение разрешения поможет точнее локализовать мелкие объекты | `imgsz: 640 -> 704` |
| H3 | AdamW и более мягкий learning rate сделают обучение стабильнее | `optimizer: auto -> AdamW`, `lr0: 0.01 -> 0.001` |
| H4 | Изменение режима обучения и MixUp повысят устойчивость модели | `epochs: 3 -> 2`, `mixup: 0.0 -> 0.05` |

---

## 6. Результаты

### 6.1 Общие метрики (валидационная выборка)

| Метрика | Baseline (YOLO11n) | Improved (YOLO11s) |
|---------|:------------------:|:------------------:|
| **mAP@0.5** | 0.1138 | 0.2389 |
| **mAP@0.5:0.95** | 0.0610 | 0.1341 |
| **Precision** | 0.2805 | 0.4074 |
| **Recall** | 0.1723 | 0.2770 |

### 6.2 Per-class AP@0.5 (валидационная выборка)

| Класс | Baseline | Improved |
|-------|:--------:|:--------:|
| pedestrian | 0.1190 | 0.3040 |
| people | 0.0526 | 0.1761 |
| bicycle | 0.0013 | 0.0430 |
| car | 0.5441 | 0.6946 |
| van | 0.0860 | 0.2772 |
| truck | 0.0762 | 0.1684 |
| tricycle | 0.0039 | 0.0890 |
| awning-tricycle | 0.0001 | 0.0396 |
| bus | 0.1562 | 0.3334 |
| motor | 0.0983 | 0.2632 |

---

## 7. Выводы

### Пункт 2 — Baseline

Бейзлайн показал следующие результаты: `mAP@0.5 = 0.1138`, `mAP@0.5:0.95 = 0.0610`, `Precision = 0.2805`, `Recall = 0.1723`. Эти значения используются как точка отсчета для проверки улучшений.

### Пункт 3 — Улучшенный baseline vs baseline

| Гипотеза | Результат |
|---------|-----------|
| H1: Более ёмкая модель лучше детектирует объекты | **Подтверждена** — сравнение по `mAP@0.5:0.95`. Улучшение: +0.0730. |
| H2: Повышенное разрешение помогает малым классам | **Подтверждена** — сравнение среднего `AP@0.5` по классам `pedestrian, people, bicycle, motor`. Улучшение: +0.1288. |
| H3: AdamW и более мягкий learning rate делают обучение стабильнее | **Подтверждена** — косвенная проверка по `Precision` и `mAP@0.5:0.95`. `Precision`: +0.1269, `mAP@0.5:0.95`: +0.0730. |
| H4: Изменение режима обучения и MixUp уменьшают пропуски | **Подтверждена** — сравнение по `Recall`. Улучшение: +0.1048. |

**Итог:** Улучшенный эксперимент превосходит бейзлайн по основной быстрой метрике `mAP@0.5` на 0.1251. Это означает, что замена модели на `YOLO11s`, рост разрешения и корректировка гиперпараметров дали положительный эффект.
