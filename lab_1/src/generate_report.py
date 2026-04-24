"""Generate a Markdown report from experiment configs and saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import dump_json, load_json, load_yaml, resolve_from_root

BASE_CONFIG: dict[str, Any] = {}

DEFAULT_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

METRIC_DEFINITIONS = [
    (
        "metrics/mAP50(B)",
        "mAP@0.5",
        "Основная метрика object detection: усредняет качество по всем классам при IoU=0.5.",
    ),
    (
        "metrics/mAP50-95(B)",
        "mAP@0.5:0.95",
        "Более строгая метрика, усредняющая mAP по нескольким порогам IoU и штрафующая за неточную локализацию.",
    ),
    (
        "metrics/precision(B)",
        "Precision",
        "Показывает долю корректных детекций среди всех найденных объектов и отражает уровень ложных срабатываний.",
    ),
    (
        "metrics/recall(B)",
        "Recall",
        "Показывает долю найденных объектов среди всех реальных и отражает количество пропусков.",
    ),
]

PROJECT_TREE = """```text
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
```"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        required=True,
        help="Path to the baseline experiment YAML file.",
    )
    parser.add_argument(
        "--improved-config",
        required=True,
        help="Path to the improved experiment YAML file.",
    )
    parser.add_argument(
        "--output",
        default="reports/final_report.md",
        help="Path to the generated Markdown report.",
    )
    return parser.parse_args()


def load_bundle(config_path: str | Path) -> dict[str, Any]:
    """Load config and optional metrics for one experiment."""
    config = load_yaml(config_path)
    run_dir = resolve_from_root(config.get("project", "runs/lab1")) / config["experiment_name"]
    train_metrics = load_json(run_dir / "metrics_train.json")
    val_metrics = load_json(run_dir / "metrics_val.json")
    return {
        "config": config,
        "run_dir": run_dir,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def format_metric(value: Any) -> str:
    """Format one metric for Markdown tables."""
    if value is None:
        return "н/д"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def format_delta(value: Any) -> str:
    """Format a signed delta value."""
    if value is None or not isinstance(value, (int, float)):
        return "н/д"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def metric_value(bundle: dict[str, Any], key: str) -> Any:
    """Extract one summary metric from the validation report."""
    val_metrics = bundle.get("val_metrics") or {}
    summary = val_metrics.get("summary", {})
    return summary.get(key)


def dataset_names(bundle: dict[str, Any]) -> list[str]:
    """Resolve dataset class names from saved metrics, dataset yaml or defaults."""
    val_metrics = bundle.get("val_metrics") or {}
    names = val_metrics.get("dataset_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]

    dataset_path = resolve_from_root(bundle["config"]["dataset_yaml"])
    if dataset_path.exists():
        dataset_config = load_yaml(dataset_path)
        raw_names = dataset_config.get("names", {})
        if isinstance(raw_names, dict):
            indexes = sorted(int(index) for index in raw_names.keys())
            return [str(raw_names[str(index)] if str(index) in raw_names else raw_names[index]) for index in indexes]
        if isinstance(raw_names, list) and raw_names:
            return [str(name) for name in raw_names]

    configured_names = bundle["config"].get("expected_class_names")
    if isinstance(configured_names, list) and configured_names:
        return [str(name) for name in configured_names]

    return list(DEFAULT_CLASS_NAMES)


def per_class_map(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index per-class metrics by class name."""
    val_metrics = bundle.get("val_metrics") or {}
    per_class = val_metrics.get("per_class", [])
    indexed: dict[str, dict[str, Any]] = {}
    if isinstance(per_class, list):
        for item in per_class:
            if isinstance(item, dict) and "class_name" in item:
                indexed[str(item["class_name"])] = item
    return indexed


def per_class_metric(bundle: dict[str, Any], class_name: str, key: str) -> Any:
    """Extract one per-class metric value."""
    item = per_class_map(bundle).get(class_name, {})
    return item.get(key)


def average_per_class_metric(bundle: dict[str, Any], class_names: list[str], key: str) -> float | None:
    """Average a per-class metric over a selected subset of classes."""
    values = [
        per_class_metric(bundle, class_name, key)
        for class_name in class_names
        if isinstance(per_class_metric(bundle, class_name, key), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def render_metric_table() -> str:
    """Render the metric definition table."""
    rows = ["| Метрика | Обоснование |", "|---------|-------------|"]
    for _, title, rationale in METRIC_DEFINITIONS:
        rows.append(f"| **{title}** | {rationale} |")
    return "\n".join(rows)


def render_dataset_description(class_names: list[str]) -> str:
    """Render the dataset section."""
    dataset_name = BASE_CONFIG.get("dataset_name", "Dataset")
    dataset_url = BASE_CONFIG.get("dataset_url", "#")
    business_context = BASE_CONFIG.get("business_context", "не указан")
    dataset_stats = BASE_CONFIG.get("dataset_stats", {})
    class_list = ", ".join(class_names)
    class_count = len(class_names)
    class_word = "класс" if class_count == 1 else "класса" if 2 <= class_count <= 4 else "классов"
    stats_line = ""
    if isinstance(dataset_stats, dict):
        train_images = dataset_stats.get("train_images")
        val_images = dataset_stats.get("val_images")
        test_images = dataset_stats.get("test_images")
        parts: list[str] = []
        if train_images is not None:
            parts.append(f"train: {train_images}")
        if val_images is not None:
            parts.append(f"val: {val_images}")
        if test_images is not None:
            parts.append(f"test: {test_images}")
        if parts:
            stats_line = f"**Размер:** {'; '.join(parts)} изображений\n\n"
    return (
        f"**Датасет:** [{dataset_name}]({dataset_url})\n\n"
        f"**{class_count} {class_word}:** {class_list}\n\n"
        f"{stats_line}"
        "**Обоснование выбора:**  \n"
        f"{business_context} "
        "Датасет уже размечен в формате object detection и подходит для воспроизводимого "
        "сравнения компактной и более емкой модели YOLO."
    )


def summarize_augmentations(train_config: dict[str, Any]) -> str:
    """Render a short summary of augmentation settings."""
    return (
        f"mosaic={train_config.get('mosaic', 'н/д')}, "
        f"mixup={train_config.get('mixup', 'н/д')}, "
        f"fliplr={train_config.get('fliplr', 'н/д')}, "
        f"translate={train_config.get('translate', 'н/д')}, "
        f"scale={train_config.get('scale', 'н/д')}"
    )


def render_baseline_design_table(bundle: dict[str, Any]) -> str:
    """Render the baseline experiment design table."""
    config = bundle["config"]
    train_config = config.get("train", {})
    rows = [
        "| Параметр | Значение |",
        "|----------|----------|",
        f"| Модель | {config['model']} |",
        f"| Веса | {'Pretrained' if train_config.get('pretrained', True) else 'Без предобучения'} |",
        f"| Эпохи | {train_config.get('epochs', 'н/д')} |",
        f"| Размер изображения | {train_config.get('imgsz', 'н/д')} |",
        f"| Оптимизатор | {train_config.get('optimizer', 'н/д')} |",
        f"| Аугментации | {summarize_augmentations(train_config)} |",
    ]
    return "\n".join(rows)


def render_improved_hypotheses_table(baseline: dict[str, Any], improved: dict[str, Any]) -> str:
    """Render the hypotheses and experiment changes for the improved run."""
    baseline_train = baseline["config"].get("train", {})
    improved_train = improved["config"].get("train", {})
    rows = [
        "| # | Гипотеза | Изменение |",
        "|---|---------|-----------|",
        (
            "| H1 | Более ёмкая модель улучшит качество детекции | "
            f"`{baseline['config']['model']}` -> `{improved['config']['model']}` |"
        ),
        (
            "| H2 | Увеличение разрешения поможет точнее локализовать мелкие объекты | "
            f"`imgsz: {baseline_train.get('imgsz', 'н/д')} -> {improved_train.get('imgsz', 'н/д')}` |"
        ),
        (
            "| H3 | AdamW и более мягкий learning rate сделают обучение стабильнее | "
            f"`optimizer: {baseline_train.get('optimizer', 'н/д')} -> {improved_train.get('optimizer', 'н/д')}`, "
            f"`lr0: {baseline_train.get('lr0', 'н/д')} -> {improved_train.get('lr0', 'н/д')}` |"
        ),
        (
            "| H4 | Изменение режима обучения и MixUp повысят устойчивость модели | "
            f"`epochs: {baseline_train.get('epochs', 'н/д')} -> {improved_train.get('epochs', 'н/д')}`, "
            f"`mixup: {baseline_train.get('mixup', 'н/д')} -> {improved_train.get('mixup', 'н/д')}` |"
        ),
    ]
    return "\n".join(rows)


def render_overall_metrics_table(baseline: dict[str, Any], improved: dict[str, Any]) -> str:
    """Render the overall metrics comparison table."""
    rows = ["| Метрика | Baseline (YOLO11n) | Improved (YOLO11s) |", "|---------|:------------------:|:------------------:|"]
    for metric_key, title, _ in METRIC_DEFINITIONS:
        rows.append(
            f"| **{title}** | {format_metric(metric_value(baseline, metric_key))} | "
            f"{format_metric(metric_value(improved, metric_key))} |"
        )
    return "\n".join(rows)


def render_per_class_table(baseline: dict[str, Any], improved: dict[str, Any], class_names: list[str]) -> str:
    """Render the per-class AP@0.5 table."""
    rows = ["| Класс | Baseline | Improved |", "|-------|:--------:|:--------:|"]
    for class_name in class_names:
        rows.append(
            f"| {class_name} | {format_metric(per_class_metric(baseline, class_name, 'map50'))} | "
            f"{format_metric(per_class_metric(improved, class_name, 'map50'))} |"
        )
    return "\n".join(rows)


def render_run_commands() -> str:
    """Render installation and execution commands."""
    slash = "\\"
    return f"""### 4.1 Установка зависимостей

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

.venv/bin/python src/generate_report.py {slash}
  --baseline-config configs/experiments/baseline.yaml {slash}
  --improved-config configs/experiments/improved.yaml {slash}
  --output reports/final_report.md
```

При необходимости этот же генератор можно направить и в `README.md`, чтобы обновить таблицы после запуска экспериментов."""


def baseline_result_summary(bundle: dict[str, Any]) -> str:
    """Build a short baseline conclusion paragraph."""
    map50 = metric_value(bundle, "metrics/mAP50(B)")
    map5095 = metric_value(bundle, "metrics/mAP50-95(B)")
    precision = metric_value(bundle, "metrics/precision(B)")
    recall = metric_value(bundle, "metrics/recall(B)")
    if all(isinstance(value, (int, float)) for value in (map50, map5095, precision, recall)):
        return (
            "Бейзлайн показал следующие результаты: "
            f"`mAP@0.5 = {map50:.4f}`, `mAP@0.5:0.95 = {map5095:.4f}`, "
            f"`Precision = {precision:.4f}`, `Recall = {recall:.4f}`. "
            "Эти значения используются как точка отсчета для проверки улучшений."
        )
    return (
        "Фактические значения бейзлайна пока не сохранены. После запуска `evaluate.py` "
        "эта секция автоматически заполнится итоговыми метриками."
    )


def compare_values(baseline_value: Any, improved_value: Any) -> tuple[str, str]:
    """Compare two numeric values and return status and explanation."""
    if not isinstance(baseline_value, (int, float)) or not isinstance(improved_value, (int, float)):
        return "н/д", "Метрики появятся после запуска `evaluate.py`."

    delta = improved_value - baseline_value
    if delta > 0.003:
        return "Подтверждена", f"Улучшение: {format_delta(delta)}."
    if delta < -0.003:
        return "Не подтверждена", f"Ухудшение: {format_delta(delta)}."
    return "Частично", f"Изменение близко к нулю: {format_delta(delta)}."


def render_hypothesis_results(baseline: dict[str, Any], improved: dict[str, Any]) -> str:
    """Render a results table for the improved hypotheses."""
    rows = ["| Гипотеза | Результат |", "|---------|-----------|"]

    status_h1, details_h1 = compare_values(
        metric_value(baseline, "metrics/mAP50-95(B)"),
        metric_value(improved, "metrics/mAP50-95(B)"),
    )
    rows.append(
        "| H1: Более ёмкая модель лучше детектирует объекты | "
        f"**{status_h1}** — сравнение по `mAP@0.5:0.95`. {details_h1} |"
    )

    focus_classes = list(BASE_CONFIG.get("focus_classes", []))
    if not focus_classes:
        focus_classes = dataset_names(baseline)
    small_baseline = average_per_class_metric(baseline, focus_classes, "map50")
    small_improved = average_per_class_metric(improved, focus_classes, "map50")
    status_h2, details_h2 = compare_values(small_baseline, small_improved)
    rows.append(
        "| H2: Повышенное разрешение помогает малым классам | "
        f"**{status_h2}** — сравнение среднего `AP@0.5` по классам "
        f"`{', '.join(focus_classes)}`. {details_h2} |"
    )

    precision_base = metric_value(baseline, "metrics/precision(B)")
    precision_improved = metric_value(improved, "metrics/precision(B)")
    map_base = metric_value(baseline, "metrics/mAP50-95(B)")
    map_improved = metric_value(improved, "metrics/mAP50-95(B)")
    if all(isinstance(value, (int, float)) for value in (precision_base, precision_improved, map_base, map_improved)):
        precision_delta = precision_improved - precision_base
        map_delta = map_improved - map_base
        if precision_delta > 0 and map_delta >= 0:
            status_h3 = "Подтверждена"
        elif precision_delta < 0 and map_delta < 0:
            status_h3 = "Не подтверждена"
        else:
            status_h3 = "Частично"
        details_h3 = (
            f"`Precision`: {format_delta(precision_delta)}, "
            f"`mAP@0.5:0.95`: {format_delta(map_delta)}."
        )
    else:
        status_h3 = "н/д"
        details_h3 = "Метрики появятся после запуска `evaluate.py`."
    rows.append(
        "| H3: AdamW и более мягкий learning rate делают обучение стабильнее | "
        f"**{status_h3}** — косвенная проверка по `Precision` и `mAP@0.5:0.95`. {details_h3} |"
    )

    status_h4, details_h4 = compare_values(
        metric_value(baseline, "metrics/recall(B)"),
        metric_value(improved, "metrics/recall(B)"),
    )
    rows.append(
        "| H4: Изменение режима обучения и MixUp уменьшают пропуски | "
        f"**{status_h4}** — сравнение по `Recall`. {details_h4} |"
    )
    return "\n".join(rows)


def build_final_conclusion(baseline: dict[str, Any], improved: dict[str, Any]) -> str:
    """Build the concluding paragraph."""
    baseline_map = metric_value(baseline, "metrics/mAP50(B)")
    improved_map = metric_value(improved, "metrics/mAP50(B)")
    if isinstance(baseline_map, (int, float)) and isinstance(improved_map, (int, float)):
        delta = improved_map - baseline_map
        if delta > 0:
            return (
                "Улучшенный эксперимент превосходит бейзлайн по основной быстрой метрике "
                f"`mAP@0.5` на {delta:.4f}. Это означает, что замена модели на `YOLO11s`, "
                "рост разрешения и корректировка гиперпараметров дали положительный эффект."
            )
        if delta < 0:
            return (
                "Улучшенный эксперимент не превзошел бейзлайн по `mAP@0.5`. "
                "Стоит дополнительно проверить размер батча, длительность обучения и интенсивность аугментаций."
            )
        return "Оба эксперимента показали одинаковый `mAP@0.5`, поэтому текущее улучшение не дало прироста."

    return (
        "Итоговый вывод будет сформирован автоматически после появления фактических метрик. "
        "Сейчас шаблон уже подготовлен в формате, пригодном для сдачи."
    )


def build_report(baseline: dict[str, Any], improved: dict[str, Any]) -> str:
    """Build the final Markdown report."""
    class_names = dataset_names(baseline)
    report = f"""# Лабораторная работа 1 — Компьютерное зрение: обнаружение объектов с YOLOv11

**Курс:** Киберфизические системы  
**Студент:** {BASE_CONFIG.get("student_name", "не указан")}  
**Группа:** {BASE_CONFIG.get("student_group", "не указана")}  
**Вариант:** Задание на тройку — обнаружение и распознавание объектов  
**Фреймворк:** [ultralytics](https://github.com/ultralytics/ultralytics) — семейство моделей YOLOv11

---

## 1. Выбор датасета

{render_dataset_description(class_names)}

---

## 2. Выбор метрик

{render_metric_table()}

---

## 3. Структура проекта

{PROJECT_TREE}

---

## 4. Установка и запуск

{render_run_commands()}

---

## 5. Дизайн экспериментов

### Пункт 2 — Baseline

{render_baseline_design_table(baseline)}

### Пункт 3 — Улучшенный baseline

Гипотезы, проверяемые относительно baseline:

{render_improved_hypotheses_table(baseline, improved)}

---

## 6. Результаты

### 6.1 Общие метрики (валидационная выборка)

{render_overall_metrics_table(baseline, improved)}

### 6.2 Per-class AP@0.5 (валидационная выборка)

{render_per_class_table(baseline, improved, class_names)}

---

## 7. Выводы

### Пункт 2 — Baseline

{baseline_result_summary(baseline)}

### Пункт 3 — Улучшенный baseline vs baseline

{render_hypothesis_results(baseline, improved)}

**Итог:** {build_final_conclusion(baseline, improved)}
"""
    return report


def main() -> None:
    """Generate the Markdown report and a small machine-readable manifest."""
    args = parse_args()
    baseline = load_bundle(args.baseline_config)
    improved = load_bundle(args.improved_config)
    global BASE_CONFIG
    BASE_CONFIG = baseline["config"]

    output_path = resolve_from_root(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_report(baseline, improved)
    output_path.write_text(report_text, encoding="utf-8")

    manifest = {
        "baseline_run_dir": str(baseline["run_dir"]),
        "improved_run_dir": str(improved["run_dir"]),
        "report_path": str(output_path),
    }
    dump_json(manifest, output_path.with_suffix(".json"))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
