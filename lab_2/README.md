# Лабораторная работа 2 — NLP: инференс через Ollama

**Курс:** Киберфизические системы  
**Студент:** Мудров Павел Федорович  
**Группа:** M80-403Б-22  
**Вариант:** Задание на тройку  
**Стек:** `Ollama`, `Qwen2.5:0.5b`, `Python`, `requests`

---

## 1. Постановка задачи

Для варианта на тройку требуется:

1. установить `ollama`;
2. скачать модель `qwen2.5:0.5b`;
3. поднять `Ollama` сервер;
4. написать скрипт, отправляющий HTTP-запросы к локальному серверу;
5. прогнать скрипт на `10` произвольных запросах;
6. оформить отчет инференса в виде таблицы из двух столбцов:
   `запрос к LLM` и `вывод LLM`.

---

## 2. Структура проекта

```text
lab_2/
├── README.md
├── requirements.txt
├── data/
│   └── prompts.json
├── reports/
│   ├── inference_report.md
│   └── inference_results.json
└── src/
    ├── generate_report.py
    ├── ollama_client.py
    ├── run_inference.py
    └── utils.py
```

---

## 3. Установка и запуск

### 3.1 Установка Ollama

```bash
brew install ollama
```

### 3.2 Загрузка модели

```bash
ollama pull qwen2.5:0.5b
```

### 3.3 Запуск сервера Ollama

```bash
ollama serve
```

По умолчанию сервер поднимается на `http://127.0.0.1:11434`.

### 3.4 Установка зависимостей Python

```bash
cd /Users/stiks/Desktop/ai/lab_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.5 Запуск инференса

```bash
.venv/bin/python src/run_inference.py
.venv/bin/python src/generate_report.py
```

После выполнения команд будут сформированы:

- `reports/inference_results.json`
- `reports/inference_report.md`

---

## 4. Реализация

- Скрипт [run_inference.py](/Users/stiks/Desktop/ai/lab_2/src/run_inference.py) отправляет `POST`-запросы на эндпоинт `/api/generate`.
- Модель задается параметром `--model`, по умолчанию используется `qwen2.5:0.5b`.
- Набор из `10` запросов хранится в [prompts.json](/Users/stiks/Desktop/ai/lab_2/data/prompts.json).
- Генерация markdown-отчета выполняется скриптом [generate_report.py](/Users/stiks/Desktop/ai/lab_2/src/generate_report.py).
- Все функции снабжены `docstring`-документацией.

---

## 5. Результаты инференса

Актуальный отчет находится в файле [inference_report.md](/Users/stiks/Desktop/ai/lab_2/reports/inference_report.md).

---

## 6. Вывод

В работе был поднят локальный `Ollama`-сервер и реализован Python-клиент на `requests`, отправляющий HTTP-запросы к модели `Qwen2.5:0.5b`. Скрипт был прогнан на десяти произвольных запросах, а результаты автоматически сохранены в `json` и оформлены в виде markdown-отчета.
