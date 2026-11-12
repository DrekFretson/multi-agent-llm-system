# Adaptive Multi-Agent LLM System

Проект Python-системы, которая разбивает пользовательский запрос на задачи,
передаёт их специализированным LLM-агентам, координирует исполнение и собирает
результаты. Модели вызываются через OpenRouter; роли и параметры генерации
настраиваются в YAML.

## Что реализовано

- `LLMClient` для запросов к OpenRouter, обработка ошибок и JSONL-логирование.
- YAML-каталог моделей и конфигурации ролей с проверкой при загрузке.
- Агенты Orchestrator, Research, Coding, Math и Critic.
- Декомпозиция запросов, Adaptive Router, Task Store и Task Executor с
  зависимостями, ограничениями ресурсов, fallback и ограниченными повторами.
- Межагентные сообщения и делегирование задач.
- Инструменты поиска, математических вычислений и проверки результатов.
- Сбор метрик и запуск конфигурируемых экспериментов; конфигурации и наборы
  сценариев лабораторных работ находятся в `experiments/`.
- Mock-тесты компонентов и интеграционных сценариев. Unit-тесты не обращаются
  к OpenRouter или Tavily.

Полный путь запуска доступен из Python/Jupyter; отдельной CLI-команды для
пользовательских запросов пока нет. Реальные запросы требуют API-ключей и могут
расходовать лимиты провайдеров.

## Структура проекта

```text
src/
  agents/          базовый класс и специализированные агенты, Orchestrator
  llm/             OpenRouter-клиент и реестр моделей
  routing/         Adaptive Router
  coordination/    Task Store, Task Executor, делегирование и эксперименты
  communication/   сообщения и Message Bus
  evaluation/      метрики и лабораторные experiment runners
  logging/         структурированные JSONL-события
  security/        политики безопасности
  tools/           поиск, математика и проверка результатов
config/            модели и роли (`models.yaml`, `agents.yaml`)
notebooks/         ручные эксперименты и примеры
experiments/       конфигурации, наборы сценариев и локальные результаты
tests/              автоматические тесты
Отчёты/             текстовые отчёты по лабораторным работам
```

## Отчёты по лабораторным работам

- **ЛР-1:** отдельный отчёт в текущем репозитории не найден.
- **ЛР-2:** [Отчёт ЛР-2](Отчёты/lab_2/Отчёт%20ЛР-2.md), [методика](Отчёты/lab_2/methodology.md), результаты — в `Отчёты/lab_2/results/`.
- **ЛР-3:** [Отчёт ЛР-3](Отчёты/lab_3/Отчёт%20ЛР-3.md), [методика](Отчёты/lab_3/methodology.md), результаты — в `Отчёты/lab_3/results/`.
- **ЛР-4:** [Отчёт ЛР-4](Отчёты/lab_4/Отчёт%20ЛР-4.md), [методика](Отчёты/lab_4/methodology.md), результаты — в `Отчёты/lab_4/results/`.

Конфигурации лабораторных экспериментов лежат в `experiments/configs/`, входные
сценарии — в `experiments/datasets/`. Отчёты и результаты — разные материалы:
в `Отчёты/` находятся оформленные выводы, в `Отчёты/lab_N/results/` — данные
запусков.

## Технологии

Python 3.14, OpenAI SDK для OpenRouter API, `python-dotenv`, PyYAML, Tavily,
SymPy, NumPy, SciPy и Jupyter Notebook. Зависимости перечислены в
[`requirements.txt`](requirements.txt).

## Установка и настройка

Из корня репозитория создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Создайте локальный `.env` из шаблона и заполните нужные ключи:

```bash
cp .env.example .env
```

Для LLM-запросов задайте `OPENROUTER_API_KEY`. Для поиска Research Agent также
может понадобиться `TAVILY_API_KEY`. `.env` исключён из Git; не помещайте в него
примерные или реальные секреты, которые затем попадут в коммит.

## Запуск и проверки

Запустить автоматические тесты из корня проекта:

```bash
python -m pytest -q
```

Чтобы выполнить пользовательский запрос, откройте `notebooks/test.ipynb` в
Jupyter с корнем репозитория в качестве рабочей директории. Если Jupyter ещё не
установлен, установите его отдельно и запустите ноутбук:

```bash
python -m pip install jupyter
jupyter notebook notebooks/test.ipynb
```

В ноутбуке создайте
`LLMClient`, загрузите роли через `AgentConfigRegistry`, передайте агентов в
`Orchestrator` и вызовите `run_adaptive()`:

```python
from src.agents import (
    AgentConfigRegistry, ResearchAgent, CodingAgent,
    MathAgent, CriticAgent, Orchestrator,
)
from src.llm import LLMClient

configs = AgentConfigRegistry()
client = LLMClient(timeout=60.0)
agents = {
    "research": ResearchAgent(client=client, config=configs.get_agent("research")),
    "coding": CodingAgent(client=client, config=configs.get_agent("coding")),
    "math": MathAgent(client=client, config=configs.get_agent("math")),
    "critic": CriticAgent(client=client, config=configs.get_agent("critic")),
}
orchestrator = Orchestrator(
    client=client,
    agents=agents,
    config=configs.get_agent("orchestrator"),
)
result = orchestrator.run_adaptive("Объясни принцип 3-2-1 для резервных копий.")
print(result.status)
print(result.final_answer)
client.close()
```

Изменить модели и параметры генерации можно в `config/agents.yaml`; сами модели
описаны по логическим ID в `config/models.yaml`. Запросы к моделям выполняются
вручную из Notebook или Python и используют лимиты OpenRouter. Тесты выше
реальные API-запросы не отправляют.

## Результаты запусков

Метрики adaptive-запусков и локальные результаты экспериментов сохраняются в
`experiments/results/` (JSON и CSV). Примеры ответов из Notebook находятся в
`notebooks/notebooks/`. Эти файлы являются результатами конкретных запусков,
а не исходным кодом системы.
