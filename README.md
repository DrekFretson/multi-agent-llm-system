# Adaptive Multi-Agent LLM System

## Краткое описание

Проект предназначен для системы, которая будет распределять запросы между специализированными LLM-агентами и объединять их результаты. Сейчас реализованы OpenRouter-клиент и локальный реестр моделей; сами агенты и координация ещё не реализованы.

## Идея и назначение

Предполагается, что система будет адаптивно выбирать подходящих исполнителей, координировать их работу и проверять итоговый ответ. Разделение на компоненты должно упростить развитие маршрутизации, взаимодействия, оценки качества и безопасности.

## Предполагаемые роли агентов

- **Orchestrator** — планирует обработку запроса и координирует выполнение задач.
- **Adaptive Router** — выбирает подходящего агента или набор агентов.
- **Research Agent** — выполняет исследовательские задачи и собирает сведения.
- **Coding Agent** — помогает с задачами по программированию.
- **Math Agent** — решает математические задачи.
- **Critic Agent** — проверяет результаты и отмечает возможные ошибки.

Это планируемые роли, а не уже работающие компоненты.

## Структура проекта

```text
src/
├── agents/          # Загрузка и проверка конфигурации ролей
├── routing/         # Маршрутизация задач
├── coordination/   # Координация и распределение задач
├── communication/  # Обмен сообщениями между компонентами
├── evaluation/     # Оценка качества и производительности
├── security/       # Проверка сообщений и контроль полномочий
├── llm/            # LLMClient и локальный Model Registry
└── tools/          # Инструменты для агентов (будущие)
config/
├── models.yaml      # Реальные модели и их характеристики
└── agents.yaml      # Привязки ролей к логическим ID моделей
notebooks/           # Jupyter Notebook для исследований
experiments/
├── configs/         # Конфигурации экспериментов
└── results/         # Локальные результаты экспериментов
tests/               # Автоматические тесты
```

## Технологии

- Python
- OpenRouter API через совместимый SDK `openai`
- `python-dotenv` для локальной загрузки переменных из `.env`
- PyYAML для чтения и проверки локальных конфигураций
- Jupyter Notebook для исследований и экспериментов

`LLMClient` и `ModelRegistry` реализованы. Специализированные агенты,
маршрутизация и координация пока не реализованы.

## Локальная установка

Клонируйте репозиторий и перейдите в его каталог:

```bash
git clone <URL-репозитория>
cd multi-agent-llm-system
```

Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

В Windows PowerShell окружение активируется командой:

```powershell
.venv\Scripts\Activate.ps1
```

## Установка зависимостей

Установите зависимости из `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Настройка `.env`

Скопируйте шаблон и добавьте свой ключ OpenRouter в локальный `.env`:

```bash
cp .env.example .env
```

Задайте свой `OPENROUTER_API_KEY` в локальном `.env`. Не добавляйте реальные ключи
в исходный код или `.env.example`. Файл `.env` исключён из Git.

## Проверка LLMClient

Обычные mock-тесты не обращаются к OpenRouter и не расходуют API-лимиты:

```bash
python -m pytest tests/test_llm_client.py -v
```

Для ручной проверки двух моделей выполните отдельную команду:

```bash
python -m tests.integration.test_openrouter
```

Она отправляет по одному реальному запросу для каждой модели и выводит JSON с
ответами, фактическими ID моделей, временем выполнения и статистикой токенов.
Этот запуск требует API-ключа и расходует лимиты OpenRouter. ID по умолчанию:

- `nvidia/nemotron-3-ultra-550b-a55b:free` — NVIDIA Nemotron 3 Ultra;
- `cohere/north-mini-code:free` — Cohere North Mini Code.

При необходимости ID можно переопределить переменными окружения
`OPENROUTER_ORCHESTRATOR_MODEL` и `OPENROUTER_CODING_MODEL`.

Чтобы проверить, что локальный секрет не отслеживается Git:

```bash
git check-ignore -v .env
git ls-files --error-unmatch .env.example
```

Первая команда показывает правило, исключающее `.env`; вторая успешно
завершается, если безопасный шаблон `.env.example` отслеживается Git.

Полноценный запуск multi-agent приложения появится после реализации основных компонентов.

## Реестр моделей

`config/models.yaml` хранит модели по логическим идентификаторам (`reasoning`,
`coding`). Чтобы добавить модель, добавьте в этот файл запись с провайдером,
реальным `model_id`, стоимостью, размером контекста и возможностями. Для
неизвестной стоимости или размера контекста используйте `null`. Реестр проверяет
схему и не обращается к OpenRouter.

`config/agents.yaml` связывает роли с логическими идентификаторами. Несколько
ролей могут ссылаться на один ID, не копируя характеристики модели.

```python
from src.agents import AgentConfigRegistry
from src.llm import LLMClient

agents = AgentConfigRegistry()
agent_config = agents.get_agent("coding")

with LLMClient() as client:
    response = client.generate(
        model=agent_config.model_id,
        messages=[{"role": "user", "content": "Hello!"}],
        **agent_config.generation_params,
    )
```

`AgentConfigRegistry` проверяет YAML, находит модель по логическому ID через
`ModelRegistry` и возвращает реальный `model_id` вместе с параметрами генерации.
Поэтому замену модели или её параметров можно выполнить в YAML без изменения
Python-кода потребителя. Роли могут ссылаться на одну запись в `models.yaml`.

## Локальный Message Bus

`src.communication.MessageBus` — синхронная внутрипроцессная доставка сообщений
между зарегистрированными идентификаторами агентов. `register_agent(agent_id)`
создаёт входящую очередь, `send(message)` доставляет `Message` только указанному
`receiver`, а `receive(agent_id)` сразу возвращает самое старое сообщение либо
`None`, если очередь пуста. Очереди работают по FIFO. Неизвестный отправитель
вызывает `UnknownSenderError`, неизвестный получатель — `UnknownReceiverError`.
Агент с непрочитанными сообщениями не может быть удалён через `unregister_agent`.

Для JSONL-событий передайте существующий `Logger` в `MessageBus(logger=logger)`.
Логируются метаданные доставки без payload; сбой записи не отменяет уже
выполненную доставку и доступен через `last_logging_error`. Без Logger Bus
работает без записи событий. Очереди локальны памяти процесса: нет персистентности,
широковещания, асинхронной доставки или автоматических повторов.

```python
agents.has_agent("coding")
```

Проверки конфигураций запускаются командами:

```bash
python -m pytest tests/test_model_registry.py -v
python -m pytest tests/test_agent_config.py -v
```

## Межагентное делегирование

`src.coordination.DelegationManager` координирует передачу отдельной подзадачи
между зарегистрированными агентами. Он использует существующие `TaskStore`,
`MessageBus`, `Message` и `Logger`: создаёт дочернюю задачу, отправляет
`HELP_REQUEST`, запускает получателя через его существующий метод и возвращает
`RESULT` исходному исполнителю. Завершение дочерней задачи не завершает родителя.

Начальные правила разрешают направления `research_agent → coding_agent` и
`coding_agent → math_agent`; у Math Agent нет следующего получателя. Правила
можно переопределить при создании менеджера. Максимальная глубина равна 3
(корень имеет глубину 0), а на одну корневую задачу разрешено не более 5
успешно отправленных передач. Повтор агента блокируется по цепочке родителей;
одинаковый исполнитель в независимых ветвях допустим. Для защиты от повтора
запроса можно передать стабильный `request_id` в `request_help()`.

Пример обмена Research → Coding → Math (ID ниже должны быть зарегистрированы в
MessageBus, а агенты переданы менеджеру под соответствующими логическими ID):

```python
from src.coordination import DelegationManager, Task, TaskStore
from src.communication import MessageBus

store = TaskStore()
bus = MessageBus()
for agent_id in ("research_agent", "coding_agent", "math_agent"):
    bus.register_agent(agent_id)
store.create_task(Task(task_id="root", description="Исследовать задачу"))
store.assign_task("root", "research_agent")
manager = DelegationManager(
    store,
    bus,
    agents={"research_agent": research_agent,
            "coding_agent": coding_agent,
            "math_agent": math_agent},
)
coding_help = manager.request_help(
    parent_task_id="root", initiator_id="research_agent",
    recipient_id="coding_agent", description="Подготовить прототип",
    context={"requirements": ["..." ]}, request_id="root-coding-v1",
)
# Получатель может запросить вложенную помощь до выполнения своей HELP_REQUEST.
coding_request = bus.receive("coding_agent")
math_help = manager.request_help(
    parent_task_id=coding_help.task_id, initiator_id="coding_agent",
    recipient_id="math_agent", description="Выполнить требуемый расчёт",
)
manager.receive_and_handle("math_agent")
math_result = bus.receive("coding_agent")
assert math_result is not None
manager.handle_result(math_result)  # результат математики теперь доступен в TaskStore
coding_request.payload["context"] = {
    "requirements": ["..."],
    "math_result": store.get_task(math_help.task_id).result,
}
manager.handle_help_request(coding_request)  # Coding выполняется и возвращает RESULT Research
manager.receive_and_handle("research_agent")
```

Менеджер работает синхронно и хранит счётчики/дедупликацию в памяти процесса;
это не очередь с персистентностью и не автоматический цикл исполнения. Клиент
приложения должен вызывать `receive_and_handle(agent_id)` для обработки inbox;
агент, решивший запросить вложенную помощь, должен получить своё `HELP_REQUEST`
из inbox и вызвать `request_help()` до передачи задачи своему обычному методу
выполнения.
Подзадачи запускаются только через известные интерфейсы агентов; код из
сообщений автоматически не исполняется. Unit-тесты используют поддельных
исполнителей и не обращаются к LLM API или Tavily.

## Graceful Degradation для веб-поиска

`SearchTool` ведёт состояние доступности Tavily (`enabled`, `disabled_reason`)
в памяти процесса. При подтверждённом исчерпании плана или лимита — HTTP 432 или
433 — поиск отключается до ручного `research_agent.search_tool.reset()`.
Авторизационная ошибка 401 и запрет доступа 403 также останавливают дальнейшие
попытки, но не обозначаются как исчерпанная квота. HTTP 429 считается временным
rate limit; timeout, сетевые ошибки и ответы 5xx не открывают circuit breaker.
Повторных попыток с автоматическим retry нет.

В закреплённом проектом `tavily-python==0.8.5` SDK превращает HTTP 429 в
`UsageLimitExceededError`, а 403/432/433 — в общий `ForbiddenError` и теряет
код ответа. `SearchTool` оборачивает официальный клиент небольшим адаптером,
который сохраняет HTTP status для классификации; детали ответа и API-ключи в
логи не попадают. Классификация соответствует [справочнику Tavily по HTTP-ошибкам](https://help.tavily.com/articles/8645538886-understanding-http-errors).

Если поиск не выполнен, `ResearchAgent` продолжает синтез без источников и
возвращает `ResearchReport` с `search_performed=False`, `search_status` и
`search_unavailable_reason`. В отчёт не добавляются фиктивные ссылки. Если поиск
выполнен, но не дал пригодных источников, `search_performed=True` и статус
`no_results` отличают это от недоступности. `CriticAgent` помечает внешнюю
проверку фактов недоступной и не считает пустой список источников ошибкой
формата или подтверждением достоверности.

После проверки и восстановления доступа можно включить сервис вручную:

```python
research_agent.search_tool.reset()
assert research_agent.search_tool.enabled
```

Состояние circuit breaker не сохраняется после перезапуска процесса. В
`notebooks/test.ipynb` вручную проверьте обычный поиск, работу отчёта после
отключения Tavily, повторный вызов без повторного обращения к сервису, `reset()`
и поля `ResearchReport` без источников. Эти интеграционные проверки требуют
настроенных ключей и не выполняются unit-тестами.
