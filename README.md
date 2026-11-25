# d16nn

Исследовательская реализация нейросети Qwen на Python/PyTorch с комплексными объектами `D16Complex` и `D16ComplexTensor`. Комплексное число хранится через непрерывные координаты фундаментального клина `[0°, 22,5°]` и секторную маску. Групповые преобразования соответствуют диэдральной группе **ℤ₈ ⋊ ℤ₂ порядка 16**.

В текущем проекте D16 используется в секторной реализации SwiGLU и в комплексном умножении для RoPE. Реализованы самостоятельная модель `D16QwenForCausalLM`, перенос весов Qwen 1:1, обучение с нуля, генерация с KV-кэшем и исследование орбитных связей между каналами MLP.

## Model2 — полный импорт и один шаг обучения

В `qwen/src/d16_qwen_model2.py` режим MLP по умолчанию — `bilinear`. В каждом слое вычисляется комплексное произведение `up_proj(h) * gate_proj(h)`, затем применяется `down_proj`. Функция `import_qwen_weights` из `qwen/src/d16_qwen_importer.py` импортирует все слои Qwen2.5-0.5B с аналитическим дополнением `d16_sat_rope_gate`. Вещественная часть хранит исходный вес. Мнимая часть вычисляется из весов партнёрской проекции или пары координат. У коэффициентов RMSNorm и смещений V она равна нулю.

Для воспроизведения эксперимента в PowerShell укажите локальный каталог Hugging Face с весами и токенизатором Qwen, а также новое имя выходной папки.

```powershell
conda activate torch_env
Set-Location D:\repo\d16nn
$snapshot = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987'
python qwen/src/run_model2_one_step.py $snapshot runs/model2_one_step --lr 1e-5
```

Скрипт сверяет вещественную компоненту всех импортированных значений с оригиналом и делает один шаг `D16AdamW` по всей сети. Обучаются обе координаты всех весов, включая общую матрицу эмбеддингов и выходного слоя. В каталоге запуска сохраняются `imported.pt`, обновлённая модель `step-0001.pt`, состояние оптимизатора `optimizer-0001.pt` и `metrics.json`. В метриках записаны входные токены, NLL, perplexity, количество верных предсказаний и число изменившихся весов по каждому слою. Полная модель занимает около 5,9 ГБ, моменты оптимизатора всей сети — ещё 7,9 ГБ. Модель можно открыть через `load_d16_checkpoint`.

Чтобы сделать ещё один шаг от сохранённых весов **и моментов AdamW**, выполните команду с номером завершённого шага. Второй шаг сохранит отдельные `step-0002.pt`, `optimizer-0002.pt` и `metrics-step-0002.json`.

```powershell
python qwen/src/run_model2_one_step.py $snapshot runs/model2_one_step --from-step 1 --lr 1e-5
```

## Представление числа и архитектура

Объект D16 хранит `(x, y, rotation)`, где

```text
0 <= y <= (sqrt(2) - 1) * x
```

| Состояние | Маска |
|---|---|
| Внутренняя точка сектора 0–15 | Один установленный бит |
| Граница соседних секторов | Два соседних бита |
| Ноль — `Indifferent`, состояние 16 | `0xFFFF`, координаты `(0, 0)` |
| Неопределённое значение — `Undefined`, состояние 17 | `0x0000` |

Координаты внутри клина сохраняют непрерывную информацию об исходном числе. Выбор сектора сам по себе не является угловым квантованием. Маска содержит 16 значащих битов; в тензорной реализации она хранится в целочисленном тензоре PyTorch.

**MLP.** Для каждого канала строится `z = L + iE`, где `L = up_proj(h)`, `E = gate_proj(h)`. Метод `D16QwenMLP._eval_d16` непосредственно вычисляет секторную формулу отклика:

$$
h_i = L_i\operatorname{SiLU}(E_i)
    = \frac12\operatorname{Im}(z_i^2)\,\sigma(E_i).
$$

Для координат клина модуль произведения `LE` равен `x*y` в осевых секторах и `(x*x - y*y)/2` в диагональных; знак и восстановление `E` определяются сектором. Поэтому `_eval_d16` выбирает нужную ветвь формулы квадрата по маске. На выходе выполняется `down_proj`.

**RoPE.** Пары компонент Q и K кодируются как `D16ComplexTensor` и умножаются на `cos(θ) + i sin(θ)` внутри клина. Результат декодируется обратно в компоненты голов внимания.

**Остальные части модели.** Линейные проекции, attention scores, softmax, RMSNorm, embeddings и остаточный поток используют вещественные тензоры. Конвертация сохраняет число каналов и матрицы весов. Параметры конфигурации по умолчанию соответствуют Qwen2.5-0.5B: 24 слоя, скрытая размерность 896, 4864 канала MLP, 14 голов внимания и 2 KV-головы.

## Структура репозитория

```text
d16nn/
├── README.md
└── qwen/
    ├── src/
    │   ├── d16_complex.py          # Скалярная и тензорная арифметика D16
    │   ├── d16_qwen_model.py       # Модель, MLP, RoPE, генерация, телеметрия
    │   ├── d16_qwen_converter.py   # Перенос весов и анализ орбит
    │   └── d16_vis_utils.py        # Таблицы и визуализация
    ├── tests/
    │   ├── test_d16_complex.py
    │   ├── test_d16_qwen_model.py
    │   └── test_d16_qwen_converter.py
    └── ipynb/
        ├── d16_qwen_pipeline.ipynb
        └── d16_conversion_inspector.ipynb
```

## Окружение и зависимости

Для модели и тестов нужен PyTorch. Скалярный `D16Complex` может использоваться без PyTorch. Дополнительные зависимости:

| Задача | Пакеты |
|---|---|
| Загрузка весов `.safetensors` | `safetensors` |
| Токенизатор Qwen и сравнение с Hugging Face | `transformers` |
| Визуализация | `numpy`, `matplotlib` |
| Блокноты | `jupyterlab`, `ipykernel` |

Для существующего окружения Anaconda:

```powershell
conda activate torch_env
Set-Location D:\repo\d16nn
python -c "import torch; print(torch.__version__)"
```

Если дополнительные пакеты ещё не установлены:

```powershell
python -m pip install safetensors transformers numpy matplotlib jupyterlab ipykernel
```

В новом окружении сначала установите подходящую сборку PyTorch. Отдельного устанавливаемого пакета `d16nn` пока нет: примеры ниже добавляют `qwen/src` в `sys.path`. Все Python-примеры предполагают рабочую папку **корня репозитория**.

## Быстрая проверка

```powershell
python -m unittest discover -s qwen/tests -v
```

На 16 ноября 2026 года проходят **27 тестов**: арифметика и границы D16, операции группы, секторная формула MLP, D16-RoPE, конвертация весов, градиенты, обучение небольшой модели с нуля, KV-кэш, генерация и каноничность состояния `Indifferent`. Проверенное окружение: Python 3.12.8, PyTorch 2.14.1+cu126; тесты выполнялись на CPU.

Тесты создают небольшие модели и локальные временные веса. Для них не требуется скачивать Qwen2.5-0.5B. Результаты этих тестов описывают проверенные свойства реализации; качество конкретного перенесённого checkpoint оценивается отдельно на выбранном корпусе.

## Пример комплексной арифметики

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("qwen/src").resolve()))
from d16_complex import D16Complex

z = D16Complex.from_xy(2.0, 3.0)
print(z.square().to_xy())          # Приблизительно (-5.0, 12.0)
print(z.conjugate().to_xy())       # (2.0, -3.0)
print(z.apply_group_op(1).to_xy())  # Поворот на +45°
```

Скалярный класс поддерживает сложение, вычитание, умножение, квадрат и групповые операции. `D16ComplexTensor` предоставляет векторизованные кодирование/декодирование, умножение, квадрат и операции над масками.

## Небольшая модель: forward, backward и генерация

Этот пример можно сохранить как отдельный Python-скрипт в корне репозитория. Он создаёт модель со случайными весами; для осмысленной генерации текста нужны обучение или перенос checkpoint.

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path("qwen/src").resolve()))
from d16_qwen_model import D16QwenConfig, D16QwenForCausalLM

torch.manual_seed(42)
torch.set_num_threads(4)
config = D16QwenConfig.tiny_test_config(vocab_size=256)
model = D16QwenForCausalLM(config)
tokens = torch.randint(0, config.vocab_size, (2, 16))

output = model(tokens, labels=tokens)
output.loss.backward()
print(output.logits.shape)           # torch.Size([2, 16, 256])
print(float(output.loss.detach()))

model.eval()
generated = model.generate(tokens[:, :4], max_new_tokens=4, use_cache=True)
print(generated.shape)               # torch.Size([2, 8])
```

Для обучения используются обычные оптимизаторы PyTorch, например `torch.optim.AdamW(model.parameters(), ...)`. При передаче `labels` модель сама сдвигает логиты и метки для предсказания следующего токена.

## Перенос локальных весов Qwen

`QwenToD16Converter` загружает каталог с `config.json` и весами `*.safetensors` либо `*.bin`/`*.pt`. Скачивание checkpoint не выполняется. Для работы с текстом в каталоге также нужны файлы исходного токенизатора.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("qwen/src").resolve()))
from d16_qwen_converter import QwenToD16Converter
from transformers import AutoTokenizer

checkpoint_dir = Path(r"D:\models\Qwen2.5-0.5B")  # Укажите свой каталог
model, report = QwenToD16Converter.from_pretrained_dir(
    checkpoint_dir,
    indiff_eps=0.0,
    analyze_orbits=False,
    device="cpu",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
input_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
result = model.generate(
    input_ids,
    max_new_tokens=24,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
print(tokenizer.decode(result[0], skip_special_tokens=True))
```

При стандартном dtype PyTorch конвертер создаёт модель в `float32`. Для уже загруженных весов предусмотрен `QwenToD16Converter.convert_state_dict(state_dict, config_dict, ...)`.

Базовый режим сохраняет каждый канал MLP 1:1. Параметр `analyze_orbits=False` отключает поиск орбит, но сохраняет перенос весов. При `True` заполняются диагностические четвёрки, оценки сходства и гистограммы операций в `report.layer_reports`.

`activate_bitwise_orbit_links(model, report, pairs_per_layer=..., min_orbit_score=...)` экспериментально связывает выбранные каналы с опорными объектами через преобразования масок. Это изменяет вычисление модели и требует отдельной оценки качества. Число объектов и матриц при связывании сохраняется. Значение `pairs_per_layer=0` очищает связи.

## Телеметрия и управление секторами

Для модели и входных токенов из примера небольшой сети:

```python
model.set_indifferent_threshold(0.0)
model.set_collect_d16_stats(True, reset=True)
with torch.no_grad():
    model(tokens)

summary = model.get_d16_state_summary()
print(summary["state_counts"])       # 18 состояний
print(summary["indifferent_pct"])

# Групповое преобразование: +45° в MLP слоя 0.
model.set_sector_steering(mode="group_op", shift=1, target_layers=[0])
# Возврат к исходному режиму:
model.set_sector_steering(mode="identity")
model.set_collect_d16_stats(False)
```

`set_indifferent_threshold(eps)` переводит входное число в ноль при **`|L| + |E| <= eps`**. Это порог входных координат. При `eps=0` остаётся только точный ноль.

Режимы `active_not_logic`, `active_to_dislike`, `active_swap_le` и `active_to_indifferent` применяются к состояниям верхней полуплоскости. При заданном `channels_by_layer` выбираются указанные каналы последнего токена; без выбора каналов преобразование применяется ко всем подходящим позициям. Переход в `Indifferent` устанавливает маску `0xFFFF` и координаты `(0, 0)`.

**Нумерации операций различаются.** `apply_group_op` и режим `group_op` используют восемь поворотов, затем восемь отражений. Орбитный анализатор, `OP_NAMES`, `set_orbit_links` и `apply_all_16_d16_ops_u16` используют порядок «осевая подгруппа, затем смежный класс 45°»:

| ID | `apply_group_op` / `group_op` | Орбитный анализатор / `set_orbit_links` |
|---|---|---|
| 1 | Поворот +45° | Поворот 180° |
| 8 | Комплексное сопряжение | Поворот +45° |

Номера секторов 0–15 также следует отличать от ID преобразований.

## Сохранение и восстановление модели

`D16QwenForCausalLM` — самостоятельный `nn.Module`. Для сохранения конфигурации и весов:

```python
from dataclasses import asdict

torch.save(
    {"config": asdict(model.config), "state_dict": model.state_dict()},
    "d16_model.pt",
)

saved = torch.load("d16_model.pt", map_location="cpu", weights_only=True)
restored = D16QwenForCausalLM(D16QwenConfig(**saved["config"]))
restored.load_state_dict(saved["state_dict"])
restored.eval()
```

Орбитные связи, текущая телеметрия и настройки секторных преобразований не сохраняются в `state_dict`; после загрузки они задаются отдельно. Загрузка исходного checkpoint Qwen выполняется через конвертер, а собственного checkpoint D16 — через `load_state_dict`.

Текущий `generate` поддерживает `max_new_tokens`, `do_sample`, `temperature`, `eos_token_id` и `use_cache`. Пример генерации выше рассчитан на одиночный запрос без padding; аргументы Hugging Face вроде `top_p` или `attention_mask` в этот метод не передаются.

## Блокноты

Запускайте Jupyter из папки `qwen`, чтобы поиск `src` внутри блокнотов работал корректно:

```powershell
conda activate torch_env
Set-Location D:\repo\d16nn\qwen
python -m ipykernel install --user --name torch_env --display-name "Python (torch_env)"
python -m jupyterlab
```

Выберите ядро `Python (torch_env)` и откройте:

- `ipynb/d16_qwen_pipeline.ipynb` — обучение небольшой модели с нуля, генерация и телеметрия.
- `ipynb/d16_conversion_inspector.ipynb` — экспериментальный инспектор локального Qwen checkpoint, орбитных связей, генерации и perplexity.

Инспектор ищет уже скачанный Qwen2.5-0.5B в кэше Hugging Face. Его ячейки с анализом орбит относятся к отдельному исследовательскому режиму. Во вводном тексте pipeline-блокнота сохранились описания прежних `sector_re_gain` / `sector_im_gain`; текущую архитектуру определяет `qwen/src/d16_qwen_model.py`, описанный в этом README.
