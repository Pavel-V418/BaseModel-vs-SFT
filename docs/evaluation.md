# Замеры base и SFT

17 ноября 2026 года завершены два полных zero-shot сравнения на Tesla T4 в Google Colab. База — Qwen/Qwen2.5-0.5B; SFT — та же база с LoRA sft_0.5b_nall_r16_lr1e-4_s42, обученной участником 3 на Kaggle. Base не дообучалась. Новый запуск обучения в Colab для этих результатов не выполнялся.

| Метрика, 1034 примера | Base | SFT |
|---|---:|---:|
| Execution accuracy | 24.47% (253) | 49.23% (509) |
| Exact match | 7.93% (82) | 31.62% (327) |
| Valid SQL rate | 61.61% (637) | 73.40% (759) |
| Stop rate | 76.98% (796) | 99.90% (1033) |

Execution accuracy выросла на 24.76 процентного пункта. Разбивка easy / medium / hard / extra находится в JSON метрик. Это проектная оценка на 1034 валидных примерах из Spider dev, а не официальный результат на скрытом Spider test.

## Одинаковая процедура

Один подготовленный test.jsonl и одинаковые русские prompt (схема, вопрос, SQL:), без chat template и без few-shot примеров. Общий src.evaluate вызывает load_model из src.utils; для SFT меняется только adapter. Генерация: do_sample=False, num_beams=1, max_new_tokens=256, batch=16, tokenizer.padding_side='left', seed=42. В исходной конфигурации измеренной Qwen eos_token_id=151643, num_beams по умолчанию равен 1. Версии окружения измерения сохранены в results/environment_colab.json и results/requirements-colab-measured.txt.

extract_sql берёт SQL из блока sql, если он есть, затем обрезает по первому разделителю ;, пустой строке, Вопрос: или <|endoftext|>. Запросы выполняются отдельно на SQLite, открытой через URI mode=ro. Authorizer разрешает чтение и запрещает изменения, ATTACH, VACUUM и административные операции; пустые ответы не считаются валидными. Progress handler прерывает запрос примерно через 5 секунд. Временные таблицы сортировки остаются в памяти.

Сравнение результатов сохраняет дубликаты; порядок строк учитывается, если в gold SQL есть ORDER BY. Вещественные числа округляются до 6 знаков, целые и BLOB сохраняются без потери точности. Exact match — схлопывание пробелов, удаление конечной точки с запятой и сравнение без учёта регистра; это не семантическая эквивалентность SQL. Stop rate — EOS встретился до последнего из 256 разрешённых токенов.

## Проверка сохранённых результатов

Архивы 20261117_145842_results.zip и 20261117_152306_results.zip содержат по одному полному base/SFT прогону и повторные smoke проверки на 16 примерах. Полные JSONL для каждой модели совпали побайтово между двумя архивами. В Git сохранена одна копия каждого полного результата.

При аудите исправлены обработка пустого SQL, ограничения SQLite, точность чисел/BLOB и выбор EOS. Все 2068 сохранённых ответов повторно выполнены исправленным evaluator (evaluation_version=2); ни одна строка флагов и ни одна метрика не изменились. Это повторная проверка сохранённых генераций без нового model inference. Новые хеши JSONL отличаются от исходных из-за сериализации/переводов строк; хеши обоих вариантов и архивов указаны в results/evaluation_manifest.json. Идентификаторы сохранены из измеренного test.jsonl; свежая подготовка данных после изменений master может дать новые строковые id и другой SHA.

Для независимой проверки с тем же внешним пакетом:

~~~bash
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --predictions results/base_0.5b_zero_eval.jsonl --mode zero --run_name verify_base --results_dir /tmp/verify_eval
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --adapter outputs/sft_0.5b_nall_r16_lr1e-4_s42/final --predictions results/sft_0.5b_zero_eval.jsonl --mode zero --run_name verify_sft --results_dir /tmp/verify_eval
~~~

Скрипт проверяет число записей, id, question и gold_sql перед повторной оценкой. --predictions не загружает модель; нужны JSONL данных и SQLite-базы. Итоговые overall и by_difficulty должны совпасть с опубликованными.

## Новый запуск на GPU

Открыть notebooks/colab_gpu.ipynb в Colab, выбрать T4 GPU и выполнить Run all. Ноутбук загружает исходники из ветки participant-2/evaluation; для фиксированной версии указать SHA коммита в REPO_REF. Код и вывод не требуют локального CPU inference. RUN_TRAINING=False оставляет готовый адаптер; True запускает новое обучение и создаёт новый эксперимент.

Для точного набора данных нужен внешний basemodel_colab_bundle.zip (116789255 байт; SHA-256 в manifest). Он уже передан участникам проекта; публичной ссылки на него в репозитории нет. Загрузить его через Files → Upload to session storage в /content. Альтернатива — две части .part001/.part002 в MyDrive/BaseModel-vs-SFT-Colab. Ноутбук проверяет SHA и извлекает только data/ и outputs/, сохраняя исходники из Git. Веса base скачиваются из Hugging Face. Результаты записываются на Drive, если он подключён, либо скачиваются последней ячейкой в ZIP; локальные файлы Colab исчезают после удаления среды.

Без этого пакета можно подготовить новый набор из официального Spider: см. scripts/download_spider.sh, затем python -m src.data --spider_dir data/spider --out_dir data --val_size 400 --seed 42. Текущий master добавляет id и нормализует вопрос, поэтому для сопоставления с историческим измерением необходимо сверить данные; считать такой запуск новым экспериментом. Готовый LoRA отдельно поставляется в results_sft_0.5b_nall_r16_lr1e-4_s42.zip, папка outputs/.../final. Хеш весов в manifest. Новое обучение не гарантирует побайтового совпадения адаптера.

Обычные команды генерации:

~~~bash
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --mode zero --run_name base_0.5b_zero
python -m src.evaluate --model Qwen/Qwen2.5-0.5B --adapter outputs/sft_0.5b_nall_r16_lr1e-4_s42/final --mode zero --run_name sft_0.5b_zero
~~~

Для короткой проверки добавить --limit 16; для few-shot — --mode few. Каждый запуск создаёт <run_name>_eval.jsonl и <run_name>_eval_metrics.json. Few-shot и модели другого размера этими результатами не покрыты.

## Что опубликовано и что осталось внешним

В Git входят evaluator, parser, тесты, чистый Colab ноутбук, зависимости, полный результат по каждому примеру, сводные метрики, версия окружения, контрольные суммы и небольшие отчёты обучения. Данные Spider, SQLite, ZIP, LoRA и базовые веса остаются внешними файлами.

BaseModel-vs-SFT-export.zip — резервная копия: исходники/данные совпадают с исходным bundle, дополнительно вложены оба каталога результатов. Отдельно публиковать его содержимое целиком не нужно. Архивы *_results.zip — подтверждение двух запусков. Присланный выполненный .ipynb содержит выводы этих запусков; в Git сохранена очищенная версия с подготовкой исправленных исходников.
