# Ноутбуки

Ноутбуки запускают скрипты src/; перед коммитом выводы и execution_count очищаются.

## Google Colab (GPU)

Открыть colab_gpu.ipynb в Colab, выбрать Runtime → Change runtime type → T4 GPU и выполнить Run all. По умолчанию запускаются два smoke теста для base/SFT на 16 примерах и полные zero-shot замеры. Новое обучение включается RUN_TRAINING=True.

Исходники клонируются из GitHub (REPO_REF можно заменить SHA коммита). Из внешнего basemodel_colab_bundle.zip импортируются только данные и готовый адаптер; SHA-256 проверяется. Архив загрузить в /content через Files → Upload to session storage. Вариант с Drive: части .part001/.part002 в MyDrive/BaseModel-vs-SFT-Colab. Сам пакет и веса в Git не хранятся.

При подключённом Drive результаты сохраняются туда. Иначе последняя ячейка скачивает ZIP, который нужно сохранить до завершения среды. Без CUDA ноутбук останавливается.

Методика, подтверждённые результаты и состав файлов: [docs/evaluation.md](../docs/evaluation.md).
