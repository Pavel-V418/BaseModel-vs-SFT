#!/usr/bin/env bash
# Скачивает Spider 1.0 ВМЕСТЕ с базами данных в data/spider/.
#
#   bash scripts/download_spider.sh            # в data/spider
#   bash scripts/download_spider.sh path/to/dir
#
# Источник: официальный архив spider_data.zip, ссылка с yale-lily.github.io/spider
# (Google Drive id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J). Нужен gdown: pip install gdown
set -euo pipefail

OUT_DIR="${1:-data/spider}"
GDRIVE_ID="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [ -f "$OUT_DIR/tables.json" ] && [ -d "$OUT_DIR/database" ]; then
  echo "Spider уже лежит в $OUT_DIR, пропускаю скачивание."
else
  command -v gdown >/dev/null 2>&1 || { echo "Нужен gdown: pip install gdown"; exit 1; }
  echo "Скачиваю spider_data.zip (~200 МБ)..."
  gdown "$GDRIVE_ID" -O "$TMP_DIR/spider_data.zip"

  # Если Google Drive отказал (лимит скачиваний), можно взять зеркало
  # с той же версией (sha256 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b):
  #   wget -O "$TMP_DIR/spider_data.zip" \
  #     https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip

  echo "Распаковываю..."
  unzip -q "$TMP_DIR/spider_data.zip" -d "$TMP_DIR"
  mkdir -p "$OUT_DIR"
  # В архиве всё лежит в папке spider_data/
  cp -r "$TMP_DIR"/spider_data/* "$OUT_DIR"/
fi

# Официальный скрипт оценки: нужен только для сверки eval_hardness в тестах.
mkdir -p third_party
if [ ! -f third_party/spider_evaluation.py ]; then
  curl -fsSL https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py \
    -o third_party/spider_evaluation.py || echo "WARN: не удалось скачать evaluation.py (не критично)"
fi

# Проверка: все нужные файлы на месте + один эталонный запрос выполняется.
for f in train_spider.json train_others.json dev.json tables.json; do
  [ -f "$OUT_DIR/$f" ] || { echo "ERROR: нет $OUT_DIR/$f"; exit 1; }
done
N_DB=$(find "$OUT_DIR/database" -name '*.sqlite' | wc -l)
echo "Баз данных: $N_DB"
echo "Проверочный запрос к concert_singer:"
sqlite3 "$OUT_DIR/database/concert_singer/concert_singer.sqlite" "SELECT count(*) FROM singer"
echo "Готово: $OUT_DIR"
