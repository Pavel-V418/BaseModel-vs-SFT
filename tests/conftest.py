"""Мини-Spider для тестов: такая же структура файлов, как у настоящего, но 8 крошечных баз.

Тесты не зависят от скачанного Spider и проходят без интернета.
"""

import json
import sqlite3

import pytest


# ---- разобранные SQL в формате поля "sql" из Spider -------------------------
def col(cid, agg=0):
    return [agg, cid, False]


def vu(cid, agg=0):
    return [0, col(cid, agg), None]


def sql_dict(select, tables=(1,), where=(), group=(), order=None, limit=None,
             union=None, conds=()):
    return {
        "select": [False, [[a, vu(c)] for a, c in select]],
        "from": {"table_units": [["table_unit", t] for t in tables], "conds": list(conds)},
        "where": list(where), "groupBy": list(group), "having": [],
        "orderBy": order or [], "limit": limit,
        "intersect": None, "union": union, "except": None,
    }


def cond(cid, op=3, val=20.0, not_op=False):
    return [not_op, op, vu(cid), val, None]


EASY = sql_dict(select=[(3, 0)])                                       # SELECT count(*) FROM singer
MEDIUM = sql_dict(select=[(0, 9), (0, 10)], where=[cond(13)])          # 2 колонки + WHERE
HARD = sql_dict(select=[(0, 9)], where=[cond(13)], order=["desc", [vu(13)]], limit=1)
EXTRA = sql_dict(select=[(0, 9)], where=[cond(13)], order=["desc", [vu(13)]],
                 union=sql_dict(select=[(0, 9)]))

# ---- concert_singer: настоящая схема из Spider (урезанные данные) ------------
CONCERT_SINGER = {
    "db_id": "concert_singer",
    "table_names_original": ["stadium", "singer", "concert", "singer_in_concert"],
    "table_names": ["stadium", "singer", "concert", "singer in concert"],
    "column_names_original": [
        [-1, "*"],
        [0, "Stadium_ID"], [0, "Location"], [0, "Name"], [0, "Capacity"],
        [0, "Highest"], [0, "Lowest"], [0, "Average"],
        [1, "Singer_ID"], [1, "Name"], [1, "Country"], [1, "Song_Name"],
        [1, "Song_release_year"], [1, "Age"], [1, "Is_male"],
        [2, "concert_ID"], [2, "concert_Name"], [2, "Theme"], [2, "Stadium_ID"], [2, "Year"],
        [3, "concert_ID"], [3, "Singer_ID"],
    ],
    "column_types": ["text", "number", "text", "text", "number", "number", "number", "number",
                     "number", "text", "text", "text", "text", "number", "others",
                     "number", "text", "text", "text", "text", "number", "text"],
    "primary_keys": [1, 8, 15, 20],
    "foreign_keys": [[18, 1], [21, 8], [20, 15]],
}
CONCERT_SQL = """
CREATE TABLE stadium (Stadium_ID int PRIMARY KEY, Location text, Name text, Capacity int,
  Highest int, Lowest int, Average int);
CREATE TABLE singer (Singer_ID int PRIMARY KEY, Name text, Country text, Song_Name text,
  Song_release_year text, Age int, Is_male bool);
CREATE TABLE concert (concert_ID int PRIMARY KEY, concert_Name text, Theme text,
  Stadium_ID text, Year text);
CREATE TABLE singer_in_concert (concert_ID int, Singer_ID text, PRIMARY KEY (concert_ID, Singer_ID));
INSERT INTO stadium VALUES (1, 'Raith Rovers', 'Stark''s Park', 10104, 4812, 1294, 2106);
INSERT INTO singer VALUES (1, 'Joe Sharp', 'Netherlands', 'You', '1992', 52, 'F');
INSERT INTO singer VALUES (2, 'Timbaland', 'United States', 'Dangerous', '2008', 32, 'T');
INSERT INTO singer VALUES (3, 'Justin Brown', 'France', 'Hey Oh', '2013', 29, 'T');
INSERT INTO concert VALUES (1, 'Auditions', 'Free choice', '1', '2014');
INSERT INTO singer_in_concert VALUES (1, '2');
"""


def simple_db(db_id):
    """Однотабличная база item(item_id, name, price)."""
    schema = {
        "db_id": db_id,
        "table_names_original": ["item"], "table_names": ["item"],
        "column_names_original": [[-1, "*"], [0, "item_id"], [0, "name"], [0, "price"]],
        "column_types": ["text", "number", "text", "number"],
        "primary_keys": [1], "foreign_keys": [],
    }
    sql = ("CREATE TABLE item (item_id int PRIMARY KEY, name text, price real);"
           "INSERT INTO item VALUES (1, 'pen', 1.5), (2, 'book', 12.0);")
    return schema, sql


def ex(db_id, question, query, sql=EASY):
    return {"db_id": db_id, "question": question, "query": query, "sql": sql,
            "question_toks": question.split(), "query_toks": query.split()}


@pytest.fixture(scope="session")
def fake_spider(tmp_path_factory):
    root = tmp_path_factory.mktemp("spider")
    tables, dbs = [CONCERT_SINGER], {"concert_singer": CONCERT_SQL}
    train_dbs = [f"shop_{i}" for i in range(6)]
    for db_id in train_dbs + ["museum_test"]:
        schema, sql = simple_db(db_id)
        tables.append(schema)
        dbs[db_id] = sql
    for db_id, sql in dbs.items():
        d = root / "database" / db_id
        d.mkdir(parents=True)
        conn = sqlite3.connect(d / f"{db_id}.sqlite")
        conn.executescript(sql)
        conn.commit()
        conn.close()
    (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")

    train = [
        ex("concert_singer", "How many singers do we have?", "SELECT count(*) FROM singer"),
        ex("concert_singer", "Show name and country of singers older than 20.",
           "SELECT Name ,  Country FROM singer\n WHERE Age > 20 ;", MEDIUM),
        ex("concert_singer", "Name of the oldest singer?",
           "SELECT Name FROM singer ORDER BY Age DESC LIMIT 1", HARD),
        # битый пример: такой колонки нет
        ex("concert_singer", "Broken one", "SELECT salary FROM singer"),
    ]
    for i, db_id in enumerate(train_dbs):
        for j in range(5 + i):
            train.append(ex(db_id, f"How many items? ({j})", "SELECT count(*) FROM item"))
    test = [
        ex("museum_test", "How many items are there?", "SELECT count(*) FROM item"),
        ex("museum_test", "List the most expensive item name and the cheapest one.",
           "SELECT name FROM item WHERE price > 10 UNION SELECT name FROM item WHERE price < 2",
           EXTRA),
    ]
    (root / "train_spider.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "train_others.json").write_text("[]", encoding="utf-8")
    (root / "dev.json").write_text(json.dumps(test), encoding="utf-8")
    return root
