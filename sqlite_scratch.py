import os
import sqlite3
import sys
import traceback
from enum import Enum
from pathlib import Path
from sqlite3 import OperationalError
from time import sleep
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal


class Isolation(Enum):
    Exclusive = "EXCLUSIVE"
    Deferred = "DEFERRED"


def insert_i(i: int) -> int:
    with sqlite3.connect(
        "test_db.db", timeout=5, isolation_level=Isolation.Exclusive.value
    ) as connection:
        # see https://www.sqlite.org/wal.html, won't work on cluster :(
        # connection.execute("pragma journal_mode=wal")
        cursor = connection.cursor()
        cursor.execute(
            f"""
    INSERT INTO fits (repeat, run, fold, json) VALUES ({i}, 0, 0, "{{0: 'a', 1: 'b'}}")
    """
        )
        cursor.close()
        connection.commit()
        # connection.close()


def insert_junk(i: int) -> None:
    try:
        insert_i(i)
    except OperationalError as e:
        attempts = 0
        while attempts < 2:
            try:

                sleep(int(np.random.randint(attempts, 10)) )  #
                insert_i(9)
                return
            except OperationalError:
                attempts += 1
        print(f"Could not insert value {i}")


if __name__ == "__main__":
    connection = sqlite3.connect("test_db.db")
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS fits")
    cursor.execute(
        """
CREATE TABLE fits(
id INTEGER PRIMARY KEY AUTOINCREMENT,
repeat INTEGER,
run INTEGER,
fold INTEGER CHECK(fold >= 0 AND fold < 5),
json TEXT
)
"""
    )
    connection.commit()
    cursor.close()
    connection.close()
    #     cursor.execute("""
    # INSERT INTO fits (repeat, run, fold, json) VALUES (0, 0, 10, "{0: 'a', 1: 'b'}")
    # """)
    cluster = os.environ.get("CC_CLUSTER") is not None
    max_workers = 80 if cluster else 50
    n = 500 if cluster else 250
    process_map(insert_junk, list(range(250)), max_workers=max_workers)
