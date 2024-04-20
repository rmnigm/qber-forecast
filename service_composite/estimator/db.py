import sqlite3

import numpy as np

record_tuple = tuple[float | None, ...]
record_list = list[record_tuple]


class Database:
    """
    Data structure for sliding window of N Records, adapted for creation of catboost features
    """
    
    def __init__(self, db_file: str) -> None:
        """
        """
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS t1(emu REAL)
            '''
        )
        self.connection.commit()
        self.total_size = 0

    def is_full(self) -> bool:
        """
        Checks if SlidingWindow contains self.size records
        :return: bool, true if full
        """
        return self.total_size > 20
    
    def update(self, e_mu_prev: float) -> None:
        """
        Inserts new data into structure: adds latest record for all values except eMu
        and fills eMu value in previous record, pushes it to the SlidingWindow.
        :return: None
        """
        self.cursor.execute(
            '''
            INSERT INTO t1 VALUES(?)
            ''', e_mu_prev
        )
        self.connection.commit()
        self.total_size += 1

    def get_latest_rows(self, k) -> np.ndarray:
        data = self.cursor.execute(
            '''
            SELECT emu FROM t1 ORDER BY ROWID DESC LIMIT ?
            ''', k
        )
        return np.array(data.fetchall())

    def close(self):
        self.cursor.close()
        self.connection.close()
