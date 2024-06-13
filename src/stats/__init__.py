from sqlite3 import Connection, Cursor
from typing import Any, Iterable

OA_DOI_COUNT: int = 7885681
OA_OAID_COUNT: int = 13435534
OA_CITATION_COUNT: int = 113563323
OAPM_ARXIV_PM_PAPERS_IN_OA: int = 14


def runOneValueSQLQuery(db: Connection, query: str) -> Iterable[Any]:
    """
    _runOneValueSQLQuery Execute an SQL query that returns one value

    :param db: An sqlite3.Connection object
    :type db: Connection
    :param query: A SQLite3 compatible query
    :type query: str
    :return: An iterator containing any value
    :rtype: Iterator[Any]
    """
    cursor: Cursor = db.execute(query)
    return cursor.fetchone()
