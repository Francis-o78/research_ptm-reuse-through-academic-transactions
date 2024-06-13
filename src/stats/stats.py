from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any, Iterable, List
from urllib.parse import urlparse

import click
import pandas
import pandas as pd
from humanize import intcomma
from pandas import DataFrame, Series
from progress.spinner import Spinner
from pyfs import isFile, resolvePath

from src.stats import (
    OA_CITATION_COUNT,
    OA_DOI_COUNT,
    OA_OAID_COUNT,
    OAPM_ARXIV_PM_PAPERS_IN_OA,
    runOneValueSQLQuery,
)


def _createDFGeneratorFromSQL(
    db: Connection,
    query: str,
    chunkSize: int = 10000,
) -> Iterable[DataFrame]:
    """
    _createDFGeneratorFromSQL Return a generator of Pandas DataFrames to process large SQL query results

    :param db: An sqlite3.Connection object
    :type db: Connection
    :param query: The SQLite3 compatible query to run
    :type query: str
    :param chunkSize: The number of rows per DataFrame to return, defaults to 10000
    :type chunkSize: int, optional
    :return: A generator of Pandas DataFrames
    :rtype: _type_
    :yield: A pandas.DataFrame
    :rtype: Iterable[DataFrame]
    """
    return pd.read_sql_query(query, con=db, chunksize=chunkSize)


def _createDFFromSQL(db: Connection, query: str) -> DataFrame:
    """
    _createDFFromSQL Return a Pandas DataFrame of the results from a SQL query

    Only useful for small SQL query results (less than 10,000 rows returned)

    :param db: An sqlite3.Connection object
    :type db: Connection
    :param query: The SQLite3 compatible query to run
    :type query: str
    :return: A pandas.DataFrame of the SQL query results
    :rtype: DataFrame
    """
    return pd.read_sql_query(query, con=db)


def _extractNetLoc(url: str) -> str:
    """
    _extractNetLoc Return the netloc attribute of a URL if it exists

    :param url: A URL to parse
    :type url: str
    :return: the netloc attribute of the URL
    :rtype: str
    """
    return urlparse(url=url).netloc


def _convertToArXivDOI(arxivURL: str) -> str:
    """
    _convertToArXivDOI Given an arXiv compliant URL, create the arXiv DOI from it

    :param arxivURL: An arXiv compatible URL (https://arxiv.org/abs/)
    :type arxivURL: str
    :return: An arXiv compatible DOI
    :rtype: str
    """
    return arxivURL.replace(
        "https://arxiv.org/abs/",
        "10.48550/arxiv.",
    )


def _standardizeText(text: str) -> str:
    """
    _standardizeText Remove trailing whitespace(s) and make text lower case

    :param text: The text to format
    :type text: str
    :return: A formatted string of the input text
    :rtype: str
    """
    return text.strip().lower()


def connectToDB(dbPath: Path) -> Connection:
    """
    connectToDB Connect to a SQLite3 database and return the sqlite3.Connection object

    :param dbPath: Filepath to a SQLite3 database
    :type dbPath: Path
    :return: The sqlite3.Connection object
    :rtype: Connection
    """
    return Connection(database=dbPath)


def oa_CountPapersByDOI(
    oaDB: Connection,
    returnDefault: bool = True,
) -> int:
    """
    oa_CountPapersByDOI Count the number of papers within an OpenAlex dataset by the unique DOI

    DOIs that contain a space are excluded from the count

    :param oaDB: A sqlite3.Connection object to an OpenAlex dataset
    :type oaDB: Connection
    :param returnDefault: Skip computing the value and use the pre-computed value, defaults to True
    :type returnDefault: bool, optional
    :return: The number of papers in the dataset that have a unqiue DOI
    :rtype: int
    """
    if returnDefault:
        return OA_DOI_COUNT
    else:
        doiCount: int = 0
        query: str = "SELECT DISTINCT doi FROM works"
        dfs: Iterable[DataFrame] = _createDFGeneratorFromSQL(db=oaDB, query=query)

        with Spinner(
            message="Counting number of papers in OpenAlex by DOI...",
        ) as spinner:
            df: DataFrame
            for df in dfs:
                df["doi"] = df["doi"].replace(to_replace=" ", value=None)
                df.dropna(inplace=True)
                doiCount += df.shape[0]
                spinner.next()

        return doiCount


def oa_CountPapersByOAID(
    oaDB: Connection,
    returnDefault: bool = True,
) -> int:
    """
    oa_CountPapersByOAID Count the number of papers within an OpenAlex dataset by the unique OpenAlex ID

    :param oaDB: A sqlite3.Connection object to an OpenAlex dataset
    :type oaDB: Connection
    :param returnDefault: Skip computing the value and use the pre-computed value, defaults to True
    :type returnDefault: bool, optional
    :return: The number of papers in the dataset that have a unqiue OpenAlex ID
    :rtype: int
    """
    query: str = "SELECT COUNT(DISTINCT oa_id) FROM works"
    if returnDefault:
        return OA_OAID_COUNT
    else:
        return runOneValueSQLQuery(db=oaDB, query=query)[0]


def oa_ProportionOfValidPapers(oaIDCount: int, oaDOICount: int) -> float:
    """
    oa_ProportionOfValidPapers Compute the proportion of papers with a valid DOI in an OpenAlex dataset

    :param oaIDCount: Number of papers that have an OpenAlex ID
    :type oaIDCount: int
    :param oaDOICount: Number of papers that have a DOI
    :type oaDOICount: int
    :return: The proportion of DOI papers over OpenAlex ID papers
    :rtype: float
    """
    return oaDOICount / oaIDCount


def oa_CountCitations(
    oaDB: Connection,
    returnDefault: bool = True,
) -> int:
    """
    oa_CountCitations Return the number of citations in an OpenAlex database

    :param oaDB: A  sqlite3.Connection object to an OpenAlex dataset
    :type oaDB: Connection
    :param returnDefault: Skip computing the value and use the pre-computed value, defaults to True
    :type returnDefault: bool, optional
    :return: The number of citations in the OpenAlex dataset
    :rtype: int
    """
    query: str = "SELECT id FROM cites ORDER BY id DESC LIMIT 1"
    if returnDefault:
        return OA_CITATION_COUNT
    else:
        return runOneValueSQLQuery(db=oaDB, query=query)[0]


def oapm_ProportionOfPMPapersInOA(
    oaPapers: int,
    pmPapers: int,
) -> float:
    """
    oapm_ProportionOfPMPapersInOA Compute the proportion of PeaTMOSS papers captured by OpenAlex

    :param oaPapers: Number of papers in a OpenAlex dataset
    :type oaPapers: int
    :param pmPapers: Number of papers in a PeaTMOSS dataset
    :type pmPapers: int
    :return: The proportion of PeaTMOSS papers over OpenAlex papers
    :rtype: float
    """
    return pmPapers / oaPapers


def oapm_CountPMArXivPapersInOA(
    pmDB: Connection,
    oaDB: Connection,
    returnDefault: bool = True,
) -> int:
    """
    oapm_CountPMArXivPapersInOA Count the number of PeaTMOSS arXiv papers in OpenAlex

    arXiv papers are determined by DOI

    :param pmDB: A sqlite3.Connection object of a PeaTMOSS database
    :type pmDB: Connection
    :param oaDB: A sqlite3.Connection object of an OpenAlex database
    :type oaDB: Connection
    :param returnDefault: Skip computing the value and use the pre-computed value, defaults to True
    :type returnDefault: bool, optional
    :return: The number of PeaTMOSS arXiv papers in OpenAlex
    :rtype: int
    """
    if returnDefault:
        return OAPM_ARXIV_PM_PAPERS_IN_OA

    count: int = 0

    oaQuery: str = "SELECT DISTINCT doi FROM works"

    arxivPMDF: DataFrame = pm_IdentifyPapersPublishedInArXiv(pmDB=pmDB)

    oaDFs: Iterable[DataFrame] = _createDFGeneratorFromSQL(
        db=oaDB,
        query=oaQuery,
    )

    with Spinner(
        message="Counting the number of PeaTMOSS papers published in arXiv that are captured in OpenAlex...",
    ) as spinner:
        df: DataFrame
        for df in oaDFs:
            df["doi"] = df["doi"].apply(_standardizeText)
            count += df[df["doi"].isin(arxivPMDF["url"])].shape[0]
            spinner.next()

    return count


def oapm_CountCitationsOfArXivPMPapers(
    pmDB: Connection,
    oaDB: Connection,
) -> Series:
    """
    oapm_CountCitationsOfArXivPMPapers Count the number of OpenAlex papers that cite PeatMOSS arXiv papers

    :param pmDB: A sqlite3.Connection of a PeaTMOSS database
    :type pmDB: Connection
    :param oaDB: A sqlite3.Connection of a OpenAlex database
    :type oaDB: Connection
    :return: A Series of the number of citations a PeaTMOSS arXiv paper recieved
    :rtype: Series
    """
    worksQuery: str = "SELECT oa_id, title FROM works"
    citesQuery: str = "SELECT reference FROM cites"

    relevantWorksDFs: List[DataFrame] = []
    relevantCitesDFs: List[DataFrame] = []

    pmDF: DataFrame = pm_IdentifyPapersPublishedInArXiv(pmDB=pmDB)
    pmDF["title"] = pmDF["title"].apply(_standardizeText)

    oaWorksDFs: Iterable[DataFrame] = _createDFGeneratorFromSQL(
        db=oaDB,
        query=worksQuery,
    )
    oaCitesDFs: Iterable[DataFrame] = _createDFGeneratorFromSQL(
        db=oaDB,
        query=citesQuery,
    )

    with Spinner(message="Identifying rows with relevant arXiv papers...") as spinner:
        df: DataFrame
        for df in oaWorksDFs:
            df["title"] = df["title"].apply(_standardizeText)
            relevantWorksDFs.append(df[df["title"].isin(pmDF["title"])])
            spinner.next()

    oaWorksDF: DataFrame = pandas.concat(
        objs=relevantWorksDFs,
        ignore_index=True,
    )

    with Spinner(message="Identifying rows that cite arXiv papers...") as spinner:
        df: DataFrame
        for df in oaCitesDFs:
            relevantCitesDFs.append(df[df["reference"].isin(oaWorksDF["oa_id"])])
            spinner.next()

    return pandas.concat(objs=relevantCitesDFs, ignore_index=True)[
        "reference"
    ].value_counts(sort=True)


def pm_CountPapersByID(pmDB: Connection) -> int:
    """
    pm_CountPapersByID Count the number of PeaTMOSS papers by their paper ID


    :param pmDB: A sqlite3.Connection to a PeaTMOSS database
    :type pbDB: Connection
    :return: The number of papers in a PeaTMOSS database
    :rtype: int
    """
    query: str = "SELECT COUNT(DISTINCT paper_id) FROM model_to_paper"
    return runOneValueSQLQuery(db=pmDB, query=query)[0]


def pm_CountPapersPerJournal(pmDB: Connection) -> Series:
    """
    pm_CountPapersPerJournal Count the number of papers per journal in PeaTMOSS

    :param pmDB: A sqlite3.Connection object to a PeaTMOSS database
    :type pmDB: Connection
    :return: A pandas.Series[int] object of the number of papers per journal
    :rtype: Series[int]
    """
    query: str = "SELECT url FROM paper"
    df: DataFrame = _createDFFromSQL(db=pmDB, query=query)
    df["url"] = df["url"].apply(_extractNetLoc)
    return df["url"].value_counts(sort=True, dropna=False)


def pm_IdentifyPapersPublishedInArXiv(pmDB: Connection) -> DataFrame:
    """
    pm_IdentifyPapersPublishedInArXiv Identify the papers in PeaTMOSS published in arXiv by DOI

    :param pmDB: A sqlite3.Connection object of a PeaTMOSS database
    :type pmDB: Connection
    :return: A pandas.DataFrame object of the relevant data for this project
    :rtype: DataFrame
    """
    query: str = "SELECT title, url FROM paper"

    pmDF: DataFrame = _createDFFromSQL(db=pmDB, query=query)
    pmDF["url"] = pmDF["url"].apply(_convertToArXivDOI)

    return pmDF[pmDF["url"].str.contains("10.48550/arxiv.")]


@click.command()
@click.option(
    "-p",
    "--peatmoss",
    "pmPath",
    type=Path,
    help="Path to PeaTMOSS database",
    required=True,
)
@click.option(
    "-o",
    "--openalex",
    "oaPath",
    type=Path,
    help="Path to OpenAlex database",
    required=True,
)
def main(pmPath: Path, oaPath: Path) -> None:
    absPMPath: Path = resolvePath(path=pmPath)
    absOAPath: Path = resolvePath(path=oaPath)

    assert isFile(path=absPMPath)
    assert isFile(path=absOAPath)

    pmDB: Connection = connectToDB(dbPath=absPMPath)
    oaDB: Connection = connectToDB(dbPath=absOAPath)

    oaPaperCountByDOI: int = oa_CountPapersByDOI(oaDB=oaDB)
    print(
        "Number of papers with DOIs in OpenAlex:",
        intcomma(value=oaPaperCountByDOI),
    )

    oaPaperCountByOAID: int = oa_CountPapersByOAID(oaDB=oaDB)
    print(
        "Number of papers with OAIDs in OpenAlex:",
        intcomma(value=oaPaperCountByOAID),
    )

    oaCitationCount: int = oa_CountCitations(oaDB=oaDB)
    print(
        "Number of citations captured in OpenAlex:",
        intcomma(value=oaCitationCount),
    )

    oaProportionOfPapersWithDOIs: float = oa_ProportionOfValidPapers(
        oaIDCount=oaPaperCountByOAID,
        oaDOICount=oaPaperCountByDOI,
    )
    print(
        "Proportion of papers with DOIs in OpenAlex:",
        f"{oaProportionOfPapersWithDOIs * 100}%",
    )

    pmPaperCountByID: int = pm_CountPapersByID(pmDB=pmDB)
    print(
        "Number of papers captured in PeaTMOSS:",
        intcomma(value=pmPaperCountByID),
    )

    pmArxivPapersInOA: int = oapm_CountPMArXivPapersInOA(
        pmDB=pmDB,
        oaDB=oaDB,
    )
    print(
        "Number of PeaTMOSS papers captured in OpenAlex that were published in arXiv:",
        intcomma(value=pmArxivPapersInOA),
    )

    pmPapersPerJournal: Series = pm_CountPapersPerJournal(pmDB=pmDB)
    print(
        "Number of papers per journal in PeaTMOSS:\n",
        pmPapersPerJournal,
    )

    oapm_arXivPMPapers: Series = oapm_CountCitationsOfArXivPMPapers(
        pmDB=pmDB,
        oaDB=oaDB,
    )
    print(
        "Number of citations per PeaTMOSS published in arXiv:\n",
        oapm_arXivPMPapers,
    )


if __name__ == "__main__":
    main()
