import json
import pickle
import sqlite3
from math import ceil
from sqlite3 import Connection, Cursor
from typing import Iterator
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from pandas import DataFrame, Series
from progress.bar import PixelBar


def getNumberOfCitations(file_path: str) -> int:
    # sqlQuery: str = "SELECT id FROM cites ORDER BY id DESC LIMIT 1"
    # conn = sqlite3.Connection(database=file_path)
    # cursor: Cursor = conn.execute(sqlQuery)
    # return cursor.fetchone()[0]
    return 113563323


def getNumberOfWorks(file_path: str) -> int:
    # sqlQuery: str = "SELECT COUNT(oa_id) FROM works"
    # conn = sqlite3.Connection(database=file_path)
    # cursor: Cursor = conn.execute(sqlQuery)
    # return cursor.fetchone()[0]
    return 13435534


def createDFGeneratorFromSQL(
    file_path: str, column: str, table_from_db: str, chunksize: int
) -> Iterator[DataFrame]:  # incorporate across functions
    conn = sqlite3.Connection(database=file_path)
    query = f"SELECT {column} FROM {table_from_db}"
    df = pd.read_sql_query(query, con=conn, chunksize=chunksize)
    return df


def create_df_from_db(
    file_path: str, column: str, table_from_db: str
) -> DataFrame:  # incorporate across functions
    conn = sqlite3.Connection(database=file_path)
    query = f"SELECT {column} FROM {table_from_db}"
    df = pd.read_sql_query(query, con=conn)
    return df


def count_papers_in_db(column: str, table: str, conn: Connection) -> int:
    query = f"SELECT COUNT(DISTINCT {column} FROM {table}"
    cursor: Cursor = conn.execute(query)
    return cursor.fetchone()[0]


def proportion_PM_papers_in_OA(PMconn: Connection):
    # num of unique PM papers in OA db
    PMproportion = count_papers_in_db(
        column="paper_id", table="model_to_paper", conn=PMconn
    )
    OAproportion = count_papers_in_db(column="doi", table="works", conn=OAconn)
    print("Proportion of PM papers in OA dataset: ", PMproportion / OAproportion)


# papers per publication for PM
def count_papers_per_journalPM(conn: Connection, db_file_path: str):
    def extract_base_url(url):
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    df = create_df_from_db(db_file_path, "*", "paper")
    print(df["url"].apply(extract_base_url).value_counts())


def count_PMarxiv_papers_in_OA(
    PMconn: Connection, OAconn: Connection
):  # option - do for other major publications in PM
    paper_count = 0

    query = f"SELECT * FROM paper"
    PM_df = create_df_from_db(PM_file_path, "*", "paper")  # PM_df: Dataframe = syntax?
    arxiv_series: Series = PM_df["url"].str.lower()
    arxiv_formatted: Series = arxiv_series.str.replace(
        pat="https://arxiv.org/abs/", repl="10.48550/arxiv."
    )

    OA_query = f"SELECT oa_id, doi FROM works"
    OA_df: Iterator[DataFrame] = pd.read_sql_query(
        OA_query, con=OAconn, chunksize=10000
    )

    df: DataFrame
    for df in OA_df:
        df_lower: Series = df["doi"].str.lower()
        paper_count += df["doi"][df_lower.isin(arxiv_formatted)].size
    return paper_count


# access PM titles of papers without URLs, cross-ref with OA titles and access those papers' DOIs//new vis for publications of papers with no URLs for underreported/misreported vis
# find publications for entries in OA that have no DOI that correspond to entries in PM without URLs by brute force publication identification, still want to vis in underreported/general
def unknown_URL_PMpapers_getting_DOI_from_OA(PMconn: Connection, OAconn: Connection):
    print("Hello WOrld")
    PMquery = f"SELECT title, url FROM paper WHERE (url IS NULL OR url = '')"
    PM_titles_nullURLs_df = pd.read_sql_query(PMquery, PMconn)

    print("hello WOrld")
    OAquery = f"SELECT doi, title FROM works WHERE (doi IS NOT NULL)"
    OA_titles_df = pd.read_sql_query(OAquery, OAconn)

    PM_titles_nullURLs_df["title_normalized"] = (
        PM_titles_nullURLs_df["title"].str.strip().str.lower()
    )
    OA_titles_df["title_normalized"] = OA_titles_df["title"].str.strip().str.lower()

    # Filter OA DataFrame based on  normalized titles in PM_df
    print("Hello Oworld")
    filtered_df = OA_titles_df[
        OA_titles_df["title_normalized"].isin(PM_titles_nullURLs_df["title_normalized"])
    ]

    # return df with redirected urls based on doi
    doi_array = filtered_df["doi"].to_numpy()
    doi_URLs_df = pd.DataFrame(columns=["url"])
    doi_URLs_list = []
    for doi in doi_array:
        initial_url = "https://doi.org/" + doi
        redirection_with_doi = requests.get(initial_url, allow_redirects=True)
        final_url = redirection_with_doi.url
        doi_URLs_list.append(pd.DataFrame({"url": [final_url]}))
    doi_URLs_df = pd.concat(doi_URLs_list, ignore_index=True)

    def extract_base_url(url):
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    print(doi_URLs_df["url"].apply(extract_base_url).value_counts())


# Citation relationships


# total citations across the board
def generate_total_cites():
    OA_works_column = "work"

    OA_works_df = create_df_from_db(OA_file_path, OA_works_column, "cites")

    total_cites = OA_works_df.shape[0]
    print("Total number of citations from any work to any other work: ", total_cites)


# tally of citations made in OA that correspond to papers in PM - total number and organized dataframe based on number of unique matches
def total_citations_of_PM_papers():
    def standardize_columns(df):
        return df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

    titleStore: list[DataFrame] = []
    citesStore: list[DataFrame] = []

    chunksize: int = 1000
    numberOfCitations: int = getNumberOfCitations(OA_file_path)
    numberOfWorks: int = getNumberOfWorks(file_path=OA_file_path)

    PM_titles_df = create_df_from_db(PM_file_path, "title", "paper")
    peatmossDF: DataFrame = standardize_columns(PM_titles_df)

    OA_titles_df: Iterator[DataFrame] = createDFGeneratorFromSQL(
        OA_file_path, "oa_id, title", "works", chunksize=chunksize
    )

    with PixelBar(
        "Iterating through OA Works table...", max=ceil(numberOfWorks / chunksize)
    ) as bar:
        df: DataFrame
        for df in OA_titles_df:
            df.replace(to_replace=" ", value=None, inplace=True)
            df.dropna(inplace=True)
            df = standardize_columns(df)
            titleStore.append(df[df["title"].isin(peatmossDF["title"])])
            bar.next()

    oaTitlesDF: DataFrame = pd.concat(objs=titleStore)

    # print(oaTitlesDF)

    oaCitesDFs: Iterator[DataFrame] = createDFGeneratorFromSQL(
        OA_file_path, "work, reference", "cites", chunksize=chunksize * 10
    )
    with PixelBar(
        "Iterating through OA Cites table... ",
        max=ceil(numberOfCitations / (chunksize * 10)),
    ) as bar:
        df: DataFrame
        for df in oaCitesDFs:
            df = standardize_columns(df)
            citesStore.append(df[df["reference"].isin(oaTitlesDF["oa_id"])])
            bar.next()

    oaCitesDF: DataFrame = pd.concat(objs=citesStore)

    oaCitesDF["reference"].value_counts().to_json(path_or_buf="OA_Citing_PM.json")

    PMcitesOAStore: list[DataFrame] = []
    with PixelBar(
        "Iterating through OA Cites table... ",
        max=ceil(numberOfCitations / (chunksize * 10)),
    ) as bar:
        df: DataFrame
        for df in oaCitesDFs:
            df = standardize_columns(df)
            PMcitesOAStore.append(df[df["work"].isin(oaTitlesDF["oa_id"])])
            bar.next()

    oaCitesDF: DataFrame = pd.concat(objs=citesStore)

    oaCitesDF["work"].value_counts().to_json(path_or_buf="PM_Citing_OA.json")

    # use name of json file and read json pandas function for stats, functionality within names of json themselves


def dataset_comparison():
    # labels = ['OpenAlex Dataset', 'PeaTMOSS Dataset']
    # values = [7885681, 1937]

    # fig, ax = plt.subplots()
    # ax.bar(labels, values)
    # ax.set_xlabel('Dataset')
    # ax.set_ylabel('Number of Unique Papers')
    # ax.set_title('Comparison of Paper Counts between OpenAlex & PeaTMOSS')
    # ax.set_yscale('log')
    # #ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    # plt.savefig('dataset_comparison.png')
    # plt.show()

    data = {
        "Dataset": ["OpenAlex Dataset", "PeaTMOSS Dataset"],
        "Number of Unique Papers": [7885681, 1937],
    }
    df = pd.DataFrame(data)
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="Dataset", y="Number of Unique Papers", data=df)

    # Set the title and labels
    bar_plot.set_title("Comparison of Paper Counts betwen OpenAlex & PeaTMOSS")
    bar_plot.set_xlabel("Dataset")
    bar_plot.set_ylabel("Number of Unique Papers")

    bar_plot.set_yscale("log")

    # Rotate x-axis labels if needed
    bar_plot.set_xticklabels(
        bar_plot.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    # Save the plot as a PNG file
    plt.savefig("dataset_comparison", bbox_inches="tight")


def PM_publication_venues():
    data = {
        "Publication": ["arXiv", "Other", "ACL Anthology", "GitHub", "Hugging Face"],
        "Unique Papers": [1151, 418, 152, 64, 57],
    }
    df = pd.DataFrame(data)
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="Publication", y="Unique Papers", data=df)

    # Set the title and labels
    bar_plot.set_title("Popular Publication Venues")
    bar_plot.set_xlabel("Publication")
    bar_plot.set_ylabel("Unique Papers")

    # Rotate x-axis labels if needed
    bar_plot.set_xticklabels(
        bar_plot.get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    # Save the plot as a PNG file
    plt.savefig("PM_publication_venues", bbox_inches="tight")


def PM_DOIs_citedby_OA(top_num_of_models: int):
    def standardize_columns(df):
        return df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

    OA_doi_df: Iterator[DataFrame] = createDFGeneratorFromSQL(
        OA_file_path, "oa_id, doi", "works", 10000
    )
    standardized_chunks = []
    for chunk in OA_doi_df:
        standardized_chunk = standardize_columns(chunk)
        standardized_chunks.append(standardized_chunk)
    OA_doi_df_stand = pd.concat(standardized_chunks)

    with open(OA_citing_PM, "r") as f:
        citation_data = json.load(f)

    # JSON -> df
    citation_df = pd.DataFrame(
        list(citation_data.items()), columns=["oa_id", "citation_count"]
    )
    citation_df_stand = standardize_columns(citation_df)

    # putting OA and JSON together based on JSON
    filtered_OA_JSON = pd.merge(OA_doi_df_stand, citation_df_stand, on="oa_id")
    filtered_OA_JSON_sort = filtered_OA_JSON.sort_values(
        by="citation_count", ascending=False
    )

    top_models_df = filtered_OA_JSON_sort.head(top_num_of_models)

    # bar plot
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="doi", y="citation_count", data=top_models_df)
    plt.title(
        "Top "
        + str(top_num_of_models)
        + " PeaTMOSS Models by Citation Count".format(top_num_of_models)
    )
    plt.xlabel("DOI")
    plt.ylabel("Number of Citations")
    plt.xticks(rotation=45, ha="center")
    plt.tight_layout()
    plt.savefig("top_PMmodels_cited_byOA", bbox_inches="tight")


OA_citing_PM = "/Users/fran-pellegrino/Desktop/ptm-reuse_academic_transactions/research_ptm-reuse-through-academic-transactions/OA_Citing_PM.json"

OA_file_path = "/Users/fran-pellegrino/Desktop/ptm-reuse_academic_transactions/research_ptm-reuse-through-academic-transactions/nature/db/feedStorage/prod.db"
OAconn = sqlite3.Connection(database=OA_file_path)


PM_file_path = "/Users/fran-pellegrino/Desktop/ptm-reuse_academic_transactions/research_ptm-reuse-through-academic-transactions/nature/db/feedStorage/PeaTMOSS.db"
PMconn = sqlite3.Connection(database=PM_file_path)


if __name__ == "__main__":
    PM_DOIs_citedby_OA(10)
    quit()

    dataset_comparison()
    quit()

    total_citations_of_PM_papers()
    quit()

    unknown_URL_PMpapers_getting_DOI_from_OA(PMconn, OAconn)

    print("PM arxiv papers in OA: ", count_PMarxiv_papers_in_OA(PMconn, OAconn))

    print(
        "PM database count: ",
        count_papers_in_db(column="paper_id", table="model_to_paper", conn=PMconn),
    )
    print(
        "OA database count: ",
        count_papers_in_db(column="doi", table="works", conn=OAconn),
    )